import os
import uuid
import subprocess
import requests
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Calm Drama Stories - Render Service")

STORAGE_DIR = "/data/storage"
os.makedirs(STORAGE_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=STORAGE_DIR), name="files")

# Railway sets this automatically on the public domain; fallback for local testing.
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

# In-memory task trackers for async operations (TTS + video render).
# NOTE: these reset if the service restarts mid-job. Fine for daily single-video use;
# for heavier parallel use, swap this for a small SQLite/Redis store later.
render_tasks = {}
tts_tasks = {}


# ============================================================
# 1. TEXT-TO-SPEECH  (Kokoro-82M, self-hosted, free)
# ============================================================
from kokoro_onnx import Kokoro
import soundfile as sf

kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")

# Confirmed available voice presets (American English) as of the kokoro-onnx model-files-v1.0 release.
VOICE_MAP = {
    "female": "af_bella",
    "male": "am_michael",
}


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None          # exact kokoro voice id, overrides `gender` if given
    gender: Optional[str] = "female"      # "female" or "male" -> mapped to a default voice
    speed: float = 1.0


def _run_tts(task_id: str, text: str, voice: str, speed: float):
    try:
        import re
        import numpy as np

        # Kokoro works best in chunks (a few thousand characters at a time) for long text.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) < 2000:
                current += " " + s
            else:
                chunks.append(current.strip())
                current = s
        if current.strip():
            chunks.append(current.strip())

        all_samples = []
        sample_rate = None
        for chunk in chunks:
            if not chunk.strip():
                continue
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang="en-us")
            sample_rate = sr
            all_samples.append(samples)

        full_audio = np.concatenate(all_samples) if len(all_samples) > 1 else all_samples[0]

        filename = f"{task_id}.wav"
        filepath = os.path.join(STORAGE_DIR, filename)
        sf.write(filepath, full_audio, sample_rate)

        tts_tasks[task_id] = {"status": "completed", "audio_url": f"{BASE_URL}/files/{filename}"}
    except Exception as e:
        tts_tasks[task_id] = {"status": "failed", "error": str(e)}


@app.post("/tts")
def generate_tts(req: TTSRequest, background_tasks: BackgroundTasks):
    voice = req.voice or VOICE_MAP.get(req.gender, "af_bella")
    task_id = str(uuid.uuid4())
    tts_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(_run_tts, task_id, req.text, voice, req.speed)
    return {"taskId": task_id}


@app.get("/tts/status")
def tts_status(taskId: str):
    return tts_tasks.get(taskId, {"status": "not_found"})


class ConcatAudioRequest(BaseModel):
    audio_urls: List[str]   # ordered list — chapter 1 first, chapter 2 next, etc.


@app.post("/concat-audio")
def concat_audio(req: ConcatAudioRequest):
    """Joins multiple chapter audio files (in the given order) into one final audio file.
    This is a fast, non-TTS operation — runs synchronously."""
    work_id = str(uuid.uuid4())
    work_dir = os.path.join(STORAGE_DIR, f"concat_{work_id}")
    os.makedirs(work_dir, exist_ok=True)

    local_paths = []
    for i, url in enumerate(req.audio_urls):
        local_path = os.path.join(work_dir, f"chapter_{i}.wav")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        local_paths.append(local_path)

    concat_list_path = os.path.join(work_dir, "concat.txt")
    with open(concat_list_path, "w") as f:
        for p in local_paths:
            f.write(f"file '{p}'\n")

    final_filename = f"{work_id}_combined.wav"
    final_path = os.path.join(STORAGE_DIR, final_filename)
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy", final_path
    ], check=True, capture_output=True)

    return {"audio_url": f"{BASE_URL}/files/{final_filename}"}


# ============================================================
# 2. SUBTITLES  (faster-whisper, self-hosted, free, word-level timestamps)
# ============================================================
from faster_whisper import WhisperModel

whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")


class TranscribeRequest(BaseModel):
    audio_url: str


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    local_path = os.path.join(STORAGE_DIR, f"transcribe_{uuid.uuid4()}.wav")
    r = requests.get(req.audio_url)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)

    segments, _info = whisper_model.transcribe(local_path, word_timestamps=True)

    words = []
    for seg in segments:
        for w in seg.words:
            words.append({
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end": round(w.end, 3),
            })

    os.remove(local_path)
    return {"words": words}


# ============================================================
# 3. VIDEO RENDER  (FFmpeg, self-hosted, free — images + audio + subtitles)
# ============================================================
class RenderRequest(BaseModel):
    story_title: str
    images: List[dict]           # [{ "chapter_number": 1, "image_url": "..." }, ...]
    audio_url: str
    subtitle_words: List[dict]   # [{ "word": "...", "start": 0.1, "end": 0.4 }, ...]
    aspect_ratio: str = "16:9"
    ken_burns: bool = True


def _format_srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _write_srt(words: List[dict], path: str, words_per_chunk: int = 5):
    """Groups words into short chunks for big, catchy on-screen captions."""
    chunks, chunk = [], []
    for w in words:
        chunk.append(w)
        if len(chunk) >= words_per_chunk:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)

    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            start, end = c[0]["start"], c[-1]["end"]
            text = " ".join(w["word"] for w in c).upper()
            f.write(f"{i + 1}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n{text}\n\n")


def _run_render(task_id: str, payload: dict):
    try:
        work_dir = os.path.join(STORAGE_DIR, task_id)
        os.makedirs(work_dir, exist_ok=True)

        # --- download voiceover audio ---
        audio_path = os.path.join(work_dir, "audio.mp3")
        r = requests.get(payload["audio_url"])
        r.raise_for_status()
        with open(audio_path, "wb") as f:
            f.write(r.content)

        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]).decode().strip())

        images = sorted(payload["images"], key=lambda x: x["chapter_number"])
        n = len(images)
        per_image_duration = duration / n

        # --- download scene images ---
        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(work_dir, f"img_{i}.png")
            r = requests.get(img["image_url"])
            r.raise_for_status()
            with open(img_path, "wb") as f:
                f.write(r.content)
            image_paths.append(img_path)

        # --- subtitles file ---
        srt_path = os.path.join(work_dir, "subs.srt")
        _write_srt(payload["subtitle_words"], srt_path)

        # --- Ken Burns pan/zoom per image, then concat ---
        fps = 25
        w, h = (1920, 1080) if payload.get("aspect_ratio", "16:9") == "16:9" else (1080, 1920)
        segment_paths = []
        for i, img_path in enumerate(image_paths):
            seg_path = os.path.join(work_dir, f"seg_{i}.mp4")
            frames = max(int(per_image_duration * fps), fps)
            zoom_in = (i % 2 == 0)
            if zoom_in:
                zoompan = f"zoompan=z='min(zoom+0.0006,1.12)':d={frames}:s={w}x{h}:fps={fps}"
            else:
                zoompan = f"zoompan=z='if(lte(on,1),1.12,max(1.0,zoom-0.0006))':d={frames}:s={w}x{h}:fps={fps}"
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1", "-i", img_path,
                "-vf", f"scale={w*2}:{h*2},{zoompan}",
                "-t", str(per_image_duration),
                "-pix_fmt", "yuv420p", seg_path
            ], check=True, capture_output=True)
            segment_paths.append(seg_path)

        concat_list_path = os.path.join(work_dir, "concat.txt")
        with open(concat_list_path, "w") as f:
            for p in segment_paths:
                f.write(f"file '{p}'\n")

        concat_video_path = os.path.join(work_dir, "concat.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c", "copy", concat_video_path
        ], check=True, capture_output=True)

        # --- add voiceover audio + burn subtitles ---
        final_path = os.path.join(work_dir, "final.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", concat_video_path, "-i", audio_path,
            "-vf",
            f"subtitles={srt_path}:force_style='FontName=Arial Black,FontSize=26,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Alignment=2'",
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-shortest", final_path
        ], check=True, capture_output=True)

        final_filename = f"{task_id}_final.mp4"
        final_dest = os.path.join(STORAGE_DIR, final_filename)
        os.replace(final_path, final_dest)

        render_tasks[task_id] = {"status": "completed", "video_url": f"{BASE_URL}/files/{final_filename}"}

    except subprocess.CalledProcessError as e:
        render_tasks[task_id] = {"status": "failed", "error": e.stderr.decode()[-800:] if e.stderr else str(e)}
    except Exception as e:
        render_tasks[task_id] = {"status": "failed", "error": str(e)}


@app.post("/render")
def submit_render(req: RenderRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    render_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(_run_render, task_id, req.dict())
    return {"taskId": task_id}


@app.get("/render/status")
def render_status(taskId: str):
    return render_tasks.get(taskId, {"status": "not_found"})


@app.get("/")
def health():
    return {"status": "ok"}
