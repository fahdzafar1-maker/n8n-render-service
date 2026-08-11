import os
import re
import uuid
import shutil
import subprocess
import requests
import numpy as np
from typing import List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
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
transcribe_tasks = {}


# ============================================================
# 0. SHARED HELPER — robust file download (handles Google Drive
#    large-file "can't scan for viruses" warning page transparently)
# ============================================================
def download_file(url: str, dest_path: str, timeout: int = 600):
    """Downloads a file to dest_path.

    Google Drive serves files over 100MB behind a "can't scan for viruses"
    interstitial. That page is an HTML <form> that posts back to a DIFFERENT
    endpoint (drive.usercontent.google.com/download) carrying a per-request
    `uuid` token. Rebuilding the URL by hand against the original
    drive.google.com/uc endpoint does NOT work — that token is only valid on
    the endpoint the form names, and Drive answers with 404. So: parse the
    form's action plus every hidden input, and submit exactly that.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    })

    response = session.get(url, stream=True, timeout=timeout)

    if "text/html" in response.headers.get("Content-Type", ""):
        html = response.text

        action_match = re.search(
            r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', html
        ) or re.search(r'<form[^>]+action="([^"]+)"', html)

        # every <input type="hidden" name="..." value="..."> in the page
        params = dict(
            re.findall(
                r'<input[^>]+type="hidden"[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
                html,
            )
        )

        if action_match:
            action = action_match.group(1).replace("&amp;", "&")
            # the action itself may already carry query params; merge, don't drop
            parsed = urlparse(action)
            merged = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            merged.update(params)
            merged.setdefault("confirm", "t")
            retry_url = urlunparse(parsed._replace(query=urlencode(merged)))
        else:
            # Fallback: stay on whatever URL we were actually redirected to
            # (response.url), not the original one, and just add confirm.
            parsed = urlparse(response.url)
            merged = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            merged.update(params)
            merged["confirm"] = "t"
            retry_url = urlunparse(parsed._replace(query=urlencode(merged)))

        response = session.get(retry_url, stream=True, timeout=timeout)

        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError(
                f"Google Drive kept returning an HTML page instead of the file "
                f"for {url} — check the file is shared as 'Anyone with the link'."
            )

    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

    # An HTML error page saved as audio.mp3 is the failure mode that has cost
    # us the most time; fail loudly and early instead of letting ffprobe choke.
    if os.path.getsize(dest_path) < 10000:
        raise RuntimeError(
            f"Downloaded file from {url} is only {os.path.getsize(dest_path)} bytes "
            f"— that is not the real media file."
        )


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
        download_file(url, local_path)
        local_paths.append(local_path)

    concat_list_path = os.path.join(work_dir, "concat.txt")
    with open(concat_list_path, "w") as f:
        for p in local_paths:
            f.write(f"file '{p}'\n")

    final_filename = f"{work_id}_combined.wav"
    final_path = os.path.join(STORAGE_DIR, final_filename)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c", "copy", final_path
        ], check=True, capture_output=True)
    finally:
        # Clean up per-chapter source files — only the combined file needs to stay.
        shutil.rmtree(work_dir, ignore_errors=True)

    return {"audio_url": f"{BASE_URL}/files/{final_filename}"}


# ============================================================
# 2. SUBTITLES  (faster-whisper, self-hosted, free, word-level timestamps)
# ============================================================
from faster_whisper import WhisperModel

whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")


class TranscribeRequest(BaseModel):
    audio_url: str


def _run_transcribe(task_id: str, audio_url: str):
    """Runs Whisper in the background so long audio (50+ min) doesn't hit the
    gateway request timeout (which was causing 502 Bad Gateway)."""
    local_path = os.path.join(STORAGE_DIR, f"transcribe_{task_id}.wav")
    try:
        # download with a generous timeout for large files
        download_file(audio_url, local_path, timeout=600)

        segments, _info = whisper_model.transcribe(local_path, word_timestamps=True)

        words = []
        for seg in segments:
            for w in seg.words:
                words.append({
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })

        transcribe_tasks[task_id] = {"status": "completed", "words": words}
    except Exception as e:
        transcribe_tasks[task_id] = {"status": "failed", "error": str(e), "words": []}
    finally:
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass


@app.post("/transcribe")
def transcribe(req: TranscribeRequest, background_tasks: BackgroundTasks):
    """Submit a transcription job. Returns a taskId immediately; poll /transcribe/status."""
    task_id = str(uuid.uuid4())
    transcribe_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(_run_transcribe, task_id, req.audio_url)
    return {"taskId": task_id}


@app.get("/transcribe/status")
def transcribe_status(taskId: str):
    return transcribe_tasks.get(taskId, {"status": "not_found"})


# ============================================================
# 3. VIDEO RENDER  (FFmpeg, self-hosted, free — images + audio + subtitles)
# ============================================================
class RenderRequest(BaseModel):
    story_title: str
    images: List[dict]           # [{ "chapter_number": 1, "image_url": "...", "duration": 12.5 }, ...]
    audio_url: str
    subtitle_words: List[dict]   # [{ "word": "...", "start": 0.1, "end": 0.4 }, ...]
    aspect_ratio: str = "16:9"
    ken_burns: bool = True


def _format_ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    cs = int((s - int(s)) * 100)  # centiseconds
    return f"{h:d}:{m:02d}:{int(s):02d}.{cs:02d}"


def _write_ass(words: List[dict], path: str, w: int, h: int, words_per_chunk: int = 5):
    """Writes a self-contained .ass subtitle file with an explicit PlayResX/
    PlayResY matching the actual video frame. This is the fix for the
    'gigantic subtitles' bug: when a plain .srt is burned via ffmpeg's
    subtitles filter, ffmpeg converts it to ASS internally and has to GUESS
    the canvas size — that guess does not reliably match the real video
    resolution, so the font ends up wildly oversized or undersized. Writing
    the .ass ourselves removes the guesswork entirely: what we declare here
    is exactly what libass renders against.
    """
    chunks, chunk = [], []
    for word in words:
        chunk.append(word)
        if len(chunk) >= words_per_chunk:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{int(h * 0.058)},&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,3,2,{int(w * 0.06)},{int(w * 0.06)},{int(h * 0.14)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for c in chunks:
            start, end = c[0]["start"], c[-1]["end"]
            text = " ".join(word["word"] for word in c).upper()
            f.write(f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Default,,0,0,0,,{text}\n")


def _run_render(task_id: str, payload: dict):
    try:
        work_dir = os.path.join(STORAGE_DIR, task_id)
        os.makedirs(work_dir, exist_ok=True)

        # --- download voiceover audio ---
        audio_path = os.path.join(work_dir, "audio.mp3")
        download_file(payload["audio_url"], audio_path)

        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]).decode().strip())

        images = sorted(payload["images"], key=lambda x: x["chapter_number"])
        n = len(images)

        # --- per-shot durations -------------------------------------------
        # Splitting the audio evenly across every image assumes each shot is
        # narrated for the same length of time. It is not: a rank reveal runs
        # long, a one-line aside runs short. When the caller sends a `duration`
        # on each image, those are treated as SHARES and rescaled so they add
        # up to the real audio length exactly — the pictures then stay in step
        # with the voice for the whole video.
        #
        # No durations supplied -> even split, exactly as before. That is what
        # keeps the older storytelling pipeline working without any change.
        raw = [float(img.get("duration") or 0) for img in images]
        if raw and all(r > 0 for r in raw):
            scale = duration / sum(raw)
            durations = [r * scale for r in raw]
        else:
            durations = [duration / n] * n

        # --- download scene images ---
        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(work_dir, f"img_{i}.png")
            download_file(img["image_url"], img_path)
            image_paths.append(img_path)

        # --- full-bleed layout: the scene image fills the entire frame with a
        # Ken Burns pan/zoom. A soft gradient — clear at the top, fading to black
        # toward the bottom — sits behind the subtitle area so captions stay
        # readable no matter what's in the shot, without hiding the subject
        # (which is normally framed in the upper/middle two-thirds of the image). ---
        fps = 25
        w, h = (1920, 1080) if payload.get("aspect_ratio", "16:9") == "16:9" else (1080, 1920)

        # --- subtitles file: short bursts, own PlayResX/PlayResY so the font
        # renders at the correct size against this exact frame — no guessing. ---
        ass_path = os.path.join(work_dir, "subs.ass")
        _write_ass(payload["subtitle_words"], ass_path, w, h, words_per_chunk=5)

        # Build the gradient overlay once (reused for every segment).
        gradient_path = os.path.join(work_dir, "gradient.png")
        gradient_alpha_expr = f"if(gte(Y,H*0.55),(Y-H*0.55)/(H*0.45)*190,0)"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=black:s={w}x{h}:d=1,format=yuva420p,"
                  f"geq=lum=0:cb=128:cr=128:a='{gradient_alpha_expr}'",
            "-frames:v", "1", gradient_path
        ], check=True, capture_output=True)

        segment_paths = []
        for i, img_path in enumerate(image_paths):
            # Single pass: Ken Burns pan/zoom at full frame size, with the
            # bottom gradient composited on top in the same ffmpeg call.
            seg_path = os.path.join(work_dir, f"seg_{i}.mp4")
            seg_duration = durations[i]
            frames = max(int(seg_duration * fps), fps)
            zoom_in = (i % 2 == 0)
            if zoom_in:
                zoompan = f"zoompan=z='min(zoom+0.0006,1.12)':d={frames}:s={w}x{h}:fps={fps}"
            else:
                zoompan = f"zoompan=z='if(lte(on,1),1.12,max(1.0,zoom-0.0006))':d={frames}:s={w}x{h}:fps={fps}"

            subprocess.run([
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-loop", "1", "-i", gradient_path,
                "-filter_complex",
                f"[0:v]scale={w*2}:{h*2},{zoompan}[zoomed];[zoomed][1:v]overlay=0:0[out]",
                "-map", "[out]",
                "-t", str(seg_duration),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
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

        # --- add voiceover audio + burn subtitles, bottom-centered over the
        # gradient. Bold yellow text with a black shadow — all styling is
        # already baked into the .ass file itself, so no force_style needed. ---
        final_path = os.path.join(work_dir, "final.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", concat_video_path, "-i", audio_path,
            "-vf", f"ass={ass_path}",
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
    finally:
        # Always clean up the working folder (source images, per-chapter video
        # segments, raw audio) — whether the render succeeded or failed. Only
        # the final .mp4 (saved directly under STORAGE_DIR, not work_dir) survives.
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/render")
def submit_render(req: RenderRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    render_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(_run_render, task_id, req.dict())
    return {"taskId": task_id}


@app.get("/render/status")
def render_status(taskId: str):
    return render_tasks.get(taskId, {"status": "not_found"})


# ============================================================
# 4. CLEANUP  — free up storage/compute once files are no longer needed
# ============================================================
import time


@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Delete one specific file (e.g. call this from n8n right after the
    final video has been successfully uploaded to Google Drive)."""
    path = os.path.join(STORAGE_DIR, filename)
    # guard against path traversal — only allow deleting files directly inside STORAGE_DIR
    if os.path.dirname(path) != STORAGE_DIR.rstrip("/"):
        return {"status": "error", "message": "invalid filename"}
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted", "filename": filename}
    return {"status": "not_found", "filename": filename}


class CleanupRequest(BaseModel):
    older_than_hours: float = 24.0   # delete files older than this; 0 = delete everything


@app.post("/cleanup")
def cleanup(req: CleanupRequest = CleanupRequest()):
    """Safety-net endpoint: deletes any file (and any leftover folder) sitting
    directly in STORAGE_DIR older than `older_than_hours`. Normal renders/concat
    jobs already clean up their own working folders — this catches anything that
    was left behind by a crash, a killed deploy, or an old test run."""
    cutoff = time.time() - (req.older_than_hours * 3600)
    deleted = []
    for name in os.listdir(STORAGE_DIR):
        path = os.path.join(STORAGE_DIR, name)
        try:
            if os.path.getmtime(path) < cutoff:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                deleted.append(name)
        except FileNotFoundError:
            pass
    return {"status": "ok", "deleted_count": len(deleted), "deleted": deleted}


@app.get("/storage-usage")
def storage_usage():
    """Quick check of what's currently sitting in storage, without needing the console."""
    items = []
    total_bytes = 0
    for name in os.listdir(STORAGE_DIR):
        path = os.path.join(STORAGE_DIR, name)
        if os.path.isdir(path):
            size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, files in os.walk(path) for f in files
            )
        else:
            size = os.path.getsize(path)
        total_bytes += size
        items.append({"name": name, "size_mb": round(size / (1024 * 1024), 2)})
    items.sort(key=lambda x: -x["size_mb"])
    return {"total_mb": round(total_bytes / (1024 * 1024), 2), "items": items}


# ============================================================
# 5. UPLOAD  — accept a file directly from n8n (raw binary body)
# ============================================================
# Google Drive refuses to serve files >100MB to unauthenticated servers.
# Rather than fight that, n8n downloads the file with its own OAuth
# credential and POSTs the bytes here; we hand back a plain URL that
# ffmpeg/ffprobe can fetch with zero friction.
# Streamed to disk so a 250MB upload never sits in memory.
from fastapi import Request


@app.post("/upload")
async def upload(request: Request, filename: str = "upload.bin"):
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)[-80:]
    stored_name = f"{uuid.uuid4()}_{safe_name}"
    dest_path = os.path.join(STORAGE_DIR, stored_name)

    size = 0
    with open(dest_path, "wb") as f:
        async for chunk in request.stream():
            if chunk:
                f.write(chunk)
                size += len(chunk)

    if size == 0:
        os.remove(dest_path)
        return {"status": "error", "message": "empty upload — no bytes received"}

    return {
        "status": "ok",
        "filename": stored_name,
        "bytes": size,
        "file_url": f"{BASE_URL}/files/{stored_name}",
    }


@app.get("/")
def health():
    return {"status": "ok"}
