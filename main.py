import os
import re
import json
import uuid
import shutil
import tempfile
import subprocess
import urllib.request
import requests
import numpy as np
from typing import List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- local modules, loaded defensively -------------------------------------
# A missing or broken local module must NOT stop the whole service from
# booting. If it did, one bad file takes down /tts, /render, /transcribe and
# everything else with it, and Railway shows only "Crashed" with no clue why.
# Instead we record the failure and report it on the health endpoint.
_MODULE_ERRORS = {}

try:
    import visuals
except Exception as _e:
    visuals = None
    _MODULE_ERRORS["visuals"] = f"{type(_e).__name__}: {_e}"

try:
    import qc_gate
except Exception as _e:
    qc_gate = None
    _MODULE_ERRORS["qc_gate"] = f"{type(_e).__name__}: {_e}"

try:
    import thumbnail
except Exception as _e:
    thumbnail = None
    _MODULE_ERRORS["thumbnail"] = f"{type(_e).__name__}: {_e}"

app = FastAPI(title="Calm Drama Stories - Render Service")

STORAGE_DIR = "/data/storage"
os.makedirs(STORAGE_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=STORAGE_DIR), name="files")

# Railway sets this automatically on the public domain; fallback for local testing.
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

# Directory this file lives in — used to find bed.mp3 beside main.py (R008).
_HERE_MAIN = os.path.dirname(os.path.abspath(__file__))

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
# MUSIC BED  — validated, and generated if it is missing
# ============================================================
# A render died with "Failed to find two consecutive MPEG audio frames" on
# /app/bed.mp3. The file was present but not decodable - it had been mangled
# somewhere between being downloaded and being committed. Forty minutes of
# rendering was lost to a decorative audio track.
#
# Two changes follow from that:
#   1. The bed is probed before use. Anything ffmpeg cannot open is ignored,
#      and the render continues without it.
#   2. If no usable bed exists, one is synthesised here with ffmpeg. There is
#      then no file to download, upload, or corrupt.
#
# The generated bed: Am - F - C - Am, 45s per chord with 5s crossfades, and a
# plucked note every 3 seconds. Every frequency is a multiple of 0.05 Hz, so
# 180 seconds is a whole number of cycles for each one and the file loops with
# no click and no dip at the join.

_BED_CACHE = os.path.join(STORAGE_DIR, "_generated_bed.mp3")


def _bed_is_playable(path: str) -> bool:
    """True only if ffmpeg can actually decode it. Existence is not enough."""
    if not path or not os.path.exists(path) or os.path.getsize(path) < 10000:
        return False
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type:format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, timeout=30,
        )
        if r.returncode != 0:
            return False
        out = r.stdout.decode("utf-8", "replace")
        if "audio" not in out:
            return False
        # A badly truncated file can still present a readable header. Looping a
        # two-second fragment under a seven-minute video is worse than no bed.
        for tok in out.split():
            try:
                if float(tok) >= 5.0:
                    return True
            except ValueError:
                continue
        return False
    except Exception:
        return False


def _generate_bed(dest: str) -> bool:
    """Synthesise the ambient bed. Returns True if dest is now playable."""
    m = "mod(t\,60)"
    am = f"(clip((15-{m})/2\,0\,1)+clip(({m}-45)/2\,0\,1))"
    fg = f"(clip(({m}-13)/2\,0\,1)*clip((30-{m})/2\,0\,1))"
    cg = f"(clip(({m}-28)/2\,0\,1)*clip((45-{m})/2\,0\,1))"

    def pluck(offset):
        return (f"(0.08+1.0*clip(mod(t-{offset}\,12)/0.10\,0\,1)"
                f"*exp(-mod(t-{offset}\,12)/1.3))")

    p0, p3, p6, p9 = pluck(0), pluck(3), pluck(6), pluck(9)
    notes = [
        (220, 0.30, am, p0), (261.65, 0.30, am, p3),
        (329.65, 0.28, am, p6), (440, 0.18, am, p9),
        (174.60, 0.28, fg, p0), (220, 0.30, fg, p3),
        (261.65, 0.28, fg, p6), (349.20, 0.20, fg, p9),
        (261.65, 0.30, cg, p0), (329.65, 0.28, cg, p3),
        (392, 0.26, cg, p6), (523.25, 0.16, cg, p9),
    ]
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for freq, _, _, _ in notes:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration=180:sample_rate=48000"]
    parts, labels = [], []
    for i, (_, vol, gate, pk) in enumerate(notes):
        parts.append(f"[{i}:a]volume='{vol}*{gate}*{pk}':eval=frame[n{i}]")
        labels.append(f"[n{i}]")
    parts.append("".join(labels) + f"amix=inputs={len(notes)}:normalize=0[mix]")
    parts.append("[mix]lowpass=f=3200,aformat=channel_layouts=stereo[out]")

    raw = dest + ".raw.wav"
    cmd += ["-filter_complex", ";".join(parts), "-map", "[out]",
            "-c:a", "pcm_s16le", raw]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)

        # Measure, then apply the exact correction. A hard-coded gain was wrong
        # by 36 dB the first time it was written; the file has to be measured.
        # A fixed gain, not loudnorm: loudnorm varies the gain over time, which
        # would leave a step at the loop point.
        meas = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", raw, "-af", "ebur128",
             "-f", "null", "-"],
            capture_output=True, timeout=300,
        ).stderr.decode("utf-8", "replace")

        current = None
        for line in meas.splitlines():
            t = line.strip()
            if t.startswith("I:") and "LUFS" in t:
                try:
                    current = float(t.split()[1])
                except (ValueError, IndexError):
                    pass
        gain = 0.0 if current is None else round(-22.0 - current, 2)
        gain = max(-40.0, min(40.0, gain))

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", raw,
             "-af", f"volume={gain}dB",
             "-c:a", "libmp3lame", "-b:a", "160k", "-ar", "48000", dest],
            check=True, capture_output=True, timeout=300,
        )
    except Exception:
        for p in (dest, raw):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return False
    finally:
        if os.path.exists(raw):
            try:
                os.remove(raw)
            except OSError:
                pass
    return _bed_is_playable(dest)


def resolve_bed() -> Optional[str]:
    """The bed to mix in, or None if there should not be one."""
    supplied = os.environ.get("MUSIC_BED", os.path.join(_HERE_MAIN, "bed.mp3"))
    if _bed_is_playable(supplied):
        return supplied
    if _bed_is_playable(_BED_CACHE):
        return _BED_CACHE
    if _generate_bed(_BED_CACHE):
        return _BED_CACHE
    return None


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
    # Each image is either a URL to fetch, or a visual_spec this service draws
    # itself. Drawing locally is preferred: the graphic is built from the same
    # design system every time, and there is no external chart service deciding
    # what our video looks like.
    #   { "chapter_number": 1, "visual_spec": {...}, "duration": 12.5, "camera_motion": "none" }
    #   { "chapter_number": 1, "image_url": "https://...", "duration": 12.5 }
    images: List[dict]
    audio_url: str
    subtitle_words: List[dict]   # [{ "word": "...", "start": 0.1, "end": 0.4 }, ...]
    aspect_ratio: str = "16:9"
    ken_burns: bool = True
    subtitle_config: dict = {}      # R005 — max_chars / break_on, optional
    audio_master: dict = {}         # R008 — loudness_lufs / true_peak_dbtp / music_bed, optional
    safe_zone_bottom: int = 200     # R007 — reserved for future use by visuals.py callers


def _format_ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    cs = int((s - int(s)) * 100)  # centiseconds
    return f"{h:d}:{m:02d}:{int(s):02d}.{cs:02d}"


def _write_ass(words: List[dict], path: str, w: int, h: int,
               words_per_chunk: int = 5, max_chars: int = 32,
               phrase_break: bool = True):
    """Writes a self-contained .ass subtitle file with an explicit PlayResX/
    PlayResY matching the actual video frame. This is the fix for the
    'gigantic subtitles' bug: when a plain .srt is burned via ffmpeg's
    subtitles filter, ffmpeg converts it to ASS internally and has to GUESS
    the canvas size — that guess does not reliably match the real video
    resolution, so the font ends up wildly oversized or undersized. Writing
    the .ass ourselves removes the guesswork entirely: what we declare here
    is exactly what libass renders against.

    R005: chunks break on phrase boundaries and never split a number. Cutting
    every N words put "$1" on one card and ",000" on the next - and the figure
    is the whole reason the section exists.
    """
    def _joins_number(prev_word, next_word):
        """True when these two words are two halves of one figure."""
        a = str(prev_word or "").strip()
        b = str(next_word or "").strip()
        # "$1" + ",000"  |  "1" + ",000"  |  "$1,000" + ".50"
        if re.search(r"\d$", a) and re.match(r"^[,.]\d", b):
            return True
        # "twelve" + "hundred" style pairs read as one figure too
        if re.search(r"\d$", a) and re.match(r"^(hundred|thousand|million|percent|dollars?)\b", b, re.I):
            return True
        return False

    def _ends_sentence(word):
        return bool(re.search(r"[.!?]$", str(word or "").strip()))

    def _ends_clause(word):
        return bool(re.search(r"[,;:]$", str(word or "").strip()))

    # A card should never end on one of these. "AND PROPERTY TAX IS ONE LINE ON"
    # leaves the reader hanging on a preposition; the word belongs with what
    # follows it.
    _DANGLING = set((
        "a an the and or but of to in on at by for from with as is are was were "
        "that this it its his her their our your my than then so if when while"
    ).split())

    chunks, chunk, chars = [], [], 0
    for i, word in enumerate(words):
        token = str(word.get("word", "")).strip()
        nxt = words[i + 1] if i + 1 < len(words) else None

        chunk.append(word)
        chars += len(token) + 1

        if not chunk:
            continue

        # never cut a figure in half
        if nxt is not None and _joins_number(token, nxt.get("word")):
            continue

        # A sentence end ALWAYS ends the card, however short it is.
        #
        # This used to require three words first, so a two-word sentence could
        # not close a card and the next sentence ran on into it:
        #     "TWENTY-SEVEN THIRTY A YEAR IN PROPERTY"
        #     "TAX. SO DOES MOVING SOUTH REALLY"
        # - one card carrying the tail of one sentence and the head of the
        # next, with "PROPERTY TAX" torn in half across the join.
        if phrase_break and _ends_sentence(token):
            chunks.append(chunk)
            chunk, chars = [], 0
            continue

        at_clause = phrase_break and _ends_clause(token) and (len(chunk) >= 3 or chars >= 20)
        full = chars >= max_chars or len(chunk) >= words_per_chunk + 2

        if at_clause or full:
            # do not leave the card hanging on a function word
            if (full and not at_clause and len(chunk) > 2
                    and token.lower().strip(".,;:!?") in _DANGLING):
                held = chunk.pop()
                chunks.append(chunk)
                chunk = [held]
                chars = len(str(held.get("word", "")).strip()) + 1
            else:
                chunks.append(chunk)
                chunk, chars = [], 0

    if chunk:
        chunks.append(chunk)

    # ---- merge the orphans ------------------------------------------------
    # Cutting at max_chars can land inside a noun phrase and leave a card
    # holding almost nothing: "...A YEAR IN PROPERTY" followed by a card that
    # reads only "TAX." Four characters is not a subtitle, it is a flicker.
    #
    # Rather than encode grammar, anything too small to be worth its own card
    # is folded back into the neighbour it reads with - preferring the card it
    # came from, and only when the merged card stays a comfortable length.
    def _len(c):
        return sum(len(str(w.get("word", "")).strip()) + 1 for w in c) - 1

    MIN_CARD = 14
    MERGE_CEILING = max_chars + 12
    merged = []
    for c in chunks:
        if (merged and _len(c) < MIN_CARD
                and _len(merged[-1]) + _len(c) + 1 <= MERGE_CEILING):
            merged[-1] = merged[-1] + c
        else:
            merged.append(c)
    # a tiny FIRST card has no previous neighbour, so it joins the next one
    if len(merged) > 1 and _len(merged[0]) < MIN_CARD and _len(merged[0]) + _len(merged[1]) + 1 <= MERGE_CEILING:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    chunks = merged

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



# AAC encoding pushes the true peak back up by a few tenths of a dB, so the
# limiter has to aim below the target or the finished file lands over the line.
# Measured through the real chain (loudnorm -> alimiter -> aac 192k):
#     0.0 dB headroom -> -0.85 dBTP, 29 clipped samples   fails R008c
#     0.5 dB          -> -1.31 dBTP, 0 clipped            passes
#     1.0 dB          -> -1.69 dBTP, 0 clipped            passes, with margin
#     2.0 dB          -> -2.86 dBTP, 0 clipped            quieter than it needs
_CODEC_HEADROOM_DB = 1.0


def _lim(tp_db: float) -> float:
    """Linear amplitude for alimiter's `limit`, with codec headroom."""
    return round(10 ** ((float(tp_db) - _CODEC_HEADROOM_DB) / 20.0), 4)


def _audio_report(path: str) -> dict:
    """Measure the finished file and say what the audio actually came out as.

    A render was rejected by the QC gate for 1889 clipped samples and +3.1 dBTP
    on a file that had supposedly been through loudnorm. There was no way to
    tell from the outside whether the mastering had run at all, so the next step
    was a twenty-minute re-render to find out. The service now reports its own
    work instead.
    """
    rep = {}
    try:
        v = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, timeout=600,
        ).stderr.decode("utf-8", "replace")
        m = re.search(r"max_volume:\s*(-?[\d.]+) dB", v)
        if m:
            rep["max_dbfs"] = float(m.group(1))
        m = re.search(r"histogram_0db:\s*(\d+)", v)
        rep["clipped_samples"] = int(m.group(1)) if m else 0

        l = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
             "-af", "loudnorm=I=-14:TP=-1:print_format=json", "-f", "null", "-"],
            capture_output=True, timeout=600,
        ).stderr.decode("utf-8", "replace")
        m = re.search(r'"input_i"\s*:\s*"(-?[\d.]+)"', l)
        if m:
            rep["lufs"] = float(m.group(1))
        m = re.search(r'"input_tp"\s*:\s*"(-?[\d.]+)"', l)
        if m:
            rep["true_peak_dbtp"] = float(m.group(1))
    except Exception as e:
        rep["error"] = str(e)[:200]
    return rep


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

        # --- obtain scene images: draw a spec, or fetch a URL ---
        image_paths = []
        for i, img in enumerate(images):
            img_path = os.path.join(work_dir, f"img_{i}.png")
            spec = img.get("visual_spec")
            if spec:
                if visuals is None:
                    raise RuntimeError(
                        "visuals module failed to import: "
                        + _MODULE_ERRORS.get("visuals", "unknown")
                    )
                visuals.render_png(spec, img_path)
            else:
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
        sc = payload.get("subtitle_config") or {}
        _write_ass(payload["subtitle_words"], ass_path, w, h,
                   words_per_chunk=5,
                   max_chars=int(sc.get("max_chars", 32)),
                   phrase_break=str(sc.get("break_on", "phrase")) == "phrase")

        # Build the gradient overlay once (reused for every segment).
        gradient_path = os.path.join(work_dir, "gradient.png")
        # The gradient exists so captions stay readable. It used to start at 55%
        # of frame height, which put a wash over the lower half of every chart.
        # Graphics are drawn to keep meaning above 80%, so the gradient starts
        # there and stays lighter.
        gradient_alpha_expr = f"if(gte(Y,H*0.78),(Y-H*0.78)/(H*0.22)*205,0)"
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
            # Camera motion is decided per shot by the caller, not by position.
            # Ken Burns on a chart is the single worst thing you can do to one:
            # it drifts the figures out of frame and makes a clean graphic look
            # like a phone video of a monitor. Charts, stats and cards hold
            # perfectly still. Only photos and maps move, and only gently.
            motion = str(images[i].get("camera_motion") or "none").lower()
            if motion == "slow_zoom":
                chain = (f"scale={w*2}:{h*2},"
                         f"zoompan=z='min(zoom+0.00035,1.06)':d={frames}:s={w}x{h}:fps={fps}")
            elif motion == "slow_zoom_out":
                chain = (f"scale={w*2}:{h*2},"
                         f"zoompan=z='if(lte(on,1),1.06,max(1.0,zoom-0.00035))':d={frames}:s={w}x{h}:fps={fps}")
            else:
                chain = f"scale={w}:{h}:force_original_aspect_ratio=decrease," \
                        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0f172a,fps={fps}"
            subprocess.run([
                "ffmpeg", "-y",
                "-loop", "1", "-i", img_path,
                "-loop", "1", "-i", gradient_path,
                "-filter_complex",
                f"[0:v]{chain}[zoomed];[zoomed][1:v]overlay=0:0[out]",
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

        # --- add voiceover + music bed, master to broadcast loudness, burn subs ---
        # R008: the published video peaked at 0.0 dBFS with 271 clipped samples
        # and sat at -20.4 LUFS. YouTube normalises to -14, so it played quiet
        # against everything beside it in the sidebar.
        final_path = os.path.join(work_dir, "final.mp4")
        master = payload.get("audio_master") or {}
        lufs = float(master.get("loudness_lufs", -14))
        tp = float(master.get("true_peak_dbtp", -1))
        want_bed = bool(master.get("music_bed", True))

        # resolve_bed() probes the file and falls back to a generated one, so a
        # missing or corrupt bed can never take the render down with it.
        bed_path = resolve_bed() if want_bed else None
        use_bed = bool(bed_path)

        if use_bed:
            # Default -42 => -28 dB on the bed, which measures about 20 dB under
            # the voice in the finished file: felt, not heard. The old default of
            # -22 left it only 5.5 dB down, where the music competes with the
            # narration instead of supporting it.
            bed_db = float(master.get("music_bed_lufs", -42)) - lufs   # relative to voice
            cmd = [
                "ffmpeg", "-y",
                "-i", concat_video_path,
                "-i", audio_path,
                "-stream_loop", "-1", "-i", bed_path,
                "-filter_complex",
                f"[2:a]volume={bed_db}dB[bed];"
                f"[1:a][bed]amix=inputs=2:duration=first:dropout_transition=0[mix];"
                f"[mix]loudnorm=I={lufs}:TP={tp}:LRA=11,"
                f"alimiter=limit={_lim(tp)}:attack=5:release=50:level=disabled[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-vf", f"ass={ass_path}",
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", final_path,
            ]
        else:
            # No bed file present - still normalise, still stop the clipping.
            cmd = [
                "ffmpeg", "-y",
                "-i", concat_video_path,
                "-i", audio_path,
                "-filter_complex",
                f"[1:a]loudnorm=I={lufs}:TP={tp}:LRA=11,"
                f"alimiter=limit={_lim(tp)}:attack=5:release=50:level=disabled[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-vf", f"ass={ass_path}",
                "-c:v", "libx264", "-preset", "veryfast",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", final_path,
            ]

        subprocess.run(cmd, check=True, capture_output=True)

        final_filename = f"{task_id}_final.mp4"
        final_dest = os.path.join(STORAGE_DIR, final_filename)
        os.replace(final_path, final_dest)

        render_tasks[task_id] = {
            "status": "completed",
            "video_url": f"{BASE_URL}/files/{final_filename}",
            # What the audio path actually did, measured on the output file.
            "audio": {
                "mastered": True,
                "target_lufs": lufs,
                "target_dbtp": tp,
                "limiter": True,
                "music_bed": bed_path or None,
                "bed_gain_db": (bed_db if use_bed else None),
                "measured": _audio_report(final_dest),
            },
            "shots": len(image_paths),
            "duration_seconds": round(duration, 1),
        }
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


class VisualPreviewRequest(BaseModel):
    spec: dict


@app.post("/visual")
def visual_preview(req: VisualPreviewRequest):
    """Draw one graphic and return the PNG directly.
    Exists so a visual can be checked in a browser in a second, instead of
    waiting ten minutes for a render to find out a label was cut off.
    """
    if visuals is None:
        raise HTTPException(500, "visuals module failed to import: "
                                 + _MODULE_ERRORS.get("visuals", "unknown"))
    tmp = os.path.join(STORAGE_DIR, f"preview_{uuid.uuid4()}.png")
    try:
        visuals.render_png(req.spec, tmp)
        with open(tmp, "rb") as f:
            data = f.read()
        return Response(content=data, media_type="image/png")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class ThumbnailRequest(BaseModel):
    entity_a: str
    entity_b: str
    value_a: float = 0
    value_b: float = 0
    subject: str = "cost of living"
    hook: str = ""            # 2-4 words, a question, never the answer
    photo_query_a: str = ""   # override the derived Pexels query
    photo_query_b: str = ""
    layout: str = ""          # force split/diagonal/hero; normally left empty


@app.post("/thumbnail")
def make_thumbnail(req: ThumbnailRequest):
    """Draw the YouTube thumbnail and return the PNG.

    Separate from /visual because the constraints are different: an in-video
    card is read at full screen for five seconds, a thumbnail is read at 120
    pixels in a fraction of one. thumbnail.py enforces its own minimum text
    size and measures every string against the real font before drawing.

    `validate` is returned rather than raised on: a thumbnail with one label
    slightly wide is still far better than no thumbnail at all, and the
    warning surfaces in the n8n run so it is not silent.
    """
    if thumbnail is None:
        raise HTTPException(500, "thumbnail module failed to import: "
                                 + _MODULE_ERRORS.get("thumbnail", "unknown"))
    payload = {k: v for k, v in req.dict().items() if v not in ("", None)}
    try:
        png, layout, spec = thumbnail.render_png(payload)
    except Exception as e:
        raise HTTPException(500, f"thumbnail render failed: {type(e).__name__}: {e}")
    problems = thumbnail.validate()
    headers = {
        "X-Thumb-Layout": layout,
        "X-Thumb-Hook": spec.get("hook", ""),
        # which photos were actually used, so a wrong-looking picture can be
        # traced to its query instead of guessed at
        "X-Thumb-Query-A": spec.get("qa", "")[:120],
        "X-Thumb-Query-B": spec.get("qb", "")[:120],
        "X-Thumb-Photos": "%s/%s" % (int(bool(spec.get("pa"))), int(bool(spec.get("pb")))),
        "X-Thumb-Warnings": ("; ".join(problems)[:400] or "none"),
    }
    return Response(content=png, media_type="image/png", headers=headers)


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


# ============================================================
# 6. PUBLISH GATE (QC)  — scores a finished video against R001-R012
# ============================================================
# qc_gate.py cannot live on the n8n container: Railway wipes the filesystem on
# every redeploy, and the n8n-ffmpeg image has no Python interpreter. The render
# service has both Python and ffmpeg, so the gate lives here as an endpoint.
#
# W5's "Run QC Gate" node calls:
#     POST https://n8n-render-service-production.up.railway.app/qc
#     {"video_url": "...", "meta": {...}}
class QCRequest(BaseModel):
    video_url: str
    meta: Optional[dict] = None       # scenes / metrics / captions from W2 and W3


@app.post("/qc")
def run_qc(req: QCRequest):
    """
    Score a finished video against R001-R012.
    Returns:
        {"score": 8.6, "blocked": false, "facts": {...}, "rows": [...]}
    "blocked" is the only field W5 acts on. Anything true there is written to
    the Finished sheet as QC_FAILED and never reaches the review queue.
    """
    if qc_gate is None:
        raise HTTPException(503, "qc gate unavailable: "
                                 + _MODULE_ERRORS.get("qc_gate", "qc_gate.py not found in repo"))
    tmp = None
    try:
        # The video may be a local /files/ path or a full URL. Handle both.
        if req.video_url.startswith("http"):
            fd, tmp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            with urllib.request.urlopen(req.video_url, timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            path = tmp
        else:
            path = req.video_url
            if not os.path.exists(path):
                raise HTTPException(404, f"video not found: {path}")

        report, facts = qc_gate.run(path, req.meta or {})
        score = report.score()
        blocked = bool(report.hardfail) or score < qc_gate.T["min_score"]

        return {
            "score": score,
            "blocked": blocked,
            "facts": facts,
            "failed": [
                {"id": r["id"], "name": r["name"], "detail": r["detail"], "hard": r["hard"]}
                for r in report.failed
            ],
            "rows": report.rows,
        }
    except HTTPException:
        raise
    except Exception as e:
        # A gate that crashes must not be read as a pass.
        raise HTTPException(500, f"qc gate failed to run: {e}")
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _bed_describe():
    """What the health endpoint reports: which bed is in play, and why.

    Plain existence was misleading - it said true for a file ffmpeg could not
    open, so the render still failed after reporting healthy.
    """
    supplied = os.environ.get("MUSIC_BED", os.path.join(_HERE_MAIN, "bed.mp3"))
    if os.path.exists(supplied):
        size = os.path.getsize(supplied)
        if _bed_is_playable(supplied):
            return {"source": "supplied", "path": supplied, "bytes": size, "ok": True}
        return {"source": "supplied", "path": supplied, "bytes": size, "ok": False,
                "note": "file present but ffmpeg cannot decode it - a generated bed "
                        "is used instead. Re-upload it, or just leave it: the "
                        "generated one is the same music."}
    if _bed_is_playable(_BED_CACHE):
        return {"source": "generated", "path": _BED_CACHE, "ok": True}
    return {"source": "generated-on-first-render", "ok": True}


@app.get("/")
def health():
    """Health check that actually tells you something.

    If a local module failed to import, the service still runs — this endpoint
    names the file and the exact error, so a broken deploy is one curl away
    from being diagnosed instead of a silent "Crashed" in the dashboard.
    """
    return {
        "status": "ok" if not _MODULE_ERRORS else "degraded",
        "modules": {
            "visuals": "ok" if visuals is not None else _MODULE_ERRORS.get("visuals"),
            "qc_gate": "ok" if qc_gate is not None else _MODULE_ERRORS.get("qc_gate"),
            "thumbnail": "ok" if thumbnail is not None else _MODULE_ERRORS.get("thumbnail"),
        },
        "music_bed": _bed_describe(),
        "pexels_key": bool(os.environ.get("PEXELS_API_KEY")),
        # names the face the thumbnails will actually draw with, so a missing
        # font package shows up here instead of in a published thumbnail
        "thumbnail_font": (getattr(thumbnail, "_FP", None) or "NONE")
                          if thumbnail is not None else "module not loaded",
    }
