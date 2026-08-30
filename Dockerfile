FROM python:3.11-slim

# FFmpeg + ffprobe + wget for downloading model files
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-xlib-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-dejavu-core \
    fonts-roboto \
 && rm -rf /var/lib/apt/lists/*

# fonts-roboto is for the thumbnails. Without it the image has only DejaVu,
# thumbnail.py silently falls back to it, and the layout is measured against
# one face while cairosvg draws another - text that fitted locally runs off
# the frame on Railway. /  reports "thumbnail_font" so the deployed face can
# be confirmed rather than assumed.

RUN pip install --no-cache-dir -r requirements.txt

# Download Kokoro-82M model files (int8 = smaller/faster, good for CPU-only hosting)
RUN wget -q -O kokoro-v1.0.int8.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx && \
    wget -q -O voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# Copy the whole repo, not a hand-written list of filenames.
#
# This line used to read:
#     COPY main.py visuals.py us_paths.json .
# qc_gate.py was added to the repo and silently never reached the image, so
# /qc answered ModuleNotFoundError while the file sat there in GitHub looking
# correct. bed.mp3 would have failed the same way, and so would the next file.
#
# It sits below the pip install and the model download on purpose: those two
# expensive layers stay cached, and only this cheap layer rebuilds when the
# code changes.
COPY . .

RUN mkdir -p /data/storage

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
