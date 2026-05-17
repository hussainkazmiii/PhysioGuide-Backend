FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the MediaPipe pose landmarker model at build time
# so the server starts instantly without any runtime downloads.
RUN python -c "\
import urllib.request; \
url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'; \
urllib.request.urlretrieve(url, 'pose_landmarker_lite.task'); \
print('MediaPipe model downloaded successfully')"

# Copy application source (training CSVs and artifacts are excluded via .dockerignore)
COPY . .

# Railway injects PORT at runtime; default to 8000 for local docker run
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]