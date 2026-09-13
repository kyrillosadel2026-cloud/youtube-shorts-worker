FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YT_SHORTS_ROOT=/tmp/yt_shorts/jobs \
    WORKER_LOG_DIR=/tmp/yt_shorts/logs \
    KEEP_JOB_FILES=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY worker_api.py run_pipeline.py video_analyzer.py short_editor.py ./

RUN mkdir -p /tmp/yt_shorts/jobs /tmp/yt_shorts/logs

EXPOSE 8080

CMD ["sh", "-c", "uvicorn worker_api:app --host 0.0.0.0 --port ${PORT:-8080}"]
