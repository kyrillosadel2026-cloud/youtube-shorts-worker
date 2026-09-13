FROM n8nio/n8n:latest

USER root

# n8n 2.x images are Alpine-based. Restore apk when missing, then install media/python tools.
COPY --from=alpine:3.22 /sbin/apk /sbin/apk
COPY --from=alpine:3.22 /lib/apk /lib/apk
COPY --from=alpine:3.22 /usr/lib/libapk* /usr/lib/

RUN apk add --no-cache \
    python3 py3-pip ffmpeg font-dejavu ca-certificates curl bash

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /data/yt_shorts
COPY video_analyzer.py /data/yt_shorts/video_analyzer.py
COPY short_editor.py /data/yt_shorts/short_editor.py
COPY run_pipeline.py /data/yt_shorts/run_pipeline.py
RUN chmod +x /data/yt_shorts/*.py && chown -R node:node /data/yt_shorts

USER node
