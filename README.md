# YouTube Long Video → AI Shorts → Telegram

This package creates an end-to-end n8n workflow that accepts a Telegram command such as:

```text
/shorts 3 https://youtu.be/VIDEO_ID
```

It then:

1. Downloads the YouTube video with `yt-dlp`.
2. Extracts audio with FFmpeg.
3. Splits the audio into small MP3 chunks and transcribes them with OpenAI `whisper-1` so segment timestamps are available for editing.
4. Sends timestamped transcript chunks to OpenAI for candidate detection.
5. Generates more candidates than requested, scores them, removes highly overlapping candidates, and selects the best N.
6. Renders 1080x1920 vertical Shorts with captions, a truthful hook, light visual enhancement, and loudness normalization.
7. Sends each finished MP4 back to the same Telegram chat.

## Files

- `n8n_youtube_to_shorts_telegram.json` — import this into n8n.
- `run_pipeline.py` — orchestration, YouTube download, transcription, rendering and Telegram delivery.
- `video_analyzer.py` — AI candidate generation, scoring, overlap removal and final selection.
- `short_editor.py` — standalone FFmpeg vertical editor used as a fallback/default editor.
- `Dockerfile` — example custom n8n image with Python, FFmpeg and dependencies.
- `docker-compose.example.yml` — example environment configuration.

## Important n8n requirement

This build targets **self-hosted n8n** because it uses the Execute Command node. n8n's current documentation says Execute Command is unavailable on n8n Cloud and is disabled by default starting with n8n 2.0. Set `NODES_EXCLUDE=[]` (only on a trusted self-hosted instance) and restart n8n before activating the workflow.

## Setup

1. Put these files in one folder on your n8n server.
2. Build the custom image:

```bash
docker compose -f docker-compose.example.yml build
```

3. Fill in:
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - `ALLOWED_TELEGRAM_CHAT_IDS` (recommended, to stop other people from spending your API credits through the bot)
   - `N8N_HOST`
   - `WEBHOOK_URL`
4. Start n8n.
5. In n8n, create Telegram credentials using the same bot token.
6. Import `n8n_youtube_to_shorts_telegram.json`.
7. Open **Telegram Trigger** and choose your real Telegram credential.
8. Verify **Run Shorts Pipeline** points to:

```text
/data/yt_shorts/run_pipeline.py
```

9. Activate the workflow.

## Telegram commands

Preferred:

```text
/shorts 3 https://youtu.be/VIDEO_ID
```

Also accepted:

```text
/shorts https://youtu.be/VIDEO_ID 3
```

The default maximum is 8 Shorts per job. Change `MAX_SHORTS_PER_JOB` if needed.

## AI behavior

The analyzer is explicitly instructed not to invent topics. It only selects moments supported by the timestamped transcript. It initially asks for multiple candidates, then uses weighted scoring and overlap filtering, then performs a final selection pass.

Default score weights:

- Viral potential: 30%
- Emotion: 20%
- Curiosity: 20%
- Visual potential: 15%
- Standalone context: 15%

Maximum overlap between selected clips is 15%.

## Editing behavior

The included editor is deliberately conservative and stable:

- 1080x1920 output
- center crop to vertical
- burned ASS captions
- hook text near the top for the first seconds
- slight contrast/saturation lift
- audio loudness normalization
- H.264 + AAC MP4

If you upload your previous `short_editor.py`, its more advanced editing logic can be dropped into this pipeline while keeping the analyzer and n8n workflow unchanged.

## Useful environment variables

```text
OPENAI_MODEL=gpt-5.6-luna
TRANSCRIBE_MODEL=whisper-1
AUDIO_CHUNK_SECONDS=720
YT_SHORTS_ROOT=/data/yt_shorts/jobs
SHORT_MIN_DURATION=25
SHORT_MAX_DURATION=58
MAX_SHORTS_PER_JOB=8
ALLOWED_TELEGRAM_CHAT_IDS=123456789
KEEP_JOB_FILES=1
FFMPEG_PRESET=veryfast
FFMPEG_CRF=21
```

For a GPU worker, set the Whisper device/compute type to values supported by your CUDA setup.

## Notes

- Only process videos you have permission to download and repurpose.
- Audio is split into chunks before transcription because the Transcriptions API accepts audio files up to 25 MB.
- Long videos can still require significant CPU time for video rendering.
- Telegram Bot API uploads are currently limited to 50 MB for videos/files. The pipeline automatically makes a smaller 720x1280 Telegram copy if a rendered Short exceeds about 48 MB.
