# YouTube Shorts Worker — Kie.ai Edition

This edition uses your **Kie.ai API key** instead of an OpenAI API key.

## Railway Variables

Required:

- `KIE_API_KEY` — your Kie.ai API key
- `TELEGRAM_BOT_TOKEN` — your Telegram bot token

Recommended:

- `MAX_PARALLEL_JOBS=1`
- `KEEP_JOB_FILES=0`

Optional:

- `KIE_API_BASE=https://api.kie.ai`
- `KIE_CHAT_PATH=/gemini-3-7-flash-openai/v1/chat/completions`
- `AUDIO_CHUNK_SECONDS=300`
- `SHORT_MIN_DURATION=25`
- `SHORT_MAX_DURATION=58`
- `MAX_SHORTS_PER_JOB=8`

`OPENAI_API_KEY` is not used by this edition.

## How transcription works

1. It first tries to use manual or automatic YouTube subtitles, preserving their timestamps.
2. If suitable subtitles are unavailable, it extracts audio chunks, uploads them temporarily to Kie, and asks Gemini 3.7 Flash through Kie to produce a timestamped transcript.
3. The transcript is then analyzed through Kie to choose strong, non-overlapping short-form moments.
4. FFmpeg renders 9:16 Shorts with captions and sends them to Telegram.

## Telegram command

`/shorts 3 https://youtu.be/VIDEO_ID`

## Deploy

Replace the matching files in the root of your GitHub Railway repository with this bundle, commit, then let Railway redeploy.
