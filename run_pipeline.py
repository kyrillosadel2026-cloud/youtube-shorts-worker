#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests
import yt_dlp
from openai import OpenAI

from video_analyzer import Segment, analyze_transcript, save_analysis, weighted_score
from short_editor import render_short


ROOT = Path(os.environ.get("YT_SHORTS_ROOT", "/data/yt_shorts"))
MAX_SHORTS = int(os.environ.get("MAX_SHORTS_PER_JOB", "8"))
MIN_DURATION = float(os.environ.get("SHORT_MIN_DURATION", "25"))
MAX_DURATION = float(os.environ.get("SHORT_MAX_DURATION", "58"))
TRANSCRIBE_MODEL = os.environ.get("TRANSCRIBE_MODEL", "whisper-1")
AUDIO_CHUNK_SECONDS = int(os.environ.get("AUDIO_CHUNK_SECONDS", "720"))


def log(msg: str) -> None:
    print(msg, flush=True)


def tg_api(method: str) -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    return f"https://api.telegram.org/bot{token}/{method}"


def tg_message(chat_id: int, text: str) -> None:
    try:
        requests.post(tg_api("sendMessage"), data={"chat_id": chat_id, "text": text}, timeout=45).raise_for_status()
    except Exception as e:
        log(f"Telegram sendMessage failed: {e}")


def compress_for_telegram(path: Path) -> Path:
    if path.stat().st_size <= 48 * 1024 * 1024:
        return path
    compressed = path.with_name(path.stem + "_telegram.mp4")
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(path),
            "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "veryfast", "-b:v", "3800k", "-maxrate", "4200k", "-bufsize", "7600k",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(compressed),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Telegram compression failed: {proc.stderr[-2500:]}")
    return compressed


def tg_video(chat_id: int, path: Path, caption: str) -> None:
    upload_path = compress_for_telegram(path)
    with upload_path.open("rb") as f:
        response = requests.post(
            tg_api("sendVideo"),
            data={"chat_id": chat_id, "caption": caption[:1024], "supports_streaming": "true"},
            files={"video": (upload_path.name, f, "video/mp4")},
            timeout=300,
        )
        response.raise_for_status()


def parse_command(message: str) -> Tuple[int, str]:
    message = message.strip()
    pattern = re.compile(
        r"^(?:/shorts(?:@\w+)?\s+)?(?P<count>\d+)\s+(?P<url>https?://\S+)$",
        re.IGNORECASE,
    )
    match = pattern.match(message)
    if not match:
        # Also accept URL first: /shorts URL 3
        pattern2 = re.compile(
            r"^(?:/shorts(?:@\w+)?\s+)?(?P<url>https?://\S+)\s+(?P<count>\d+)$",
            re.IGNORECASE,
        )
        match = pattern2.match(message)
    if not match:
        raise ValueError("صيغة الأمر غير صحيحة")

    count = int(match.group("count"))
    url = match.group("url").strip()
    if count < 1 or count > MAX_SHORTS:
        raise ValueError(f"عدد الشورتس لازم يكون من 1 إلى {MAX_SHORTS}")
    if not re.search(r"(?:youtube\.com|youtu\.be)", url, re.IGNORECASE):
        raise ValueError("حالياً المصدر المدعوم هو YouTube فقط")
    return count, url


def download_video(url: str, job_dir: Path) -> Tuple[Path, str]:
    outtmpl = str(job_dir / "source.%(ext)s")
    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") or "YouTube video"
        prepared = Path(ydl.prepare_filename(info))

    mp4 = job_dir / "source.mp4"
    if mp4.exists():
        source = mp4
    elif prepared.exists():
        source = prepared
    else:
        files = sorted(job_dir.glob("source.*"))
        files = [f for f in files if f.suffix.lower() not in {".part", ".ytdl"}]
        if not files:
            raise RuntimeError("Downloaded video file was not found")
        source = files[0]
    return source, title


def extract_audio_chunks(source: Path, chunks_dir: Path) -> List[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunks_dir / "audio_%04d.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "64k",
        "-f", "segment", "-segment_time", str(AUDIO_CHUNK_SECONDS),
        "-reset_timestamps", "1", str(pattern),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr[-3000:]}")
    chunks = sorted(chunks_dir.glob("audio_*.mp3"))
    if not chunks:
        raise RuntimeError("No audio chunks were created")
    return chunks


def transcribe(chunks: Sequence[Path], transcript_path: Path) -> List[Segment]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    segments: List[Segment] = []
    raw: List[Dict[str, Any]] = []
    language = None

    previous_tail = ""
    for idx, chunk in enumerate(chunks):
        offset = idx * AUDIO_CHUNK_SECONDS
        kwargs: Dict[str, Any] = {
            "file": chunk.open("rb"),
            "model": TRANSCRIBE_MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if previous_tail:
            kwargs["prompt"] = previous_tail[-500:]
        try:
            result = client.audio.transcriptions.create(**kwargs)
        finally:
            kwargs["file"].close()

        if language is None:
            language = getattr(result, "language", None)

        result_segments = getattr(result, "segments", None)
        if result_segments is None and hasattr(result, "model_dump"):
            result_segments = result.model_dump().get("segments", [])
        result_segments = result_segments or []

        chunk_text_parts: List[str] = []
        for item in result_segments:
            if hasattr(item, "model_dump"):
                item = item.model_dump()
            elif not isinstance(item, dict):
                item = {
                    "start": getattr(item, "start", 0),
                    "end": getattr(item, "end", 0),
                    "text": getattr(item, "text", ""),
                }
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = offset + float(item.get("start", 0.0))
            end = offset + float(item.get("end", 0.0))
            if end <= start:
                continue
            seg = Segment(start, end, text)
            segments.append(seg)
            raw.append({"start": seg.start, "end": seg.end, "text": seg.text})
            chunk_text_parts.append(text)
        previous_tail = " ".join(chunk_text_parts)[-800:]

    transcript_path.write_text(
        json.dumps(
            {
                "language": language,
                "transcription_model": TRANSCRIBE_MODEL,
                "segments": raw,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return segments


def process_job(chat_id: int, count: int, url: str) -> None:
    job_id = uuid.uuid4().hex[:12]
    job_dir = ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    tg_message(chat_id, f"🎬 بدأت معالجة الفيديو\nالعدد المطلوب: {count}\nJob: {job_id}")
    log(f"Job {job_id}: download")
    source, title = download_video(url, job_dir)

    tg_message(chat_id, "✅ تم تنزيل الفيديو. جاري استخراج الكلام وتحليله...")
    chunks_dir = job_dir / "audio_chunks"
    transcript_path = job_dir / "transcript.json"
    audio_chunks = extract_audio_chunks(source, chunks_dir)
    segments = transcribe(audio_chunks, transcript_path)
    if not segments:
        raise RuntimeError("لم أستطع استخراج Transcript صالح من الفيديو")

    log(f"Job {job_id}: analyze {len(segments)} segments")
    selected = analyze_transcript(
        segments,
        shorts_count=count,
        min_duration=MIN_DURATION,
        max_duration=MAX_DURATION,
        max_overlap=0.15,
    )
    if not selected:
        raise RuntimeError("لم يتم العثور على مقاطع مناسبة")
    save_analysis(str(job_dir / "analysis.json"), selected)

    tg_message(chat_id, f"🧠 التحليل انتهى. اخترت {len(selected)} مقطع. جاري المونتاج...")

    transcript_json = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript_segments = transcript_json["segments"]
    outputs: List[Path] = []
    failures: List[str] = []

    for i, clip in enumerate(selected, start=1):
        out = job_dir / f"short_{i:02d}.mp4"
        try:
            render_short(
                source=source,
                output=out,
                start=clip.start,
                end=clip.end,
                hook=clip.hook,
                transcript_segments=transcript_segments,
            )
            outputs.append(out)
            caption = (
                f"Short {i}/{len(selected)}\n"
                f"{clip.topic}\n"
                f"⏱ {clip.end-clip.start:.0f}s | Score {weighted_score(clip):.1f}/10\n"
                f"Hook: {clip.hook}"
            )
            tg_video(chat_id, out, caption)
        except Exception as e:
            failures.append(f"Short {i}: {e}")
            log(traceback.format_exc())

    if outputs:
        summary = f"✅ انتهى الفيديو: {title}\nتم إرسال {len(outputs)} Short"
        if failures:
            summary += f"\n⚠️ فشل {len(failures)} مقطع أثناء الرندر/الإرسال."
        tg_message(chat_id, summary)
    else:
        raise RuntimeError("فشل رندر كل الشورتس")

    keep_jobs = os.environ.get("KEEP_JOB_FILES", "1").lower() not in {"0", "false", "no"}
    if not keep_jobs:
        shutil.rmtree(job_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-message-b64", required=True)
    parser.add_argument("--chat-id", required=True, type=int)
    args = parser.parse_args()

    allowed_raw = os.environ.get("ALLOWED_TELEGRAM_CHAT_IDS", "").strip()
    if allowed_raw:
        allowed = {int(x.strip()) for x in allowed_raw.split(",") if x.strip()}
        if args.chat_id not in allowed:
            log(f"Rejected unauthorized Telegram chat: {args.chat_id}")
            return

    try:
        message = base64.b64decode(args.telegram_message_b64).decode("utf-8")
    except Exception:
        tg_message(args.chat_id, "❌ لم أستطع قراءة الرسالة.")
        raise

    try:
        count, url = parse_command(message)
    except ValueError as e:
        tg_message(
            args.chat_id,
            f"❌ {e}\n\nاستخدم الصيغة:\n/shorts 3 https://youtu.be/VIDEO_ID\n\nأو:\n/shorts https://youtu.be/VIDEO_ID 3",
        )
        return

    try:
        process_job(args.chat_id, count, url)
    except Exception as e:
        log(traceback.format_exc())
        tg_message(args.chat_id, f"❌ حصل خطأ أثناء المعالجة:\n{str(e)[:1200]}")
        raise


if __name__ == "__main__":
    main()
