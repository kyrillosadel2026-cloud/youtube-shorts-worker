#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import subprocess
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests
import yt_dlp

from kie_client import chat, parse_json_text, upload_file
from video_analyzer import Segment, analyze_transcript, save_analysis, weighted_score
from short_editor import render_short


ROOT = Path(os.environ.get("YT_SHORTS_ROOT", "/tmp/yt_shorts/jobs"))
MAX_SHORTS = int(os.environ.get("MAX_SHORTS_PER_JOB", "8"))
MIN_DURATION = float(os.environ.get("SHORT_MIN_DURATION", "25"))
MAX_DURATION = float(os.environ.get("SHORT_MAX_DURATION", "58"))
AUDIO_CHUNK_SECONDS = int(os.environ.get("AUDIO_CHUNK_SECONDS", "300"))


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


def _preferred_subtitle_lang(info: Dict[str, Any]) -> Tuple[str | None, bool]:
    manual = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}
    preferred = ["ar", "en", "en-US", "en-GB"]
    for lang in preferred:
        if lang in manual:
            return lang, False
    if manual:
        return next(iter(manual)), False
    for lang in preferred:
        if lang in automatic:
            return lang, True
    # Avoid special live-chat/storyboard tracks if possible.
    for lang in automatic:
        if lang not in {"live_chat"}:
            return lang, True
    return None, False


def download_video(url: str, job_dir: Path) -> Tuple[Path, str, Path | None]:
    probe_opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info_probe = ydl.extract_info(url, download=False)
    lang, is_auto = _preferred_subtitle_lang(info_probe)

    outtmpl = str(job_dir / "source.%(ext)s")
    opts: Dict[str, Any] = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if lang:
        opts.update({
            "writesubtitles": not is_auto,
            "writeautomaticsub": is_auto,
            "subtitleslangs": [lang],
            "subtitlesformat": "json3/vtt/best",
        })

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
        files = [
            f for f in files
            if f.suffix.lower() not in {".part", ".ytdl", ".json3", ".vtt", ".srt"}
        ]
        if not files:
            raise RuntimeError("Downloaded video file was not found")
        source = files[0]

    subtitle = None
    subtitle_files = sorted(list(job_dir.glob("source.*.json3")) + list(job_dir.glob("source.*.vtt")))
    if subtitle_files:
        subtitle = subtitle_files[0]
    return source, title, subtitle


def _clean_caption_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_json3(path: Path) -> List[Segment]:
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    out: List[Segment] = []
    prev = ""
    for ev in data.get("events", []):
        segs = ev.get("segs") or []
        text = _clean_caption_text("".join(str(s.get("utf8", "")) for s in segs))
        if not text or text == prev:
            continue
        start = float(ev.get("tStartMs", 0)) / 1000.0
        duration = float(ev.get("dDurationMs", 0)) / 1000.0
        end = start + max(duration, 0.1)
        out.append(Segment(start, end, text))
        prev = text
    return out


def _vtt_time(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = "0", parts[0], parts[1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def _parse_vtt(path: Path) -> List[Segment]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[Segment] = []
    i = 0
    prev = ""
    ts_re = re.compile(r"(?P<s>\d{1,2}:\d{2}(?::\d{2})?[\.,]\d+)\s+-->\s+(?P<e>\d{1,2}:\d{2}(?::\d{2})?[\.,]\d+)")
    while i < len(lines):
        match = ts_re.search(lines[i])
        if not match:
            i += 1
            continue
        start = _vtt_time(match.group("s"))
        end = _vtt_time(match.group("e"))
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i])
            i += 1
        text = _clean_caption_text(" ".join(text_lines))
        if text and text != prev and end > start:
            out.append(Segment(start, end, text))
            prev = text
        i += 1
    return out


def segments_from_subtitle(path: Path) -> List[Segment]:
    if path.suffix.lower() == ".json3":
        return _parse_json3(path)
    return _parse_vtt(path)


def extract_audio_chunks(source: Path, chunks_dir: Path) -> List[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunks_dir / "audio_%04d.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "48k",
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


def transcribe_with_kie(chunks: Sequence[Path], transcript_path: Path) -> List[Segment]:
    segments: List[Segment] = []
    raw: List[Dict[str, Any]] = []
    language = None

    for idx, chunk in enumerate(chunks):
        offset = idx * AUDIO_CHUNK_SECONDS
        media_url = upload_file(chunk, upload_path="youtube-shorts/transcription")
        prompt = (
            "Transcribe this audio accurately in its original language. Return JSON only, no markdown. "
            "Use relative timestamps from the beginning of THIS audio file. Make segments roughly 3-12 seconds, "
            "aligned to speech/thought boundaries. Do not translate and do not summarize. Schema: "
            '{"language":"detected-language","segments":[{"start":0.0,"end":4.2,"text":"exact speech"}]}'
        )
        response_text = chat(
            [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": media_url}},
                ],
            }],
            reasoning_effort="low",
        )
        payload = parse_json_text(response_text)
        if isinstance(payload, dict) and language is None:
            language = payload.get("language")
        rows = payload.get("segments", []) if isinstance(payload, dict) else []
        for item in rows:
            try:
                text = str(item.get("text", "")).strip()
                start = offset + float(item.get("start", 0))
                end = offset + float(item.get("end", 0))
            except (TypeError, ValueError, AttributeError):
                continue
            if text and end > start:
                seg = Segment(start, end, text)
                segments.append(seg)
                raw.append({"start": seg.start, "end": seg.end, "text": seg.text})

    transcript_path.write_text(
        json.dumps(
            {"provider": "kie.ai", "language": language, "segments": raw},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return segments


def write_transcript(segments: Sequence[Segment], transcript_path: Path, provider: str) -> None:
    transcript_path.write_text(
        json.dumps(
            {
                "provider": provider,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in segments
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def process_job(chat_id: int, count: int, url: str) -> None:
    if not os.environ.get("KIE_API_KEY"):
        raise RuntimeError("KIE_API_KEY is not configured in Railway")

    job_id = uuid.uuid4().hex[:12]
    job_dir = ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    tg_message(chat_id, f"🎬 بدأت معالجة الفيديو\nالعدد المطلوب: {count}\nJob: {job_id}")
    log(f"Job {job_id}: download")
    source, title, subtitle = download_video(url, job_dir)

    transcript_path = job_dir / "transcript.json"
    segments: List[Segment] = []
    if subtitle:
        try:
            segments = segments_from_subtitle(subtitle)
            if segments:
                write_transcript(segments, transcript_path, provider="youtube-subtitles")
                tg_message(chat_id, "✅ تم تنزيل الفيديو وقراءة الترجمة. جاري تحليل أقوى اللحظات...")
        except Exception as e:
            log(f"Subtitle parse failed; falling back to Kie audio transcription: {e}")
            segments = []

    if not segments:
        tg_message(chat_id, "✅ تم تنزيل الفيديو. لا توجد ترجمة مناسبة؛ جاري استخراج الكلام عبر Kie AI...")
        chunks_dir = job_dir / "audio_chunks"
        audio_chunks = extract_audio_chunks(source, chunks_dir)
        segments = transcribe_with_kie(audio_chunks, transcript_path)

    if not segments:
        raise RuntimeError("لم أستطع استخراج Transcript صالح من الفيديو")

    log(f"Job {job_id}: analyze {len(segments)} segments using Kie")
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

    tg_message(chat_id, f"🧠 التحليل انتهى عبر Kie. اخترت {len(selected)} مقطع. جاري المونتاج...")

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

    keep_jobs = os.environ.get("KEEP_JOB_FILES", "0").lower() not in {"0", "false", "no"}
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
