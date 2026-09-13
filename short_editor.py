#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence


def run(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr[-5000:]}")


def ffprobe_size(path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path)
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def ass_escape(text: str) -> str:
    return (
        text.replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def wrap_caption(text: str, max_chars: int = 28) -> str:
    words = text.split()
    if not words:
        return ""
    lines: List[str] = []
    line: List[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if line else 0)
        if line and length + extra > max_chars:
            lines.append(" ".join(line))
            line = [word]
            length = len(word)
        else:
            line.append(word)
            length += extra
    if line:
        lines.append(" ".join(line))
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return r"\N".join(lines)


def write_ass(
    output_path: Path,
    hook: str,
    clip_start: float,
    clip_end: float,
    transcript_segments: Sequence[dict],
) -> None:
    header = r"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,64,&H00FFFFFF,&H000000FF,&H00000000,&H78000000,-1,0,0,0,100,100,0,0,1,5,1,2,70,70,250,1
Style: Hook,DejaVu Sans,72,&H0000FFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,5,1,8,70,70,160,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    clip_duration = clip_end - clip_start
    hook_end = min(3.2, max(1.8, clip_duration * 0.12))
    if hook.strip():
        lines.append(
            f"Dialogue: 1,{ass_time(0)},{ass_time(hook_end)},Hook,,0,0,0,,{ass_escape(wrap_caption(hook, 24))}\n"
        )

    for seg in transcript_segments:
        s = float(seg["start"])
        e = float(seg["end"])
        if e <= clip_start or s >= clip_end:
            continue
        local_s = max(0.0, s - clip_start)
        local_e = min(clip_duration, e - clip_start)
        text = wrap_caption(str(seg.get("text", "")).strip(), 30)
        if text and local_e > local_s:
            lines.append(
                f"Dialogue: 0,{ass_time(local_s)},{ass_time(local_e)},Caption,,0,0,0,,{ass_escape(text)}\n"
            )

    output_path.write_text("".join(lines), encoding="utf-8")


def render_short(
    source: Path,
    output: Path,
    start: float,
    end: float,
    hook: str,
    transcript_segments: Sequence[dict],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    ass_path = output.with_suffix(".ass")
    write_ass(ass_path, hook, start, end, transcript_segments)

    width, height = ffprobe_size(source)
    # Fill a 9:16 frame. Landscape sources are height-scaled then center-cropped;
    # portrait sources are width-scaled then vertically center-cropped.
    if width / height >= 9 / 16:
        vf_base = "scale=-2:1920,crop=1080:1920"
    else:
        vf_base = "scale=1080:-2,crop=1080:1920"

    ass_filter_path = str(ass_path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    vf = (
        f"{vf_base},"
        "eq=contrast=1.04:saturation=1.06,"
        f"subtitles='{ass_filter_path}'"
    )

    run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(source),
        "-t", f"{duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", os.environ.get("FFMPEG_PRESET", "veryfast"),
        "-crf", os.environ.get("FFMPEG_CRF", "21"),
        "-c:a", "aac", "-b:a", "160k",
        "-af", "loudnorm=I=-14:LRA=11:TP=-1.5",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        str(output),
    ])

    try:
        ass_path.unlink()
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--hook", default="")
    parser.add_argument("--transcript", required=True)
    args = parser.parse_args()

    transcript = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    segments = transcript.get("segments", transcript)
    render_short(
        Path(args.source), Path(args.output), args.start, args.end,
        args.hook, segments,
    )


if __name__ == "__main__":
    main()
