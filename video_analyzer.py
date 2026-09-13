#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field, ValidationError

from kie_client import chat, parse_json_text


class Candidate(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    hook: str
    topic: str
    reason: str
    viral_score: float = Field(ge=0, le=10)
    context_score: float = Field(ge=0, le=10)
    emotion_score: float = Field(ge=0, le=10)
    curiosity_score: float = Field(ge=0, le=10)
    visual_score: float = Field(ge=0, le=10)


@dataclass
class Segment:
    start: float
    end: float
    text: str


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    if overlap <= 0:
        return 0.0
    shorter = max(0.001, min(a.end - a.start, b.end - b.start))
    return overlap / shorter


def weighted_score(c: Candidate) -> float:
    return (
        c.viral_score * 0.30
        + c.emotion_score * 0.20
        + c.curiosity_score * 0.20
        + c.visual_score * 0.15
        + c.context_score * 0.15
    )


def _segments_to_text(segments: Sequence[Segment]) -> str:
    lines: List[str] = []
    for s in segments:
        txt = " ".join(s.text.split())
        if txt:
            lines.append(f"[{s.start:.2f} - {s.end:.2f}] {txt}")
    return "\n".join(lines)


def _chunk_segments(segments: Sequence[Segment], max_chars: int = 35000) -> List[List[Segment]]:
    chunks: List[List[Segment]] = []
    current: List[Segment] = []
    size = 0
    for s in segments:
        line_len = len(s.text) + 40
        if current and size + line_len > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(s)
        size += line_len
    if current:
        chunks.append(current)
    return chunks


def _extract_candidates(raw: Any) -> List[Candidate]:
    if isinstance(raw, dict):
        rows = raw.get("candidates", raw.get("selected", []))
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []
    out: List[Candidate] = []
    for row in rows or []:
        try:
            out.append(Candidate.model_validate(row))
        except (ValidationError, TypeError, ValueError):
            continue
    return out


def analyze_transcript(
    segments: Sequence[Segment],
    shorts_count: int,
    min_duration: float = 25.0,
    max_duration: float = 58.0,
    max_overlap: float = 0.15,
) -> List[Candidate]:
    if not segments:
        raise ValueError("Transcript is empty")
    if shorts_count < 1:
        raise ValueError("shorts_count must be >= 1")

    chunks = _chunk_segments(segments)
    per_chunk = max(5, min(12, shorts_count * 3))
    candidates: List[Candidate] = []

    system_prompt = (
        "You are a senior short-form video editor. Identify the strongest moments for vertical shorts. "
        "Use ONLY facts/dialogue contained in the timestamped transcript. Never invent events, topics, "
        "quotes, or claims. Prefer clips with an immediate hook, tension, surprise, emotion, useful insight, "
        "humor, controversy, or a satisfying payoff. Avoid intros, sponsors, housekeeping and repetition. "
        "Timestamps must come from the transcript. Return valid JSON only."
    )

    for idx, chunk in enumerate(chunks, start=1):
        text = _segments_to_text(chunk)
        prompt = f"""
Select up to {per_chunk} candidate clips from transcript chunk {idx}/{len(chunks)}.

Rules:
- Target duration: {min_duration:.0f}-{max_duration:.0f} seconds when possible.
- Start on a thought boundary and end after the payoff.
- hook must be truthful and derived only from the clip.
- score every category from 0 to 10; do not give everything 9 or 10.
- choose distinct moments, not repeated versions of the same idea.
- visual_score reflects how likely the spoken moment also has useful visuals/reactions/actions.

Return exactly this JSON structure and nothing else:
{{
  "candidates": [
    {{
      "start": 12.3,
      "end": 52.1,
      "hook": "short truthful hook",
      "topic": "short topic",
      "reason": "why this is compelling",
      "viral_score": 8.2,
      "context_score": 8.0,
      "emotion_score": 7.5,
      "curiosity_score": 9.0,
      "visual_score": 7.0
    }}
  ]
}}

Transcript:
{text}
"""
        response_text = chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="low",
        )
        raw = parse_json_text(response_text)
        for c in _extract_candidates(raw):
            duration = c.end - c.start
            if c.end > c.start and 8 <= duration <= max_duration + 10:
                candidates.append(c)

    if not candidates:
        raise RuntimeError("Kie analyzer did not return usable candidates")

    candidates.sort(key=weighted_score, reverse=True)
    selected: List[Candidate] = []
    for c in candidates:
        if all(overlap_ratio(c, s) <= max_overlap for s in selected):
            # Prefer diversity when score is close by softly suppressing same-topic repeats.
            same_topic = any(c.topic.strip().lower() == s.topic.strip().lower() for s in selected)
            if same_topic and len(candidates) > shorts_count:
                alternatives = [
                    x for x in candidates
                    if x.topic.strip().lower() != c.topic.strip().lower()
                    and all(overlap_ratio(x, s) <= max_overlap for s in selected)
                ]
                if alternatives and weighted_score(alternatives[0]) >= weighted_score(c) - 0.7:
                    continue
            selected.append(c)
        if len(selected) >= shorts_count:
            break

    if len(selected) < shorts_count:
        for c in candidates:
            if c in selected:
                continue
            if all(overlap_ratio(c, s) <= max_overlap for s in selected):
                selected.append(c)
            if len(selected) >= shorts_count:
                break

    return selected[:shorts_count]


def save_analysis(path: str, selected: Sequence[Candidate]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "provider": "kie.ai",
                "selected": [
                    {**c.model_dump(), "weighted_score": round(weighted_score(c), 3)}
                    for c in selected
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
