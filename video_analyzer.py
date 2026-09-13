#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI
from pydantic import BaseModel, Field


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


class CandidateBatch(BaseModel):
    candidates: List[Candidate]


class FinalSelection(BaseModel):
    selected: List[Candidate]


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
    lines = []
    for s in segments:
        txt = " ".join(s.text.split())
        if txt:
            lines.append(f"[{s.start:.2f} - {s.end:.2f}] {txt}")
    return "\n".join(lines)


def _chunk_segments(segments: Sequence[Segment], max_chars: int = 45000) -> List[List[Segment]]:
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


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")


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

    client = _client()
    chunks = _chunk_segments(segments)
    per_chunk = max(4, min(10, shorts_count * 3))
    candidates: List[Candidate] = []

    system_prompt = (
        "You are a senior short-form video editor. Your task is to identify moments from a long-form "
        "video transcript that can become high-retention vertical shorts. You MUST only use facts and "
        "dialogue present in the provided timestamped transcript. Never invent a topic, quote, event, "
        "or claim. Prefer clips that stand on their own, start with a strong hook or immediate tension, "
        "contain an emotional, surprising, useful, funny, controversial, or high-curiosity payoff, and "
        "have enough context to make sense without the full video. Avoid intros, housekeeping, sponsor "
        "reads, repeated points, and clips that require unseen context. Timestamps must refer to the "
        "provided transcript."
    )

    for idx, chunk in enumerate(chunks, start=1):
        text = _segments_to_text(chunk)
        prompt = f"""
Select up to {per_chunk} strong candidate clips from this transcript chunk.

Rules:
- Clip duration must be between {min_duration:.0f} and {max_duration:.0f} seconds whenever the source allows it.
- Start as close as possible to a sentence or thought boundary.
- End after the payoff or complete thought, not mid-sentence.
- The hook must be a truthful short on-screen hook derived from this clip only.
- Scores are 0-10 and should be discriminative, not all 9s or 10s.
- Prefer visual/emotional moments when the transcript suggests them.
- Do not select nearly identical moments from this chunk.

Transcript chunk {idx}/{len(chunks)}:
{text}
"""
        response = client.responses.parse(
            model=_model(),
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            text_format=CandidateBatch,
        )
        parsed = response.output_parsed
        if parsed:
            for c in parsed.candidates:
                duration = c.end - c.start
                if c.end > c.start and 8 <= duration <= max_duration + 8:
                    candidates.append(c)

    if not candidates:
        raise RuntimeError("The AI analyzer did not return any usable candidates")

    # Deduplicate and remove high-overlap candidates before final AI pass.
    candidates.sort(key=weighted_score, reverse=True)
    deduped: List[Candidate] = []
    for c in candidates:
        if all(overlap_ratio(c, d) <= max_overlap for d in deduped):
            deduped.append(c)
        if len(deduped) >= max(shorts_count * 4, shorts_count + 4):
            break

    if len(deduped) <= shorts_count:
        return deduped[:shorts_count]

    candidate_text = "\n".join(
        f"#{i+1} {c.start:.2f}-{c.end:.2f} | topic={c.topic} | hook={c.hook} | "
        f"viral={c.viral_score:.1f}, context={c.context_score:.1f}, emotion={c.emotion_score:.1f}, "
        f"curiosity={c.curiosity_score:.1f}, visual={c.visual_score:.1f} | reason={c.reason}"
        for i, c in enumerate(deduped)
    )

    final_prompt = f"""
Choose the best {shorts_count} shorts from the candidates below.

Constraints:
- Return exactly {shorts_count} when enough candidates exist.
- Keep overlap under {max_overlap*100:.0f}% between selected clips.
- Prefer topic diversity when quality is close.
- Each selected clip must work as a standalone short.
- Do not alter timestamps beyond the listed candidates.
- Do not invent new candidates.

Candidates:
{candidate_text}
"""
    response = client.responses.parse(
        model=_model(),
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
        text_format=FinalSelection,
    )
    parsed = response.output_parsed
    selected = parsed.selected if parsed else []

    # Safety pass: keep only exact-ish known timestamps and non-overlapping clips.
    final: List[Candidate] = []
    for c in selected:
        match = min(
            deduped,
            key=lambda d: abs(d.start - c.start) + abs(d.end - c.end),
        )
        if abs(match.start - c.start) + abs(match.end - c.end) > 3.0:
            continue
        if all(overlap_ratio(match, f) <= max_overlap for f in final):
            final.append(match)
        if len(final) >= shorts_count:
            break

    if len(final) < shorts_count:
        for c in deduped:
            if c in final:
                continue
            if all(overlap_ratio(c, f) <= max_overlap for f in final):
                final.append(c)
            if len(final) >= shorts_count:
                break

    return final[:shorts_count]


def save_analysis(path: str, selected: Sequence[Candidate]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected": [
                    {**c.model_dump(), "weighted_score": round(weighted_score(c), 3)}
                    for c in selected
                ]
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
