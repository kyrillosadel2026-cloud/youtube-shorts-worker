from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import requests

KIE_API_BASE = os.environ.get("KIE_API_BASE", "https://api.kie.ai").rstrip("/")
KIE_CHAT_PATH = os.environ.get(
    "KIE_CHAT_PATH", "/gemini-3-7-flash-openai/v1/chat/completions"
)
KIE_UPLOAD_URL = os.environ.get(
    "KIE_UPLOAD_URL", "https://kieai.redpandaai.co/api/file-stream-upload"
)


def _api_key() -> str:
    key = os.environ.get("KIE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("KIE_API_KEY is not configured")
    return key


def _headers(json_body: bool = True) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {_api_key()}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def upload_file(path: Path, upload_path: str = "youtube-shorts/audio") -> str:
    """Upload a local file to Kie temporary storage and return a public URL."""
    with path.open("rb") as f:
        response = requests.post(
            KIE_UPLOAD_URL,
            headers=_headers(json_body=False),
            data={"uploadPath": upload_path, "fileName": path.name},
            files={"file": (path.name, f, "application/octet-stream")},
            timeout=300,
        )
    if not response.ok:
        raise RuntimeError(
            f"Kie file upload failed ({response.status_code}): {response.text[:1500]}"
        )
    data = response.json()
    payload = data.get("data") or {}
    url = payload.get("fileUrl") or payload.get("downloadUrl")
    if not url:
        raise RuntimeError(f"Kie upload returned no file URL: {data}")
    return str(url)


def chat(messages: List[Dict[str, Any]], reasoning_effort: str = "low") -> str:
    url = f"{KIE_API_BASE}{KIE_CHAT_PATH}"
    payload: Dict[str, Any] = {
        "messages": messages,
        "stream": False,
        "reasoning_effort": reasoning_effort,
    }
    response = requests.post(
        url,
        headers=_headers(json_body=True),
        json=payload,
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(
            f"Kie chat failed ({response.status_code}): {response.text[:2000]}"
        )
    data = response.json()

    # OpenAI-style response.
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        content = ((choices[0] or {}).get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "\n".join(parts)

    # Gemini-style response sometimes returned by Kie.
    candidates = data.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = (candidates[0] or {}).get("content") or {}
        parts = content.get("parts") or []
        texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
        if texts:
            return "\n".join(str(x) for x in texts)

    # Generic fallbacks.
    for key in ("output_text", "text", "response"):
        if isinstance(data.get(key), str):
            return data[key]

    raise RuntimeError(f"Could not extract text from Kie response: {str(data)[:2000]}")


def parse_json_text(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Extract the largest likely JSON object/array from conversational wrappers.
        starts = [(text.find("{"), "{"), (text.find("["), "[")]
        starts = [(i, c) for i, c in starts if i >= 0]
        if not starts:
            raise
        start, opener = min(starts, key=lambda x: x[0])
        closer = "}" if opener == "{" else "]"
        end = text.rfind(closer)
        if end <= start:
            raise
        return json.loads(text[start : end + 1])
