from __future__ import annotations

import base64
import os
import secrets
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("WORKER_LOG_DIR", "/tmp/yt_shorts_logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YouTube Shorts Worker", version="1.0.0")
_jobs: Dict[str, subprocess.Popen] = {}


class ProcessRequest(BaseModel):
    chat_id: int
    message: str = Field(min_length=1, max_length=4096)


def _check_token(x_worker_token: str | None) -> None:
    expected = os.environ.get("WORKER_API_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="WORKER_API_TOKEN is not configured")
    if not x_worker_token or not secrets.compare_digest(x_worker_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict:
    # Clean completed process handles from memory.
    for job_id, proc in list(_jobs.items()):
        if proc.poll() is not None:
            _jobs.pop(job_id, None)
    return {"ok": True, "active_jobs": len(_jobs)}


@app.post("/process", status_code=202)
def process_video(payload: ProcessRequest, x_worker_token: str | None = Header(default=None)) -> dict:
    _check_token(x_worker_token)

    max_parallel = int(os.environ.get("MAX_PARALLEL_JOBS", "1"))
    for job_id, proc in list(_jobs.items()):
        if proc.poll() is not None:
            _jobs.pop(job_id, None)
    if len(_jobs) >= max_parallel:
        raise HTTPException(status_code=429, detail="Worker is busy. Try again shortly.")

    job_id = uuid.uuid4().hex[:12]
    message_b64 = base64.b64encode(payload.message.encode("utf-8")).decode("ascii")
    log_path = LOG_DIR / f"{job_id}.log"
    log_file = log_path.open("ab", buffering=0)

    cmd = [
        sys.executable,
        str(APP_DIR / "run_pipeline.py"),
        "--telegram-message-b64",
        message_b64,
        "--chat-id",
        str(payload.chat_id),
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(APP_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True,
    )
    _jobs[job_id] = proc

    return {
        "accepted": True,
        "job_id": job_id,
        "message": "Processing started. Progress and finished Shorts will be sent to Telegram.",
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str, x_worker_token: str | None = Header(default=None)) -> dict:
    _check_token(x_worker_token)
    proc = _jobs.get(job_id)
    log_path = LOG_DIR / f"{job_id}.log"

    if proc is None:
        if not log_path.exists():
            raise HTTPException(status_code=404, detail="Unknown job")
        status = "finished"
        return_code = None
    else:
        return_code = proc.poll()
        status = "running" if return_code is None else ("finished" if return_code == 0 else "failed")

    tail = ""
    if log_path.exists():
        try:
            data = log_path.read_bytes()
            tail = data[-6000:].decode("utf-8", errors="replace")
        except OSError:
            pass

    return {"job_id": job_id, "status": status, "return_code": return_code, "log_tail": tail}
