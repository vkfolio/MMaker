"""Jobs (SSE progress) and the voice library."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .. import jobs, voices

jobs_router = APIRouter(prefix="/api/jobs", tags=["jobs"])
voices_router = APIRouter(prefix="/api/voices", tags=["voices"])


@jobs_router.get("")
def list_jobs(project_id: str | None = None):
    return {"jobs": jobs.queue.list(project_id)}


@jobs_router.get("/{job_id}")
def get_job(job_id: str):
    job = jobs.queue.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return job.public()


@jobs_router.get("/{job_id}/events")
async def job_events(job_id: str):
    """Server-sent progress. The UI never blocks on a render."""
    job = jobs.queue.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")

    async def stream():
        try:
            async for payload in jobs.queue.events(job):
                yield f"data: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:      # client navigated away
            return

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@voices_router.get("")
def list_voices():
    return {"voices": voices.list_voices()}


@voices_router.post("")
async def add_voice(
    file: UploadFile = File(...),
    label: str = Form(...),
    description: str = Form(""),
    consent: str = Form(""),
):
    """Clone a voice from a short reference clip (~10s of singing is enough).

    `consent` records who authorised the use of this voice. It is not enforced,
    but leaving it blank is a visible reminder that the question was not answered.
    """
    suffix = Path(file.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        meta = voices.add(label, tmp_path, description=description, consent=consent)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"voice": meta}


@voices_router.get("/{voice_id}/reference")
def voice_reference(voice_id: str):
    path = voices.reference_path(voice_id)
    if not path:
        raise HTTPException(404, "no such voice")
    return FileResponse(str(path), media_type="audio/wav")


@voices_router.delete("/{voice_id}")
def delete_voice(voice_id: str):
    if not voices.delete(voice_id):
        raise HTTPException(404, "no such voice")
    return {"deleted": voice_id}
