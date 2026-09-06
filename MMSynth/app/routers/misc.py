"""Health, capabilities, jobs, and serving the audio back."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .. import bootstrap, engines, jobs, presets, storage, tools
from ..config import settings
from ..errors import NotReady

router = APIRouter()


def _gpu() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False, "note": "torch is not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "note": "no CUDA device"}
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return {"available": True, "name": name, "total_gb": round(total, 1)}


@router.get("/health")
def health():
    """Boot state, and what this pod is.

    Open without a token: the pod's own healthcheck needs it, and so does the
    boot screen, which has to be able to say "downloading" before anyone has
    typed a token.
    """
    boot = bootstrap.state.public()
    return {
        "status": boot["status"],
        "message": boot["message"],
        "models": boot["models"],
        "engines": engines.active(),
        "gpu": _gpu(),
        "auth": bool(settings.api_token),
        "data_dir": str(settings.data_dir),
    }


@router.get("/tools")
def list_tools():
    """What this pod can do, and how sure we are about each."""
    return tools.public()


@router.get("/instruments")
def instruments():
    try:
        return {"instruments": [i.public() for i in presets.library().values()]}
    except NotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/jobs")
def all_jobs():
    return {"jobs": [j.public() for j in jobs.queue.all()[:50]]}


@router.get("/jobs/{job_id}")
def one_job(job_id: str):
    job = jobs.queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job.public()


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    if not jobs.queue.cancel(job_id):
        raise HTTPException(status_code=404, detail="no such job")
    return jobs.queue.get(job_id).public()


@router.get("/jobs/{job_id}/events")
async def events(job_id: str):
    """Server-sent progress.

    A heartbeat every 15 seconds, because a proxy that sees nothing on a stream
    will eventually close it, and a render that outlives its own progress bar
    looks exactly like a render that died.
    """
    job = jobs.queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")

    async def stream():
        queue = jobs.queue.subscribe(job)
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
                if payload["status"] in ("done", "error", "cancelled"):
                    return
        finally:
            jobs.queue.unsubscribe(job, queue)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.get("/audio/{name:path}")
def audio_file(name: str):
    try:
        path = storage.resolve(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="no such audio")
    return FileResponse(path, media_type="audio/wav", filename=path.name)
