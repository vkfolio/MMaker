"""Playing a melody on an instrument."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from .. import jobs, lyrics, service, uploads
from ..errors import MMSynthError
from ..schemas import EnhanceRequest, RenderRequest, TranscribeRequest
from ..score import read_melody

router = APIRouter()


@router.post("/render")
def render(body: RenderRequest,
           idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    """Notes in, audio out.

    Returns a job. Even the dry sampler path is queued rather than answered
    inline: it shares a queue with the GPU, and a route whose latency depends on
    what else is running is a route the client has to treat as slow anyway.
    """
    job = jobs.queue.submit("render", lambda report: service.render(body, report),
                            idempotency_key=idempotency_key)
    return {"job": job.public()}


@router.post("/midi")
async def read_midi(file: UploadFile = File(...)):
    """Turn a dropped .mid into notes.

    Immediate, no GPU, no job. Returns what was done to the file as a sentence
    as well as the notes -- v1 renders a single line, so a multi-track file
    *will* lose material, and the only acceptable version of that is one the
    user is told about.

    Every note comes back carrying `da`, so a melody dropped without a thought
    about words is already singable.
    """
    suffix = Path(file.filename or "in.mid").suffix or ".mid"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        shutil.copyfileobj(file.file, handle)
        staged = handle.name
    try:
        notes, bpm, note = read_melody(staged)
    except MMSynthError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        Path(staged).unlink(missing_ok=True)

    lyrics.ensure_defaults(notes)
    return {
        "notes": [{"start": round(n.start, 4), "length": round(n.length, 4),
                   "pitch": n.pitch, "velocity": n.velocity,
                   "lyric": n.lyric, "phonemes": n.phonemes} for n in notes],
        "bpm": round(bpm, 2),
        "note": note,
    }


@router.post("/uploads")
async def upload_audio(file: UploadFile = File(...)):
    """Take a recording. Cheap, immediate, and separate from rendering.

    Uploading and rendering are split so the app can show what it found before
    anyone spends GPU on it, and so one upload can feed several attempts.
    """
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        shutil.copyfileobj(file.file, handle)
        staged = handle.name
    try:
        return uploads.save(Path(staged), file.filename or "audio.wav")
    except MMSynthError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        Path(staged).unlink(missing_ok=True)


@router.post("/enhance")
def enhance(body: EnhanceRequest,
            idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    """Re-voice a recording, keeping its timing and pitch."""
    job = jobs.queue.submit("enhance", lambda report: service.enhance(body, report),
                            idempotency_key=idempotency_key)
    return {"job": job.public()}


@router.post("/transcribe")
def transcribe(body: TranscribeRequest):
    """Read a melody out of a recording. No GPU, no job."""
    try:
        return service.transcribe_audio(body)
    except MMSynthError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
