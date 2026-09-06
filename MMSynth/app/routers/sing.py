"""Singing, and the voices to sing with."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from .. import jobs, service, voices
from ..errors import NotReady
from ..schemas import LyricsRequest, PhonemesRequest, SingRequest

router = APIRouter()


@router.post("/sing")
def sing(body: SingRequest,
         idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    job = jobs.queue.submit("sing", lambda report: service.sing(body, report),
                            idempotency_key=idempotency_key)
    return {"job": job.public()}


@router.post("/lyrics")
def lyrics(body: LyricsRequest):
    """Spread a typed line across the notes. Immediate: no GPU, no job.

    Changes nothing. It answers "what would these notes become", including how
    many hand-set ones would be replaced, so the client can warn before it
    commits. Separating this from /sing is what lets the alignment be looked at
    and corrected rather than discovered in the audio.
    """
    return service.distribute_lyrics(body)


@router.post("/phonemes")
def phonemes(body: PhonemesRequest):
    """ARPAbet per syllable, for the row above the notes."""
    return service.phonemes_for_words(body)


@router.get("/voices")
def list_voices():
    return {"voices": voices.list_voices()}


@router.post("/voices")
def add_voice(clip: UploadFile = File(...),
              label: str = Form(...),
              description: str = Form(""),
              language: str = Form("en"),
              agreement: str = Form(""),
              recorded: bool = Form(False)):
    """Add a voice from an uploaded clip.

    `agreement` names the signed document covering AI singing-voice use and
    `recorded` says whether verbal consent was captured. Both default to absent,
    and a voice missing either is stored and listed but refused at render time --
    the check lives in voices.reference(), so there is one place to argue with.
    """
    suffix = Path(clip.filename or "reference.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        shutil.copyfileobj(clip.file, handle)
        staged = handle.name
    try:
        return voices.add(staged, label=label, description=description,
                          language=language, agreement=agreement,
                          recorded=recorded)
    except NotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    finally:
        Path(staged).unlink(missing_ok=True)
