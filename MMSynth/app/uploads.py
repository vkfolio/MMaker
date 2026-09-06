"""Audio the user brings.

Uploading and rendering are separate steps on purpose. An upload is cheap and a
render costs GPU, so the file lands first and the app can show what it found --
how long it is, whether it is a single line or a full mix -- before anyone
spends anything. It also means one upload can feed several attempts.

Everything is normalised to 48 kHz WAV on the way in. The engines disagree about
rates (ACE-Step 48 kHz, SoulX 24 kHz) and about formats, and reconciling that
per-call is how you end up with a resample bug nobody can find. One shape in,
converted once, at the edge.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path

from . import audio, storage
from .config import settings
from .errors import RenderError

SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".aif"}
MAX_SECONDS = 600.0


def upload_dir() -> Path:
    d = settings.data_dir / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_wav(src: Path, dest: Path) -> None:
    """Convert with ffmpeg, which is in the image because the engines need it."""
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "2", "-ar", str(settings.sample_rate),
         "-c:a", "pcm_s24le", str(dest)],
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not dest.exists():
        raise RenderError(
            "could not read that audio file: " + (proc.stderr or "")[-240:].strip())


def save(staged: Path, filename: str) -> dict:
    """Take a staged upload and make it something the engines can use."""
    staged = Path(staged)
    suffix = Path(filename or "audio.wav").suffix.lower()
    if suffix and suffix not in SUFFIXES:
        raise RenderError(
            f"{suffix} is not an audio format this reads. Try wav, mp3, flac or ogg.")

    upload_id = uuid.uuid4().hex[:12]
    dest = upload_dir() / f"{upload_id}.wav"
    _to_wav(staged, dest)

    data, sr = audio.read(dest)
    seconds = data.shape[0] / max(sr, 1)
    if seconds > MAX_SECONDS:
        dest.unlink(missing_ok=True)
        raise RenderError(
            f"that is {seconds / 60:.1f} minutes long; the limit is "
            f"{MAX_SECONDS / 60:.0f}. Cut it down and try again.")

    meta = {
        "id": upload_id,
        "name": filename or dest.name,
        "seconds": round(seconds, 2),
        "sample_rate": sr,
        "channels": int(data.shape[1]),
        "peak_db": round(audio.peak_db(data), 1),
        "rms_db": round(audio.rms_db(data), 1),
        "at": time.time(),
    }
    (upload_dir() / f"{upload_id}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def path(upload_id: str) -> Path:
    """Where an upload lives. Rejects anything that is not a plain id.

    The id goes into a filename, so it is checked rather than trusted: this is
    reached over HTTP and "../" is the first thing anyone tries.
    """
    if not upload_id or not upload_id.isalnum():
        raise RenderError("that is not an upload id")
    found = upload_dir() / f"{upload_id}.wav"
    if not found.exists():
        raise RenderError("that upload has expired or was never here")
    return found


def meta(upload_id: str) -> dict | None:
    side = upload_dir() / f"{upload_id}.json"
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
