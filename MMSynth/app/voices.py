"""The singer library.

SoulX-Singer is zero-shot: a voice is a short reference clip plus metadata, and
no per-voice training happens. That is the same shape musicmaker's voice library
already has (server/app/voices.py), and it is what makes "pick a singer" cheap.

It also means the hard part is not technical. A reference clip is a real
person's voice, so every entry carries its provenance, and a voice with no
recorded consent is listed as unusable rather than quietly synthesised. The clip
is the easy half; the paperwork is the product.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .errors import NotReady

HERE = Path(__file__).resolve().parent.parent


def library_dir() -> Path:
    """Where voices live.

    Bundled voices ship in the repo; anything the user adds goes in the data
    directory, which on a pod is the network volume and survives redeploys.
    """
    d = settings.data_dir / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dirs():
    yield from sorted((HERE / "voices").glob("*/"))
    yield from sorted(library_dir().glob("*/"))


def _read(folder: Path) -> dict | None:
    meta = folder / "voice.json"
    if not meta.exists():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    data["_dir"] = str(folder)
    return data


def _consented(data: dict) -> bool:
    """Whether this voice may be sung with.

    Two things must both be true and both be recorded: a signed agreement
    covering AI singing-voice use, and a recorded verbal consent from the
    person. This mirrors what ACE Studio and ElevenLabs require before they will
    train or clone, and it is the minimum a product should ask of itself.
    """
    consent = data.get("consent") or {}
    return bool(consent.get("agreement")) and bool(consent.get("recorded"))


def public(data: dict) -> dict:
    return {
        "id": data.get("id", ""),
        "label": data.get("label", data.get("id", "")),
        "description": data.get("description", ""),
        "range": data.get("range", ""),
        "language": data.get("language", "en"),
        "builtin": bool(data.get("builtin", False)),
        "usable": _consented(data),
        "consent": {
            "agreement": bool((data.get("consent") or {}).get("agreement")),
            "recorded": bool((data.get("consent") or {}).get("recorded")),
        },
    }


def list_voices() -> list[dict]:
    out = []
    for folder in _dirs():
        data = _read(folder)
        if data:
            out.append(public(data))
    return sorted(out, key=lambda v: (not v["builtin"], v["label"]))


def find(voice_id: str) -> dict | None:
    for folder in _dirs():
        data = _read(folder)
        if data and data.get("id") == voice_id:
            return data
    return None


@dataclass
class Reference:
    """What SoulX-Singer needs to imitate a voice.

    Not just a clip. The model conditions on a prompt whose notes and phonemes
    it also knows, so a voice is an audio file **and** its score metadata -- the
    same JSON shape a target uses. That is more than "a reference clip plus
    metadata" implies, and it is why adding a voice from a bare recording needs
    a transcription and alignment step rather than a file copy.
    """
    wav: Path
    meta: Path
    label: str = ""


def reference(voice_id: str) -> Reference:
    """The voice to condition on. Raises with something actionable if it cannot."""
    data = find(voice_id)
    if data is None:
        known = ", ".join(v["id"] for v in list_voices()[:8]) or "none installed"
        raise NotReady(f"no voice called {voice_id!r}; installed: {known}")

    if not _consented(data):
        if not settings.allow_unconsented_voices:
            raise NotReady(
                f"{voice_id!r} has no recorded consent for AI singing-voice use, "
                "so it will not be sung with. Add `consent.agreement` and "
                "`consent.recorded` to its voice.json once both exist.")

    folder = Path(data["_dir"])
    clip = folder / data.get("clip", "reference.wav")
    meta = folder / data.get("metadata", "reference.json")
    if not clip.exists():
        raise NotReady(f"{voice_id!r} has no reference clip at {clip.name}")
    if not meta.exists():
        raise NotReady(
            f"{voice_id!r} has a clip but no score metadata at {meta.name}. "
            "SoulX-Singer conditions on the prompt's notes and phonemes as well "
            "as its audio, so a voice needs both.")
    return Reference(wav=clip, meta=meta, label=data.get("label", voice_id))


def add(clip_path, label: str, description: str = "", language: str = "en",
        agreement: str = "", recorded: bool = False) -> dict:
    """Add a voice from a clip the user supplied.

    Consent is a parameter rather than an afterthought so that the call site has
    to say something about it. The default is "no", and a voice with no consent
    is stored and listed but refused at render time.
    """
    voice_id = uuid.uuid4().hex[:12]
    folder = library_dir() / voice_id
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / "reference.wav"
    shutil.copyfile(str(clip_path), dest)

    data = {
        "id": voice_id,
        "label": label or "Untitled voice",
        "description": description,
        "language": language,
        "clip": "reference.wav",
        "builtin": False,
        "added_at": time.time(),
        "consent": {"agreement": agreement, "recorded": bool(recorded)},
    }
    (folder / "voice.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")
    data["_dir"] = str(folder)
    return public(data)
