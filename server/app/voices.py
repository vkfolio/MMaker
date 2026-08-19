"""The voice library.

ACE-Step has no voice bank -- prompt for "female soul vocal" and you get a
different singer every render. A voice here is just a reference clip plus
metadata, because Seed-VC conversion is zero-shot and needs no training. That
gives predefined voice selection, cloning from a short upload, and consistency
across songs, all from one mechanism.

Rights: every clip in a shipped library needs clear permission. Cloning a real
singer without it is not something to enable by default.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from .config import settings


def library_dir() -> Path:
    d = settings.data_dir / "voices"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_path(voice_id: str) -> Path:
    return library_dir() / voice_id / "voice.json"


def list_voices() -> list[dict]:
    out = []
    for meta in library_dir().glob("*/voice.json"):
        try:
            out.append(json.loads(meta.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(out, key=lambda v: (not v.get("builtin", False), v.get("label", "")))


def get(voice_id: str) -> dict | None:
    path = _meta_path(voice_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def reference_path(voice_id: str) -> Path | None:
    meta = get(voice_id)
    if not meta:
        return None
    path = library_dir() / voice_id / meta["reference"]
    return path if path.exists() else None


def add(label: str, source_audio, description: str = "", builtin: bool = False,
        consent: str = "") -> dict:
    """Register a reference clip as a selectable voice."""
    voice_id = f"voice_{uuid.uuid4().hex[:8]}"
    d = library_dir() / voice_id
    d.mkdir(parents=True, exist_ok=True)

    source_audio = Path(source_audio)
    ref_name = f"reference{source_audio.suffix or '.wav'}"
    shutil.copy2(source_audio, d / ref_name)

    meta = {
        "id": voice_id,
        "label": label,
        "description": description,
        "reference": ref_name,
        "builtin": builtin,
        # Recorded, not enforced -- but an empty value is a visible reminder that
        # the rights question was never answered for this clip.
        "consent": consent,
        "created_at": time.time(),
    }
    _meta_path(voice_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def delete(voice_id: str) -> bool:
    d = library_dir() / voice_id
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True
