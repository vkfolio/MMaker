"""Where audio lands.

Files, no database. A render is a file plus the parameters that made it, and
nothing is ever overwritten -- a second render of the same melody is a second
file. That makes A/B free and makes the client's cache safe, because a path
names one piece of audio for good.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .config import settings


def audio_dir() -> Path:
    d = settings.data_dir / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_dir() -> Path:
    d = settings.data_dir / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_name(kind: str, extension: str = ".wav") -> tuple[str, Path]:
    """A fresh (relative name, absolute path) pair."""
    stamp = int(time.time() * 1000)
    name = f"{kind}_{stamp}_{uuid.uuid4().hex[:8]}{extension}"
    return name, audio_dir() / name


def resolve(relative: str) -> Path:
    """Absolute path for something under the audio directory.

    Rejects anything that escapes it. This is served over HTTP, so "../" is not
    a hypothetical -- it is the first thing anyone tries.
    """
    root = audio_dir().resolve()
    candidate = (root / relative).resolve()
    # is_relative_to, not a string prefix: "/data/p1" prefixes "/data/p1-other",
    # so prefix matching lets a sibling directory through.
    if not candidate.is_relative_to(root):
        raise ValueError("that path is outside the audio directory")
    return candidate


def write_sidecar(relative: str, params: dict) -> None:
    """Record what produced a file, next to it.

    An audio file whose settings are only in a log is a file you cannot
    reproduce, and reproducing a render is how you tell a model change from a
    parameter change.
    """
    path = audio_dir() / (relative + ".json")
    payload = dict(params)
    payload["written_at"] = time.time()
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def sidecar(relative: str) -> dict | None:
    path = audio_dir() / (relative + ".json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
