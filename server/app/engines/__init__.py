"""Engine selection.

One place decides whether a capability is served by a real model or the stub, so
routers never branch on backend availability. `MUSICMAKER_ENGINE=stub` forces the
synthetic engine for local development regardless of what is installed.
"""

from __future__ import annotations

import os
from functools import lru_cache

from ..config import settings
from .base import Engine, EngineError, NotSupported
from .stub import StubEngine

_FORCED = os.environ.get("MUSICMAKER_ENGINE", "").strip().lower()


def _forced_stub() -> bool:
    return _FORCED == "stub" or settings.stub_models


@lru_cache(maxsize=1)
def music() -> Engine:
    """Generation, layering, repaint, extend."""
    if _forced_stub():
        return StubEngine()
    from .acestep import AceStepEngine
    return AceStepEngine(debug_dir=settings.data_dir / "_raw")


@lru_cache(maxsize=1)
def separator() -> Engine:
    if _forced_stub():
        return StubEngine()
    from .extras import DemucsEngine
    return DemucsEngine()


@lru_cache(maxsize=1)
def voice() -> Engine:
    if _forced_stub():
        return StubEngine()
    from .extras import SeedVCEngine
    return SeedVCEngine(checkpoint_dir=settings.models_dir / "seedvc")


@lru_cache(maxsize=1)
def sfx() -> Engine:
    if _forced_stub():
        return StubEngine()
    from .extras import StableAudioEngine
    return StableAudioEngine(model_dir=settings.models_dir / "stableaudio")


def reset_cache():
    """Tests flip env vars between cases; caches must not outlive that."""
    for fn in (music, separator, voice, sfx):
        fn.cache_clear()


def active() -> dict:
    return {
        "music": music().name,
        "separator": separator().name,
        "voice": voice().name,
        "sfx": sfx().name,
        "forced_stub": _forced_stub(),
    }


__all__ = ["Engine", "EngineError", "NotSupported", "StubEngine",
           "music", "separator", "voice", "sfx", "reset_cache", "active"]
