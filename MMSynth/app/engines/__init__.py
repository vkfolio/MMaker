"""Engine selection.

Three jobs, three engines, and the split is the finding from Phase 0 rather than
a preference:

  sampler    FluidSynth. Not the product -- the *carrier*. ACE-Step cannot read
             notes, so a melody only reaches it as audio.
  music      ACE-Step 1.5 XL. Re-timbres the carrier through `cover`, which
             holds the melody in FSQ semantic codes while the caption changes
             the instrument. Measured at 0.00 semitones of drift.
  singer     SoulX-Singer. ACE-Step will not sing a given melody -- gate 2a
             found no setting where a voice appears and the tune survives -- so
             note-exact singing needs a singing-voice-synthesis model. This is
             the same class of model ACE Studio's own singer is; ACE-Step is
             their *music* model and was never the thing doing the singing.

`MMSYNTH_ENGINE=stub` forces the synthetic ones, which is what lets the whole
API be tested on a laptop with no GPU.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import settings
from .base import Refiner, Sampler, Singer
from .stub import StubRefiner, StubSampler, StubSinger


def _forced_stub() -> bool:
    return settings.engine == "stub" or settings.stub_models


@lru_cache(maxsize=1)
def sampler() -> Sampler:
    """What carries the notes into a model that cannot read them.

    The module is `fluidsynth`, not `sampler`, and that matters: importing
    `.sampler` inside a function called `sampler()` binds the submodule onto
    this package and silently replaces the function. It works until the first
    render, then every later call raises "module object is not callable".
    """
    if _forced_stub():
        return StubSampler()
    from .fluidsynth import FluidSynthSampler
    return FluidSynthSampler()


@lru_cache(maxsize=1)
def music():
    """ACE-Step 1.5 XL: `cover` for instruments, and re-voicing a sung take."""
    if _forced_stub():
        return StubRefiner()
    from .acestep import AceStepEngine
    return AceStepEngine()


@lru_cache(maxsize=1)
def singer() -> Singer:
    if _forced_stub():
        return StubSinger()
    from .soulx import SoulXSinger
    return SoulXSinger()


def reset_cache() -> None:
    """Tests flip environment variables between cases; caches must not outlive that."""
    for fn in (sampler, music, singer):
        fn.cache_clear()


def active() -> dict:
    return {
        "sampler": sampler().name,
        "music": music().name,
        "singer": singer().name,
        "forced_stub": _forced_stub(),
        "refine_enabled": settings.refine_enabled,
    }


__all__ = ["Sampler", "Refiner", "Singer", "sampler", "music", "singer",
           "reset_cache", "active"]
