"""The three things an engine can be.

Keeping these narrow is what lets the whole API be built and tested against the
stub, so that when a pod appears only the adapters need proving. musicmaker
learned this the expensive way -- 29 tests pass there today without a GPU
because the seam is this small.

`report` is a callback taking (progress 0..1, message). Pass a negative progress
to say the step is indeterminate rather than inventing a percentage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Sampler(Protocol):
    """Notes to audio, deterministically. No model, no GPU, no network."""
    name: str

    def render(self, dest: Path, notes, bpm: float, preset, report=None) -> Path:
        ...


class Refiner(Protocol):
    """Audio to audio: the pass that turns a triggered sample into a performance.

    This is the SDEdit half of the CoSaRef pattern (arXiv 2410.16785) -- add
    noise to a sampler render, denoise it back conditioned on a prompt. The
    melody has to survive, which is exactly what gate/refine_fidelity.py exists
    to measure before anything here is trusted.
    """
    name: str

    def refine(self, dest: Path, src_audio: Path, prompt: str, strength: float,
               seed: int, report=None) -> Path:
        ...


class Singer(Protocol):
    """Notes plus syllables plus a reference clip, to a sung vocal."""
    name: str

    def sing(self, dest: Path, notes, bpm: float, reference: Path | None,
             language: str = "en", seed: int = 0, report=None) -> Path:
        ...
