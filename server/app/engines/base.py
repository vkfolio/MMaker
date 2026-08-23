"""The engine interface every backend implements.

Keeping this narrow is what lets the entire API be developed and tested against
the stub, so that when a pod appears only the adapters need validating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..schemas import Grid


class Engine(Protocol):
    name: str

    def generate(self, dest: Path, prompt: str, grid: Grid, seed: int,
                 lyrics: str = "", vocal_language: str = "en",
                 src_audio: Path | None = None,
                 cover: dict | None = None) -> Path:
        """text2music, or cover when src_audio is given.

        `cover` carries the audio-conditioning strengths for the cover case.
        """

    def layer(self, dest: Path, track_class: str, src_audio: Path, prompt: str,
              grid: Grid, seed: int, lyrics: str = "",
              vocal_language: str = "en") -> Path:
        """ACE-Step lego -- add a track that fits what it hears."""

    def repaint(self, dest: Path, src_audio: Path, start_s: float, end_s: float,
                prompt: str, grid: Grid, seed: int) -> Path:
        """Regenerate one time region, preserve the rest."""

    def extend(self, dest: Path, src_audio: Path, added_s: float, prompt: str,
               grid: Grid, seed: int) -> Path:
        """ACE-Step complete."""

    def separate(self, src_audio: Path, out_dir: Path) -> dict[str, Path]:
        """Split a mix into named stems."""

    def convert_voice(self, dest: Path, src_audio: Path, reference: Path) -> Path:
        """Singing voice conversion onto a reference timbre."""

    def sfx(self, dest: Path, prompt: str, duration_s: float, seed: int) -> Path:
        """Sound effect / texture generation."""


class EngineError(RuntimeError):
    pass


class NotSupported(EngineError):
    """Raised when a backend cannot do something the API offers."""
