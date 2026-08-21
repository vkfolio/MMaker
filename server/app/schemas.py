"""Domain types.

The central invariant: audio is never mutated. Editing a stem appends a new
StemVersion. That makes undo free, keeps every stem independent, and means the
exact params that produced any audio file are always recoverable.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

TrackClass = Literal[
    "vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard",
    "percussion", "strings", "synth", "fx", "brass", "woodwinds", "mix",
]

IdeaKind = Literal["prompt", "midi", "audio"]


def _id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Grid(BaseModel):
    """The sync contract. Every generation call inherits these."""
    bpm: int = Field(96, ge=30, le=300)
    key_scale: str = "A Minor"
    time_sig: str = "4/4"
    bars: int = Field(32, ge=1, le=512)
    # Set once detection has been confirmed by a human. Until then we refuse to
    # layer, because a wrong key silently corrupts every later stem.
    confirmed: bool = False

    @property
    def beats_per_bar(self) -> int:
        return int(self.time_sig.split("/")[0])

    @property
    def duration_s(self) -> float:
        return self.bars * self.beats_per_bar * 60.0 / self.bpm

    def bar_to_seconds(self, bar: int) -> float:
        return (bar - 1) * self.beats_per_bar * 60.0 / self.bpm


class SyncReport(BaseModel):
    detected_bpm: float | None = None
    median_dev_ms: float | None = None
    p90_dev_ms: float | None = None
    verdict: Literal["pass", "marginal", "fail", "unmeasured"] = "unmeasured"
    nudge_ms: float = 0.0


class StemVersion(BaseModel):
    id: str = Field(default_factory=lambda: _id("ver"))
    audio: str                       # path relative to the project dir
    op: str                          # generate | repaint | extend | vary | voice | import
    params: dict = Field(default_factory=dict)
    sync: SyncReport = Field(default_factory=SyncReport)
    created_at: float = Field(default_factory=time.time)
    note: str = ""


class Stem(BaseModel):
    id: str = Field(default_factory=lambda: _id("stem"))
    track_class: TrackClass
    label: str = ""
    prompt: str = ""
    versions: list[StemVersion] = Field(default_factory=list)
    current: int = 0                 # index into versions
    gain_db: float = 0.0
    pan: float = 0.0                 # -1 L .. +1 R
    muted: bool = False
    soloed: bool = False
    voice_id: str | None = None

    @property
    def version(self) -> StemVersion | None:
        if not self.versions:
            return None
        return self.versions[max(0, min(self.current, len(self.versions) - 1))]

    def add(self, version: StemVersion) -> StemVersion:
        self.versions.append(version)
        self.current = len(self.versions) - 1
        return version


class Variation(BaseModel):
    id: str = Field(default_factory=lambda: _id("var"))
    seed: int
    audio: str
    prompt: str = ""
    created_at: float = Field(default_factory=time.time)


class Project(BaseModel):
    id: str = Field(default_factory=lambda: _id("prj"))
    title: str = "Untitled"
    instrumental: bool = False
    quality: Literal["fast", "high", "ultra"] = "high"
    grid: Grid = Field(default_factory=Grid)
    idea_kind: IdeaKind = "prompt"
    prompt: str = ""
    style: str = ""
    lyrics: str = ""
    vocal_language: str = "en"
    source_audio: str | None = None   # the uploaded/rendered idea, if any
    variations: list[Variation] = Field(default_factory=list)
    chosen_variation: str | None = None
    stems: list[Stem] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def stem(self, stem_id: str) -> Stem | None:
        return next((s for s in self.stems if s.id == stem_id), None)

    def variation(self, var_id: str) -> Variation | None:
        return next((v for v in self.variations if v.id == var_id), None)

    def audible_stems(self) -> list[Stem]:
        """Solo wins over mute, as in every DAW."""
        soloed = [s for s in self.stems if s.soloed]
        pool = soloed or [s for s in self.stems if not s.muted]
        return [s for s in pool if s.version]


# ---- request bodies -------------------------------------------------------

class CreateProject(BaseModel):
    title: str = "Untitled"
    prompt: str = ""
    style: str = ""
    lyrics: str = ""
    # ACE-Step sings by default; instrumental is the deliberate choice.
    instrumental: bool = False
    quality: Literal["fast", "high", "ultra"] = "high"
    vocal_language: str = "en"
    bpm: int | None = None
    key_scale: str | None = None
    bars: int = 32
    variations: int = Field(4, ge=1, le=8)


class ConfirmGrid(BaseModel):
    bpm: int = Field(..., ge=30, le=300)
    key_scale: str
    bars: int = Field(..., ge=1, le=512)
    time_sig: str = "4/4"


class SplitRequest(BaseModel):
    variation_id: str
    method: Literal["demucs", "extract"] = "demucs"


class RepaintRequest(BaseModel):
    start_bar: int = Field(..., ge=1)
    end_bar: int = Field(..., ge=2)
    prompt: str | None = None
    seed: int | None = None


class ExtendRequest(BaseModel):
    bars: int = Field(8, ge=1, le=128)
    prompt: str | None = None


class VaryRequest(BaseModel):
    seed: int | None = None
    prompt: str | None = None


class AddLayerRequest(BaseModel):
    track_class: TrackClass
    prompt: str = ""
    seed: int | None = None


class VocalRequest(BaseModel):
    lyrics: str | None = None
    vocal_language: str | None = None
    voice_id: str | None = None
    seed: int | None = None


class VoiceRequest(BaseModel):
    voice_id: str


class MixState(BaseModel):
    gain_db: float | None = None
    pan: float | None = None
    muted: bool | None = None
    soloed: bool | None = None


class SfxRequest(BaseModel):
    prompt: str
    duration_s: float = Field(5.0, gt=0, le=60)
    variations: int = Field(3, ge=1, le=8)
    seed: int | None = None
