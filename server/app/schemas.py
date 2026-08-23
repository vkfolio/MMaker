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
    # Which separation run produced this stem. Splitting again appends a new
    # set rather than destroying the old one, so ids stay valid forever and a
    # client that saved a reference to this stem can still resolve it.
    split_id: str = ""
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
    # What the engine said it rendered with, so a silent model fallback is
    # visible in the data rather than only as an oddly fast render.
    engine: dict = Field(default_factory=dict)
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
    # The split set currently shown and mixed. Older sets stay in `stems`.
    active_split: str | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def stem(self, stem_id: str) -> Stem | None:
        return next((s for s in self.stems if s.id == stem_id), None)

    def variation(self, var_id: str) -> Variation | None:
        return next((v for v in self.variations if v.id == var_id), None)

    def active_stems(self) -> list[Stem]:
        """Stems from the current split set (all of them if none is marked)."""
        if not self.active_split:
            return list(self.stems)
        return [s for s in self.stems if s.split_id == self.active_split]

    def audible_stems(self) -> list[Stem]:
        """Solo wins over mute, as in every DAW."""
        active = self.active_stems()
        soloed = [s for s in active if s.soloed]
        pool = soloed or [s for s in active if not s.muted]
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
    # basic        -> vocals + instrumental (2)
    # professional -> 6 stems (adds guitar and piano)
    # advanced     -> everything the separator detects
    tier: Literal["basic", "professional", "advanced"] = "professional"
    # Not supported yet: it needs a de-reverb model we do not host. Asking for
    # it is accepted and reported back as unfulfilled rather than ignored.
    remove_reverb: bool = False


class RepaintRequest(BaseModel):
    """A region to regenerate, in bars or in seconds.

    ACE-Step's repaint takes seconds, so whole-bar quantisation was ours, not
    the model's. A timeline drag lands anywhere, so seconds are the honest
    interface; bars stay for callers that think in bars.
    """
    start_bar: int | None = Field(None, ge=1)
    end_bar: int | None = Field(None, ge=2)
    start_s: float | None = Field(None, ge=0)
    end_s: float | None = Field(None, gt=0)
    prompt: str | None = None
    seed: int | None = None
    # ACE-Step's repaint_strength. Plumbed because it is the documented dial
    # and a fuller checkpoint may honour it, but MEASURED TO DO NOTHING on
    # acestep-v15-turbo: 0.1, 0.3, 0.6 and 0.9 all returned mel-similarity
    # 0.844 to the source, identical to three decimal places. Do not put a
    # slider on it without re-measuring on the checkpoint in use.
    strength: float | None = Field(None, ge=0.0, le=1.0)

    def resolve(self, grid) -> tuple[float, float]:
        """Seconds win when given; otherwise convert the bar range."""
        if self.start_s is not None and self.end_s is not None:
            return float(self.start_s), float(self.end_s)
        if self.start_bar is not None and self.end_bar is not None:
            return grid.bar_to_seconds(self.start_bar), grid.bar_to_seconds(self.end_bar)
        raise ValueError("give either start_bar/end_bar or start_s/end_s")


class CoverRequest(BaseModel):
    """Reimagine existing audio.

    WARNING -- measured on acestep-v15-turbo, this does not work. A cover of a
    synth arpeggio prompted "solo violin, slow sad ballad, no drums, no synth"
    came back as the input: mel-similarity 0.991, sample correlation 0.935. The
    prompt is ignored, and so are both strength parameters -- cover_noise_
    strength 0.5 and 0.9 produced bit-identical output, with and without the
    language model enabled. The one setting that changes anything is
    cover_noise_strength = 0, which takes the branch that discards the source
    latents entirely, so it is text2music rather than a variation.

    Use repaint over the region instead: it genuinely re-performs (mel 0.844,
    correlation 0.000) in about seven seconds, and it is the path the desktop
    offers. This is kept because it is correct against a non-turbo checkpoint,
    where cover is a supported and meaningful task.

    Original description follows.

    Reimagine existing audio, with a dial for how far to stray.

    This is the answer to "here is my idea, play with it". ACE-Step's cover
    task conditions generation on source audio, and two parameters control how
    tightly: audio_cover_strength (the docs suggest ~0.2 for style transfer,
    1.0 to stay close) and cover_noise_strength (0 starts from pure noise, 1
    stays nearest the source). Both were previously left at their API defaults,
    which is the maximally faithful corner -- so every "variation" was a near
    copy and there was no way to ask for anything looser.

    One `strength` in 0..1 drives both, because they are not independent in
    any way a person would want to reason about: what is being asked for is a
    single "how much of my performance survives".

    A region is optional. Given one, only that slice is covered and spliced
    back, so a variation can apply to four bars of one layer.
    """
    prompt: str | None = None
    # 0 = improvise freely on the idea, 1 = stay faithful to the recording.
    strength: float = Field(0.5, ge=0.0, le=1.0)
    seed: int | None = None
    start_s: float | None = Field(None, ge=0)
    end_s: float | None = Field(None, gt=0)

    def region(self) -> tuple[float, float] | None:
        """The slice to cover, or None for the whole stem."""
        if self.start_s is None or self.end_s is None:
            return None
        if self.end_s <= self.start_s:
            raise ValueError("end_s must be greater than start_s")
        return float(self.start_s), float(self.end_s)

    # Diagnostic overrides. The two model parameters interact in ways the
    # single dial cannot express -- audio_cover_strength below 1.0 switches on
    # an extra text-conditioning branch, and cover_noise_strength snaps to the
    # nearest step in an 8-step schedule -- so being able to set them directly
    # is how the dial's mapping gets characterised rather than guessed.
    audio_cover_strength: float | None = Field(None, ge=0.0, le=1.0)
    cover_noise_strength: float | None = Field(None, ge=0.0, le=1.0)
    # ACE-Step skips its language model on cover and drives the DiT directly.
    # The LM is what turns a prompt into semantic audio codes, so without it a
    # cover may have nothing to reinterpret the source *towards*.
    thinking: bool | None = None

    def engine_params(self) -> dict:
        """Map one dial onto the model's two.

        Faithful (1.0) means full conditioning strength and high noise
        retention; improvise (0.0) means light conditioning and start closer to
        noise. The floor on cover_strength is deliberate: at 0 the source stops
        influencing the result at all, which is not a variation of anything.
        """
        s = float(self.strength)
        params = {
            "audio_cover_strength": round(0.2 + 0.8 * s, 3),
            "cover_noise_strength": round(s, 3),
        }
        if self.audio_cover_strength is not None:
            params["audio_cover_strength"] = round(self.audio_cover_strength, 3)
        if self.cover_noise_strength is not None:
            params["cover_noise_strength"] = round(self.cover_noise_strength, 3)
        if self.thinking is not None:
            params["thinking"] = bool(self.thinking)
        return params



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
    # Optional region. Without it the layer spans the whole arrangement, which
    # is all the API could express before.
    start_s: float | None = Field(None, ge=0)
    end_s: float | None = Field(None, gt=0)


class VocalRequest(BaseModel):
    lyrics: str | None = None
    vocal_language: str | None = None
    voice_id: str | None = None
    seed: int | None = None


class VoiceRequest(BaseModel):
    voice_id: str
    # Optional region: convert only part of a stem and splice it back.
    start_s: float | None = Field(None, ge=0)
    end_s: float | None = Field(None, gt=0)


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
