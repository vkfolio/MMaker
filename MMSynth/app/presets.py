"""The instrument library, and what a quick action is allowed to change.

Quick actions are a closed set (app/schemas.py) and this is where each one gets
its meaning. Keeping the whole mapping in one readable table is the point: a
user who asks for "brighter" and gets something else should be able to be shown
why, and so should whoever maintains this next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from .config import settings
from .errors import NotReady
from .score import Humanise

# Timbre words go to the caption; the model is what hears them. Each also nudges
# `cover_noise_strength`, and the sign is counter-intuitive: LOWER noise gives
# the model MORE freedom to reinterpret, because with `cover` the melody is held
# by the FSQ semantic codes rather than by the waveform. Measured in PHASE0.md --
# every noise level from 0.15 to 0.90 kept the tune to within 0.03 semitones.
# So a word asking for more change lowers the noise.
TIMBRE = {
    "brighter":  ("bright, present, lifted highs", -0.06),
    "darker":    ("dark, mellow, rolled-off highs", -0.06),
    "warmer":    ("warm, analog, tape saturation", -0.08),
    "cleaner":   ("clean, precise, studio, controlled", +0.06),
    "more_air":  ("airy, breathy, open, close mic", -0.07),
    "more_bite": ("biting, hard attack, edge, aggressive", -0.10),
}

# Space words are ours, not the model's -- reverb is convolution, not inference.
SPACE = {"wetter": +0.15, "drier": -0.12}

# Melody actions become deterministic transforms, applied before anything else.
MELODY = {"up_octave", "down_octave", "swing", "straighten",
          "half_time", "double_time"}

# Beyond this the prompt stops sharpening and starts diluting. musicmaker
# measured the same thing and landed in the same place (server/app/presets.py).
MAX_TAGS = 16


@dataclass
class Refine:
    """What ACE-Step is told, and how much room it gets.

    `noise` is `cover_noise_strength` and `strength` is `audio_cover_strength`.
    They are different things and the names in ACE-Step's own docs invite
    confusing them -- see app/engines/acestep.py.
    """
    prompt: str = ""
    noise: float = 0.35
    strength: float = 1.0


@dataclass
class Space:
    mix: float = 0.18
    decay_s: float = 1.6


@dataclass
class Instrument:
    id: str
    label: str
    track_class: str = "keyboard"
    program: int = 0
    sfz: str = ""
    humanise: dict = field(default_factory=dict)
    refine: Refine = field(default_factory=Refine)
    space: Space = field(default_factory=Space)

    def public(self) -> dict:
        return {"id": self.id, "label": self.label,
                "track_class": self.track_class,
                "description": self.refine.prompt}


@lru_cache(maxsize=1)
def library() -> dict[str, Instrument]:
    path = settings.instruments_path
    if not path.exists():
        raise NotReady(f"no instrument library at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, Instrument] = {}
    for entry in raw.get("instruments", []):
        out[entry["id"]] = Instrument(
            id=entry["id"],
            label=entry.get("label", entry["id"]),
            track_class=entry.get("track_class", "keyboard"),
            program=int(entry.get("program", 0)),
            sfz=entry.get("sfz", ""),
            humanise=entry.get("humanise", {}) or {},
            refine=Refine(**(entry.get("refine", {}) or {})),
            space=Space(**(entry.get("space", {}) or {})),
        )
    if not out:
        raise NotReady(f"{path} lists no instruments")
    return out


def reset_cache() -> None:
    library.cache_clear()


def get(instrument_id: str) -> Instrument:
    lib = library()
    if instrument_id not in lib:
        known = ", ".join(sorted(lib)[:8])
        raise NotReady(f"no instrument called {instrument_id!r}; try one of: {known}")
    return lib[instrument_id]


@dataclass
class Resolved:
    """One render's settings, after the preset and the actions have been merged."""
    instrument: Instrument
    humanise: Humanise
    refine: Refine
    space: Space
    transforms: list[str]


def resolve(instrument_id: str, actions=(), prompt: str = "") -> Resolved:
    inst = get(instrument_id)

    human = Humanise(**inst.humanise).with_actions(actions)

    tags: list[str] = []
    if inst.refine.prompt:
        tags.extend(t.strip() for t in inst.refine.prompt.split(",") if t.strip())
    noise = inst.refine.noise
    for a in actions:
        if a in TIMBRE:
            words, delta = TIMBRE[a]
            tags.extend(t.strip() for t in words.split(",") if t.strip())
            noise += delta
    # The user's own words go last, where they read as the qualifier rather than
    # the subject -- the instrument is still the instrument.
    for word in (t.strip() for t in prompt.split(",")):
        if word:
            tags.append(word)

    seen: list[str] = []
    for tag in tags:
        if tag.lower() not in {s.lower() for s in seen}:
            seen.append(tag)

    mix = inst.space.mix
    for a in actions:
        mix += SPACE.get(a, 0.0)

    return Resolved(
        instrument=inst,
        humanise=human,
        refine=Refine(prompt=", ".join(seen[:MAX_TAGS]),
                      noise=round(max(0.05, min(0.95, noise)), 3),
                      strength=inst.refine.strength),
        space=Space(mix=max(0.0, min(0.9, mix)), decay_s=inst.space.decay_s),
        transforms=[a for a in actions if a in MELODY],
    )
