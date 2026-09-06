"""The wire.

Times are in **seconds**, matching musicmaker's ScoreNote, and for the same
reason: a sample rate belongs to an audio device, not to music, so the client
converts at its edge and nothing downstream has to know the rate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The quick actions. A closed set, on purpose.
#
# These are the only things a prompt is allowed to turn into -- see
# app/arrange.py. An open-ended "the model writes notes" path measures at a 12%
# pass rate (Libretto, arXiv 2606.22708), and editing an existing piece is the
# harder half of that. A closed set is checkable, reversible, and explainable to
# the user, which an emitted note stream is not.
QuickAction = Literal[
    # timbre, applied to the refine prompt and the preset
    "brighter", "darker", "warmer", "cleaner", "more_air", "more_bite",
    # performance, applied to the humanise rules
    "softer", "louder", "more_legato", "more_staccato", "tighter", "looser",
    # space, applied to the reverb
    "wetter", "drier",
    # the melody itself, applied as a deterministic transform
    "up_octave", "down_octave", "swing", "straighten", "half_time", "double_time",
]

TRACK_CLASSES = (
    "keyboard", "guitar", "bass", "strings", "brass", "woodwinds",
    "synth", "percussion", "vocals",
)


class Note(BaseModel):
    """One note, in seconds, relative to the start of the melody."""
    start: float = Field(..., ge=0)
    length: float = Field(..., gt=0)
    pitch: int = Field(..., ge=0, le=127)
    velocity: int = Field(96, ge=1, le=127)
    # One syllable, and never empty for long: a note with no lyric is given
    # `da` on the way in, so a melody drawn without a thought about words is
    # still singable. `-` holds the previous syllable -- a melisma, said out
    # loud rather than implied by an absence.
    lyric: str = ""
    # ARPAbet, lower-cased for display: the [t][w][ih] row above the note.
    # Empty means "derive it from the lyric"; anything else was edited by hand
    # and is preserved.
    phonemes: list[str] = []


class RenderRequest(BaseModel):
    """Play a melody on an AI instrument."""
    notes: list[Note] = Field(..., min_length=1)
    bpm: float = Field(120.0, ge=20, le=300)
    instrument_id: str = "grand_piano"
    actions: list[QuickAction] = []
    # Free text. Composed into the refine prompt; never turned into notes.
    prompt: str = ""
    seed: int | None = None
    # Skip the neural pass. The sampler alone is offline, deterministic and
    # immediate, which is what you want while drawing notes.
    dry: bool = False


class SingRequest(BaseModel):
    """Sing a melody.

    No prose lyrics field: by the time you are rendering, the words live on the
    notes. That is the only place a syllable and a pitch cannot disagree, and
    accepting prose here as well would create a second source of truth whose
    only job would be to contradict the first. Use POST /lyrics to turn a typed
    line into notes, look at it, then send the notes.
    """
    notes: list[Note] = Field(..., min_length=1)
    bpm: float = Field(120.0, ge=20, le=300)
    voice_id: str = ""
    language: Literal["en"] = "en"
    seed: int | None = None
    # Run the sung take back through ACE-Step. Gives another, usually richer
    # take of the same performance -- it does not change who is singing, which
    # is what voice_id is for.
    revoice: bool = False
    revoice_prompt: str = ""


class LyricsRequest(BaseModel):
    """Spread a typed line across the notes.

    Pure: it changes nothing and returns what the notes *would* become. The
    document lives in the client, so the client can show the `replaced` count,
    ask, and then keep or discard the result.
    """
    notes: list[Note] = Field(..., min_length=1)
    line: str = ""


class Assigned(BaseModel):
    lyric: str
    phonemes: list[str] = []
    # [t][w][ih] -- the row as it should be drawn, so every client renders it
    # the same way rather than each inventing its own bracket style.
    display: str = ""


class LyricsResponse(BaseModel):
    notes: list[Assigned]
    # Syllables with no note yet. Ordinary mid-edit state, not an error.
    pending: list[str] = []
    held: int = 0
    # How many hand-set notes this would overwrite. The client warns with it.
    replaced: int = 0
    line: str = ""
    note: str = ""


class PhonemesRequest(BaseModel):
    """ARPAbet for words, for the row above the notes."""
    words: list[str] = Field(..., min_length=1)


class EnhanceRequest(BaseModel):
    """Re-voice audio you already have, without moving a note.

    `keep_arrangement` is the honest knob. A single instrument line can take a
    lot of freedom and still be recognisably itself; a full mix cannot, because
    ACE-Step is free to reinterpret harmony and instrumentation, not just
    timbre. Setting it raises the noise floor so the source is held harder.
    """
    upload_id: str
    prompt: str = ""
    instrument_id: str = ""
    actions: list[QuickAction] = []
    keep_arrangement: bool = False
    seed: int | None = None


class TranscribeRequest(BaseModel):
    """Read a melody out of audio you already have."""
    upload_id: str
