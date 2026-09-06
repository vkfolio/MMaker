"""The symbolic layer: notes in, notes out, and a Standard MIDI File at the end.

Nothing here is a model. Everything is arithmetic you can read, which is the
point -- this is the layer that must stay predictable so the neural layer above
it can be judged.

On humanisation being rules: RenCon 2025 (arXiv 2605.02059) restarted the
expressive-performance competition after twelve years and ran nine systems.
DirectorMusices -- rule-based, from 2002 -- scored 4.33/5 and beat every neural
entrant in the preliminaries; five 2024-25 transformer and flow-matching systems
placed below all three legacy systems. So: rules, unapologetically.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import RenderError

TICKS_PER_BEAT = 480


@dataclass
class Note:
    start: float          # seconds from the start of the melody
    length: float         # seconds
    pitch: int
    velocity: int = 96
    lyric: str = ""
    phonemes: list[str] = field(default_factory=list)

    @property
    def end(self) -> float:
        return self.start + self.length


def from_wire(items) -> list[Note]:
    return [Note(start=float(n.start), length=float(n.length), pitch=int(n.pitch),
                 velocity=int(n.velocity), lyric=getattr(n, "lyric", "") or "",
                 phonemes=list(getattr(n, "phonemes", None) or []))
            for n in items]


def duration(notes: list[Note]) -> float:
    return max((n.end for n in notes), default=0.0)


# ---------------------------------------------------------------------------
# Making a melody out of whatever arrived
# ---------------------------------------------------------------------------

def overlaps(notes: list[Note]) -> int:
    """How many notes begin while an earlier one is still sounding."""
    ordered = sorted(notes, key=lambda n: n.start)
    count = 0
    for i in range(len(ordered) - 1):
        if ordered[i + 1].start < ordered[i].end - 1e-6:
            count += 1
    return count


def flatten_to_melody(notes: list[Note]) -> tuple[list[Note], int]:
    """Reduce to one voice by keeping the top line.

    v1 renders a single melody. Handing a chord to a monophonic instrument and
    quietly playing all of it is the kind of silent wrong that only shows up in
    the audio, so overlaps are resolved here, deliberately and countably: the
    highest sounding pitch wins, and the note underneath is truncated to where
    the higher one starts.
    """
    if not notes:
        return [], 0
    ordered = sorted(notes, key=lambda n: (n.start, -n.pitch))
    out: list[Note] = []
    dropped = 0
    for n in ordered:
        if not out:
            out.append(replace(n))
            continue
        prev = out[-1]
        if n.start < prev.end - 1e-6:
            if n.pitch > prev.pitch:
                prev.length = max(1e-3, n.start - prev.start)
                out.append(replace(n))
            else:
                dropped += 1
            continue
        out.append(replace(n))
    return out, dropped


# ---------------------------------------------------------------------------
# Deterministic transforms -- what a quick action or a prompt is allowed to do
# ---------------------------------------------------------------------------

def transpose(notes: list[Note], semitones: int) -> list[Note]:
    """Shift pitch, leaving anything that would leave the range where it is.

    Folding one note back an octave to keep it legal would put it in a
    different place in the melody than the player intended, which is worse than
    a melody that stays put.
    """
    if all(0 <= n.pitch + semitones <= 127 for n in notes):
        return [replace(n, pitch=n.pitch + semitones) for n in notes]
    return [replace(n) for n in notes]


def scale_time(notes: list[Note], factor: float) -> list[Note]:
    return [replace(n, start=n.start * factor, length=n.length * factor)
            for n in notes]


def swing(notes: list[Note], bpm: float, amount: float = 0.30) -> list[Note]:
    """Delay the off-eighths.

    `amount` is the fraction of an eighth to push them by; 1/3 is a strict
    triplet feel, and 0.30 sits just under it, which is where players tend to.
    """
    eighth = 30.0 / max(bpm, 1e-6)
    out = []
    for n in notes:
        position = n.start / eighth
        index = math.floor(position + 1e-6)
        on_the_eighth = abs(position - index) < 1e-3
        if on_the_eighth and index % 2 == 1:
            push = eighth * amount
            out.append(replace(n, start=n.start + push,
                               length=max(1e-3, n.length - push)))
        else:
            out.append(replace(n))
    return out


def quantise(notes: list[Note], bpm: float, division: int = 4) -> list[Note]:
    """Snap to the grid. `division` is per beat -- 4 means sixteenths."""
    step = 60.0 / max(bpm, 1e-6) / division
    out = []
    for n in notes:
        out.append(replace(n,
                           start=round(n.start / step) * step,
                           length=max(step, round(n.length / step) * step)))
    return out


# ---------------------------------------------------------------------------
# Humanisation
# ---------------------------------------------------------------------------

@dataclass
class Humanise:
    """How a part is performed, as opposed to what it plays."""
    velocity_centre: int = 84
    velocity_range: int = 22        # peak-to-peak of the phrase arc
    accent: int = 10                # extra velocity on a bar line
    timing_jitter_ms: float = 8.0
    legato: float = 0.0             # >0 lengthens notes, <0 clips them
    phrase_gap_s: float = 0.35      # a rest this long starts a new phrase
    end_lengthen: float = 0.12      # the last note of a phrase, relatively

    def with_actions(self, actions) -> "Humanise":
        h = replace(self)
        for a in actions:
            if a == "softer":
                h.velocity_centre = max(24, h.velocity_centre - 18)
            elif a == "louder":
                h.velocity_centre = min(120, h.velocity_centre + 18)
            elif a == "more_legato":
                h.legato = min(0.9, h.legato + 0.35)
            elif a == "more_staccato":
                h.legato = max(-0.8, h.legato - 0.45)
            elif a == "tighter":
                h.timing_jitter_ms = max(0.0, h.timing_jitter_ms * 0.35)
                h.velocity_range = max(4, int(h.velocity_range * 0.6))
            elif a == "looser":
                h.timing_jitter_ms = min(40.0, h.timing_jitter_ms * 1.9 + 4)
                h.velocity_range = min(60, int(h.velocity_range * 1.5) + 4)
        return h


def phrases(notes: list[Note], gap: float) -> list[list[int]]:
    """Split into phrases at rests. Indices, so callers can write back."""
    if not notes:
        return []
    groups: list[list[int]] = [[0]]
    for i in range(1, len(notes)):
        if notes[i].start - notes[i - 1].end >= gap:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def _arc(t: float) -> float:
    """A phrase shape: rise to about two thirds through, then ease off.

    Returns 0..1. Real phrases peak late, not in the middle, which is why this
    is not a plain sine over the whole span.
    """
    peak = 0.66
    if t <= peak:
        return math.sin(math.pi * 0.5 * (t / peak))
    return math.sin(math.pi * 0.5 * (1.0 - (t - peak) / (1.0 - peak)))


def humanise(notes: list[Note], bpm: float, h: Humanise,
             seed: int = 0) -> list[Note]:
    """Shape velocity, timing and length by rule.

    Deterministic for a given seed, because a render you cannot reproduce is a
    render you cannot compare against the one before it.
    """
    if not notes:
        return []
    rng = random.Random(seed)
    ordered = sorted(notes, key=lambda n: n.start)
    out = [replace(n) for n in ordered]
    beat = 60.0 / max(bpm, 1e-6)

    for group in phrases(ordered, h.phrase_gap_s):
        count = len(group)
        for position, index in enumerate(group):
            note = out[index]
            t = position / (count - 1) if count > 1 else 0.5
            velocity = h.velocity_centre + (_arc(t) - 0.5) * h.velocity_range

            # Metrical accent, strongest on the bar line.
            beats_in = note.start / beat
            on_beat = abs(beats_in - round(beats_in)) < 0.05
            if on_beat:
                whole = int(round(beats_in))
                velocity += h.accent if whole % 4 == 0 else h.accent * 0.45

            # A little life, bounded so it never inverts the arc.
            velocity += rng.uniform(-3.0, 3.0)
            note.velocity = int(max(1, min(127, round(velocity))))

            # Micro-timing. The first note of a phrase stays put: a late
            # entrance is the one timing error people hear as a mistake.
            if position > 0 and h.timing_jitter_ms > 0:
                shifted = note.start + rng.gauss(0.0, h.timing_jitter_ms / 1000.0)
                note.start = max(0.0, shifted)

            note.length = max(0.02, note.length * (1.0 + h.legato * 0.5))
            if position == count - 1 and h.end_lengthen:
                note.length *= (1.0 + h.end_lengthen)

    out.sort(key=lambda n: n.start)
    return out


# ---------------------------------------------------------------------------
# Writing a Standard MIDI File
#
# Encoded here rather than through a library, for the reason musicmaker gives in
# server/app/midi.py: emitting note-on and note-off is two message types, and a
# dependency for two message types is one you maintain for nothing. Reading
# arbitrary files people hand us is the opposite case, and read_melody below
# defers to mido without apology.
# ---------------------------------------------------------------------------

def _vlq(value: int) -> bytes:
    if value < 0:
        value = 0
    out = bytearray([value & 0x7F])
    value >>= 7
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def _chunk(tag: bytes, body: bytes) -> bytes:
    return tag + len(body).to_bytes(4, "big") + body


def write_smf(notes: list[Note], dest, bpm: float = 120.0, program: int = 0,
              channel: int = 0) -> Path:
    """Write a format-0 file.

    Overlapping and zero-length notes are both tolerated. A note shorter than a
    tick gets one, because a note-off at the same instant as its note-on is
    silence that still looks like a note in every editor that reads it back.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    bpm = float(bpm) if bpm and bpm > 0 else 120.0
    ticks_per_second = TICKS_PER_BEAT * bpm / 60.0

    # (tick, is_note_on, pitch, velocity). Note-offs sort first at equal ticks,
    # so a repeated pitch retriggers instead of being silenced by its own
    # predecessor's release.
    events: list[tuple[int, int, int, int]] = []
    for n in notes:
        start = max(0, round(n.start * ticks_per_second))
        length = max(1, round(n.length * ticks_per_second))
        pitch = max(0, min(127, int(n.pitch)))
        velocity = max(1, min(127, int(n.velocity or 96)))
        events.append((start, 1, pitch, velocity))
        events.append((start + length, 0, pitch, 0))
    events.sort(key=lambda e: (e[0], e[1]))

    body = bytearray()
    body += _vlq(0) + b"\xff\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big")
    body += _vlq(0) + bytes([0xC0 | (channel & 0x0F), max(0, min(127, program))])

    previous = 0
    for tick, kind, pitch, velocity in events:
        body += _vlq(tick - previous)
        previous = tick
        body += bytes([(0x90 if kind else 0x80) | (channel & 0x0F), pitch, velocity])

    body += _vlq(0) + b"\xff\x2f\x00"

    header = (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + \
        TICKS_PER_BEAT.to_bytes(2, "big")
    dest.write_bytes(_chunk(b"MThd", header) + _chunk(b"MTrk", bytes(body)))
    return dest


# ---------------------------------------------------------------------------
# Reading one
# ---------------------------------------------------------------------------

def read_melody(midi_path, track: int | None = None) -> tuple[list[Note], float, str]:
    """Pull a single melody line out of a MIDI file.

    Returns the notes, the tempo, and a sentence saying what was done to get
    there. v1 renders one line, so a multi-track file *will* lose material, and
    the only acceptable version of that is one the user is told about.
    """
    try:
        import mido
    except ImportError as exc:                                # pragma: no cover
        raise RenderError("mido is required to read MIDI files") from exc

    try:
        mid = mido.MidiFile(str(midi_path))
    except (OSError, ValueError, EOFError, IndexError) as exc:
        raise RenderError(f"could not read that MIDI file: {exc}") from exc

    bpm = 120.0
    for t in mid.tracks:
        for msg in t:
            if msg.type == "set_tempo":
                bpm = float(mido.tempo2bpm(msg.tempo))
                break
        else:
            continue
        break

    per_track: dict[int, list[Note]] = {}
    for index, t in enumerate(mid.tracks):
        open_notes: dict[int, tuple[float, int]] = {}
        found: list[Note] = []
        now = 0.0
        tempo = mido.bpm2tempo(bpm)
        for msg in t:
            now += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[msg.note] = (now, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                started = open_notes.pop(msg.note, None)
                if started is not None:
                    start, velocity = started
                    found.append(Note(start=start, length=max(1e-3, now - start),
                                      pitch=msg.note, velocity=velocity))
        if found:
            per_track[index] = sorted(found, key=lambda n: n.start)

    if not per_track:
        raise RenderError("that MIDI file has no notes in it")

    if track is not None and track in per_track:
        chosen = track
    else:
        # The busiest track is the melody often enough to be a good default, and
        # the caller is told which one it was so they can ask for another.
        chosen = max(per_track, key=lambda i: len(per_track[i]))

    notes = per_track[chosen]
    before = len(notes)
    notes, dropped = flatten_to_melody(notes)

    said = []
    if len(per_track) > 1:
        said.append(f"took track {chosen} of {len(mid.tracks)}, the busiest, "
                    f"with {before} notes")
    if dropped:
        said.append(f"kept the top line, dropping {dropped} note(s) underneath")
    said.append(f"tempo {bpm:.0f} from the file")
    return notes, bpm, "; ".join(said)
