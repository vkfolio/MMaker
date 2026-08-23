"""MIDI input path.

ACE-Step has no native MIDI conditioning -- it is a roadmap item for 2.0. The
bridge is to render the MIDI to audio and feed that as src_audio, which the
model does support via cover. FluidSynth does the rendering; it ships in the
Docker image.

Tempo is read from the MIDI itself when present, so the project grid can be
seeded from the file rather than guessed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base_errors import RenderError

DEFAULT_SOUNDFONTS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
]


def have_fluidsynth() -> bool:
    return shutil.which("fluidsynth") is not None


def find_soundfont(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).exists():
        return explicit
    for candidate in DEFAULT_SOUNDFONTS:
        if Path(candidate).exists():
            return candidate
    return None


def render(midi_path, dest, soundfont: str | None = None, sample_rate: int = 48000) -> Path:
    """Render a .mid to a .wav with FluidSynth."""
    midi_path, dest = Path(midi_path), Path(dest)
    if not have_fluidsynth():
        raise RenderError("fluidsynth is not installed in this image")
    sf2 = find_soundfont(soundfont)
    if not sf2:
        raise RenderError(
            "no SoundFont found -- install fluid-soundfont-gm or set MUSICMAKER_SOUNDFONT")

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["fluidsynth", "-ni", "-F", str(dest), "-r", str(sample_rate),
           "-g", "0.8", sf2, str(midi_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not dest.exists():
        raise RenderError(f"fluidsynth failed: {proc.stderr.strip()[:400]}")
    return dest


# ---------------------------------------------------------------------------
# Writing MIDI
#
# A Standard MIDI File is a small enough format to encode directly, and doing so
# keeps the dependency list honest: mido is not installed on the pod, and pulling
# it in to emit note-on and note-off would be a package for two message types.
# Reading is a different matter -- read_tempo below still defers to mido, because
# parsing arbitrary files people hand us is where a real library earns its place.
# ---------------------------------------------------------------------------

TICKS_PER_BEAT = 480


def _vlq(value: int) -> bytes:
    """MIDI variable-length quantity: seven bits per byte, high bit as 'more'."""
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


def write_smf(notes, dest, bpm: float = 120.0, program: int = 0,
              channel: int = 0) -> Path:
    """Write notes to a format-0 Standard MIDI File.

    `notes` are objects with `start`, `length` (both **seconds**, relative to the
    clip), `pitch` and `velocity`. Seconds rather than frames because a sample
    rate is a property of the audio device, not of the music -- the desktop
    converts at its edge, as it does for every other time it sends.

    Overlapping and zero-length notes are both tolerated: a note shorter than one
    tick is given one, since a note-off at the same instant as its note-on is
    silence that looks like a note in every editor that reads the file back.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    bpm = float(bpm) if bpm and bpm > 0 else 120.0
    ticks_per_second = TICKS_PER_BEAT * bpm / 60.0

    # (tick, is_note_off, pitch, velocity). Note-offs sort before note-ons at the
    # same tick so a repeated pitch retriggers rather than being silenced by the
    # previous note's release.
    events: list[tuple[int, int, int, int]] = []
    for n in notes:
        start = max(0, round(float(n.start) * ticks_per_second))
        length = max(1, round(float(n.length) * ticks_per_second))
        pitch = max(0, min(127, int(n.pitch)))
        velocity = max(1, min(127, int(getattr(n, "velocity", 96) or 96)))
        events.append((start, 1, pitch, velocity))
        events.append((start + length, 0, pitch, 0))
    events.sort(key=lambda e: (e[0], e[1]))

    body = bytearray()
    # Tempo, so anything reading this file back agrees with the project grid.
    body += _vlq(0) + b"\xff\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big")
    body += _vlq(0) + bytes([0xC0 | (channel & 0x0F), max(0, min(127, program))])

    previous = 0
    for tick, kind, pitch, velocity in events:
        body += _vlq(tick - previous)
        previous = tick
        status = (0x90 if kind else 0x80) | (channel & 0x0F)
        body += bytes([status, pitch, velocity])

    body += _vlq(0) + b"\xff\x2f\x00"          # end of track

    header = (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + \
        TICKS_PER_BEAT.to_bytes(2, "big")
    dest.write_bytes(_chunk(b"MThd", header) + _chunk(b"MTrk", bytes(body)))
    return dest


def read_tempo(midi_path) -> int | None:
    """Tempo from the MIDI file's own metadata, if it carries one."""
    try:
        import mido
    except ImportError:
        return None
    try:
        mid = mido.MidiFile(str(midi_path))
    except (OSError, ValueError):
        return None
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return int(round(mido.tempo2bpm(msg.tempo)))
    return None
