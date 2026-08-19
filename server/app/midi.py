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
