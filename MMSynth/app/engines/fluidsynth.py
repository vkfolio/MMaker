"""The deterministic half: notes to audio through a real sampler.

FluidSynth and a General MIDI soundfont are the floor. They are also, on their
own, unmistakably a MIDI file -- which is why this is a stage rather than the
product. What it buys is that the melody, the timing and the dynamics are
exactly right before any model touches them, so when the refined version is
wrong you know which half did it.

The quality path is an SFZ library through liquidsfz (MPL-2.0). sfizz would be
the obvious choice and is what most people will suggest, but it was archived
read-only on 21 June 2026; BSD-2 makes forking free, and an unmaintained C++
audio engine is not free. That path is not wired yet, and `sfz:` entries in
instruments.yaml are read and ignored until it is.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .. import audio, storage
from ..config import settings
from ..errors import NotReady, RenderError
from ..score import write_smf

# Where the Docker image and most distributions put a General MIDI bank.
SOUNDFONTS = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
]


def have_fluidsynth() -> bool:
    return shutil.which("fluidsynth") is not None


def find_soundfont() -> str | None:
    if settings.soundfont and Path(settings.soundfont).exists():
        return settings.soundfont
    for candidate in SOUNDFONTS:
        if Path(candidate).exists():
            return candidate
    local = settings.instruments_path.parent / "sf2"
    if local.is_dir():
        for found in sorted(local.glob("*.sf2")):
            return str(found)
    return None


class FluidSynthSampler:
    name = "fluidsynth"

    def render(self, dest, notes, bpm: float, preset=None, report=None) -> Path:
        if not have_fluidsynth():
            raise NotReady(
                "fluidsynth is not installed here. The pod image installs it; "
                "locally, run with MMSYNTH_ENGINE=stub.")
        sf2 = find_soundfont()
        if not sf2:
            raise NotReady(
                "no SoundFont found. Install fluid-soundfont-gm, or set "
                "MMSYNTH_SOUNDFONT to a .sf2 file.")

        program = getattr(getattr(preset, "instrument", None), "program", 0) or 0
        if report:
            report(0.15, "writing the score")
        midi_path = storage.work_dir() / (Path(dest).stem + ".mid")
        write_smf(notes, midi_path, bpm=bpm, program=program)

        if report:
            report(0.35, "playing it")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = ["fluidsynth", "-ni", "-F", str(dest),
                   "-r", str(settings.sample_rate), "-g", "0.8",
                   sf2, str(midi_path)]
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  timeout=600)
        except subprocess.TimeoutExpired as exc:
            raise RenderError("fluidsynth did not finish within 10 minutes") from exc
        if proc.returncode != 0 or not dest.exists():
            raise RenderError(f"fluidsynth failed: {proc.stderr.strip()[:400]}")

        # FluidSynth writes at whatever level the bank happens to sit at, which
        # varies by several dB between programs. Renders that differ in loudness
        # by instrument make every later comparison a loudness comparison.
        if report:
            report(0.8, "levelling")
        data, sr = audio.read(dest)
        audio.write(dest, audio.fade(audio.normalise(data, -6.0)), sr)
        return dest
