"""Deterministic synthetic engines.

These are not models. They make real, listenable audio -- the actual notes, at
the actual times, with the actual dynamics -- so that timing, alignment, levels,
jobs, storage and the API can all be exercised end to end on a laptop with no
GPU and no weights.

That matters more than it sounds. A stub that returns silence proves the routes
work; a stub that plays the melody proves the *melody* survives the routes, and
that is where the bugs are. Same seed, same audio, always.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .. import audio
from ..score import Note

SR = audio.SR


def _freq(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def _seed_of(*parts) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def _envelope(n: int, attack: float, release: float) -> np.ndarray:
    a = min(int(SR * attack), max(1, n // 3))
    r = min(int(SR * release), max(1, n // 2))
    env = np.ones(n, dtype=np.float32)
    if a > 1:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r > 1:
        env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _canvas(notes: list[Note], tail: float = 1.0) -> np.ndarray:
    end = max((n.start + n.length for n in notes), default=0.0) + tail
    return np.zeros(int(SR * max(0.5, end)), dtype=np.float32)


class StubSampler:
    """An additive synth. Harmonics thin out as pitch rises, like a real one."""

    name = "stub-sampler"

    def render(self, dest, notes, bpm: float, preset=None, report=None) -> Path:
        if report:
            report(0.2, "playing the notes")
        out = _canvas(notes)
        program = getattr(getattr(preset, "instrument", None), "program", 0) or 0
        # The program picks a harmonic recipe, so different instruments really
        # do sound different in the stub -- otherwise an instrument-selection
        # bug is invisible until there is a GPU.
        weights = [1.0, 0.5, 0.25, 0.12, 0.06]
        rotate = program % len(weights)
        weights = weights[rotate:] + weights[:rotate]

        for note in notes:
            start = int(note.start * SR)
            length = max(64, int(note.length * SR))
            if start >= out.size:
                continue
            length = min(length, out.size - start)
            t = np.arange(length, dtype=np.float32) / SR
            f = _freq(note.pitch)
            wave = np.zeros(length, dtype=np.float32)
            for h, w in enumerate(weights, start=1):
                if f * h > SR / 2.2:
                    break
                wave += w * np.sin(2.0 * np.pi * f * h * t, dtype=np.float32)
            wave *= _envelope(length, 0.008, min(0.25, note.length * 0.4))
            wave *= (note.velocity / 127.0) ** 1.4
            out[start:start + length] += wave

        if report:
            report(0.8, "levelling")
        stereo = np.stack([out, out], axis=1)
        stereo = audio.fade(audio.normalise(stereo, -6.0))
        return audio.write(dest, stereo)


class StubRefiner:
    """A tilt filter standing in for a model.

    It changes the sound in a way you can hear and measure, at a strength that
    behaves like the real parameter, while leaving timing and pitch untouched --
    which is the property the real refiner has to be tested for.
    """

    name = "stub-refiner"

    def refine(self, dest, src_audio, prompt: str, strength: float,
               seed: int = 0, report=None) -> Path:
        if report:
            report(0.3, "reimagining it")
        data, sr = audio.read(src_audio)
        rng = np.random.default_rng(_seed_of(prompt, seed))
        # A one-pole shelf whose direction comes from the prompt. Deterministic
        # per prompt, so "brighter" is always the same "brighter".
        bright = (_seed_of(prompt) % 2 == 0)
        coefficient = 0.55 if bright else 0.90

        # The one-pole written as a truncated exponential kernel: identical to
        # the recursion within the truncation, and vectorised, because a
        # per-sample Python loop over 48 kHz stereo is seconds of test time for
        # a filter nobody ships.
        taps = int(np.ceil(np.log(1e-4) / np.log(coefficient)))
        kernel = (1.0 - coefficient) * coefficient ** np.arange(taps)
        kernel = kernel.astype(np.float32)

        filtered = np.empty_like(data)
        for c in range(data.shape[1]):
            x = data[:, c]
            y = np.convolve(x, kernel)[:x.size]
            filtered[:, c] = (x - y) if bright else y

        mix = max(0.0, min(1.0, strength))
        out = (1.0 - mix) * data + mix * filtered
        out += rng.normal(0.0, 1e-4, out.shape).astype(np.float32)
        if report:
            report(0.9, "levelling")
        return audio.write(dest, audio.normalise(out, -3.0), sr)


class StubSinger:
    """A vowel-ish tone with vibrato and one amplitude gesture per syllable.

    It is not singing. It does put a distinguishable event at every syllable
    onset, at the right pitch and the right time, which is what makes an
    alignment bug audible and testable before SoulX-Singer is anywhere near.
    """

    name = "stub-singer"

    def sing(self, dest, notes, bpm: float, reference=None, language: str = "en",
             seed: int = 0, report=None) -> Path:
        if report:
            report(0.3, "singing it")
        out = _canvas(notes)
        # The reference voice shifts the formants, so two voices are not the
        # same audio -- a voice-selection bug should not need a GPU to catch.
        who = getattr(reference, "wav", reference)
        colour = _seed_of(Path(who).name if who else "default")
        formant = 1.0 + ((colour % 100) / 100.0) * 0.4

        for note in notes:
            start = int(note.start * SR)
            length = max(64, int(note.length * SR))
            if start >= out.size:
                continue
            length = min(length, out.size - start)
            t = np.arange(length, dtype=np.float32) / SR
            f = _freq(note.pitch)

            vibrato = 1.0 + 0.006 * np.sin(2.0 * np.pi * 5.2 * t, dtype=np.float32)
            phase = 2.0 * np.pi * f * t * vibrato
            wave = np.sin(phase, dtype=np.float32)
            wave += 0.45 * np.sin(phase * 2.0 * formant, dtype=np.float32)
            wave += 0.20 * np.sin(phase * 3.0 * formant, dtype=np.float32)

            # A syllable onset gets an attack; a held note (melisma) does not,
            # so the two are distinguishable in the output.
            attack = 0.012 if (note.lyric or "").strip() else 0.09
            wave *= _envelope(length, attack, min(0.3, note.length * 0.45))
            wave *= (note.velocity / 127.0) ** 1.2
            out[start:start + length] += wave

        if report:
            report(0.85, "levelling")
        stereo = np.stack([out, out], axis=1)
        return audio.write(dest, audio.fade(audio.normalise(stereo, -6.0)))
