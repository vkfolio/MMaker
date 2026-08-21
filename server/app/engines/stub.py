"""Deterministic synthetic engine.

Produces real, listenable audio -- tones and noise shaped per track class -- so
mixing, analysis, versioning, export and the UI can all be exercised end to end
without a GPU. The same seed always gives the same audio, so tests are stable.

It is NOT a music model. Output is a diagnostic tone, not a song.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import audio
from ..schemas import Grid

SR = 48000

# Timbre and root pitch per track class.
_VOICES = {
    "drums":          {"kind": "perc",  "freq": 0},
    "percussion":     {"kind": "perc",  "freq": 0},
    "bass":           {"kind": "tone",  "freq": 55},
    "guitar":         {"kind": "tone",  "freq": 220},
    "keyboard":       {"kind": "tone",  "freq": 330},
    "synth":          {"kind": "tone",  "freq": 440},
    "strings":        {"kind": "pad",   "freq": 262},
    "brass":          {"kind": "pad",   "freq": 196},
    "woodwinds":      {"kind": "pad",   "freq": 349},
    "vocals":         {"kind": "tone",  "freq": 392},
    "backing_vocals": {"kind": "pad",   "freq": 494},
    "fx":             {"kind": "noise", "freq": 0},
    "mix":            {"kind": "tone",  "freq": 110},
}


def _env(n, attack=0.005, release=0.15):
    a = int(SR * attack)
    r = int(SR * release)
    e = np.ones(n, dtype=np.float32)
    if a and a < n:
        e[:a] = np.linspace(0, 1, a)
    if r and r < n:
        e[-r:] = np.linspace(1, 0, r)
    return e


def _render(track_class: str, grid: Grid, seed: int, duration_s: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spec = _VOICES.get(track_class, _VOICES["mix"])
    n = max(1, int(duration_s * SR))
    out = np.zeros(n, dtype=np.float32)

    step = 60.0 / grid.bpm / 2.0          # eighths, so onsets land on the grid
    hits = max(1, int(duration_s / step))

    if spec["kind"] == "perc":
        for i in range(hits):
            start = int(i * step * SR)
            length = min(int(0.09 * SR), n - start)
            if length <= 0:
                break
            # Kick on the beat, hat off it, so beat tracking has real onsets.
            if i % 2:
                body = rng.standard_normal(length) * 0.35
            else:
                body = np.sin(2 * np.pi * 60 * np.arange(length) / SR) * 0.9
            out[start:start + length] += body.astype(np.float32) * _env(length, 0.001, 0.06)

    elif spec["kind"] == "pad":
        t = np.arange(n) / SR
        for mult in (1.0, 1.5, 2.0):
            out += np.sin(2 * np.pi * spec["freq"] * mult * t).astype(np.float32) * 0.18
        out *= _env(n, 0.4, 0.6)

    elif spec["kind"] == "noise":
        out = (rng.standard_normal(n) * 0.25).astype(np.float32) * _env(n, 0.3, 0.3)

    else:
        scale = [0, 3, 5, 7, 10]
        for i in range(hits):
            start = int(i * step * SR)
            length = min(int(step * SR * 0.9), n - start)
            if length <= 0:
                break
            semis = scale[int(rng.integers(0, len(scale)))]
            freq = spec["freq"] * (2 ** (semis / 12))
            t = np.arange(length) / SR
            note = np.sin(2 * np.pi * freq * t) * 0.4
            out[start:start + length] += note.astype(np.float32) * _env(length, 0.004, 0.05)

    peak = float(np.abs(out).max())
    if peak > 0:
        out = out / peak * 0.7
    return np.stack([out, out], axis=1)


class StubEngine:
    name = "stub"

    def generate(self, dest, prompt, grid, seed, lyrics="", vocal_language="en",
                 src_audio=None, report=None, quality=None):
        return audio.write(dest, _render("mix", grid, seed, grid.duration_s), SR)

    def layer(self, dest, track_class, src_audio, prompt, grid, seed,
              lyrics="", vocal_language="en", report=None, quality=None):
        # Returns the new part alone -- the behaviour we hope ACE-Step has, and
        # which research/lego_chain.py exists to confirm on a real pod.
        base, sr = audio.load(src_audio)
        duration = len(base) / sr
        return audio.write(dest, _render(track_class, grid, seed, duration), SR)

    def repaint(self, dest, src_audio, start_s, end_s, prompt, grid, seed,
                report=None, quality=None):
        data, sr = audio.load(src_audio)
        fresh = _render("mix", grid, seed, max(end_s - start_s, 0.01))
        i0 = int(start_s * sr)
        span = min(int(end_s * sr), len(data)) - i0
        out = data.copy()
        if span > 0:
            patch = fresh[:span]
            if patch.shape[1] != out.shape[1]:
                patch = np.repeat(patch[:, :1], out.shape[1], axis=1)
            if len(patch) < span:
                pad = np.zeros((span - len(patch), out.shape[1]), np.float32)
                patch = np.vstack([patch, pad])
            # Short crossfades so the seam is not a click.
            fade = min(int(0.01 * sr), span // 2)
            if fade > 0:
                ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
                patch[:fade] = patch[:fade] * ramp + out[i0:i0 + fade] * (1 - ramp)
                patch[-fade:] = (patch[-fade:] * (1 - ramp)
                                 + out[i0 + span - fade:i0 + span] * ramp)
            out[i0:i0 + span] = patch
        return audio.write(dest, out, sr)

    def extend(self, dest, src_audio, added_s, prompt, grid, seed,
               report=None, quality=None):
        data, sr = audio.load(src_audio)
        tail = _render("mix", grid, seed, added_s)
        if tail.shape[1] != data.shape[1]:
            tail = np.repeat(tail[:, :1], data.shape[1], axis=1)
        return audio.write(dest, np.vstack([data, tail]), sr)

    def separate(self, src_audio, out_dir):
        """Fake demix: deterministic per-stem renders of the same length."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data, sr = audio.load(src_audio)
        duration = len(data) / sr
        grid = Grid(bpm=96)
        result = {}
        for i, name in enumerate(("drums", "bass", "keyboard", "vocals")):
            path = out_dir / f"{name}.wav"
            audio.write(path, _render(name, grid, 1000 + i, duration), SR)
            result[name] = path
        return result

    def convert_voice(self, dest, src_audio, reference):
        """Resample-shift so the timbre change is audible and deterministic."""
        data, sr = audio.load(src_audio)
        ref_seed = abs(hash(str(reference))) % 9973
        shift = 0.85 + (ref_seed % 30) / 100.0
        idx = np.clip((np.arange(len(data)) * shift).astype(int), 0, len(data) - 1)
        out = data[idx]
        if len(out) < len(data):
            pad = np.zeros((len(data) - len(out), data.shape[1]), np.float32)
            out = np.vstack([out, pad])
        return audio.write(dest, out[:len(data)], sr)

    def sfx(self, dest, prompt, duration_s, seed):
        rng = np.random.default_rng(seed)
        n = max(1, int(duration_s * SR))
        t = np.arange(n) / SR
        sweep = np.sin(2 * np.pi * (80 + 900 * t / max(duration_s, 1e-6)) * t)
        body = (sweep * 0.5 + rng.standard_normal(n) * 0.2).astype(np.float32)
        body *= _env(n, 0.02, 0.4)
        peak = float(np.abs(body).max()) or 1.0
        return audio.write(dest, np.stack([body, body], axis=1) / peak * 0.8, SR)
