"""Audio I/O and mixing.

Uses soundfile when it is installed (the pod), and falls back to the stdlib
`wave` module for 16-bit PCM so the whole service -- including tests -- runs on
a laptop with nothing but numpy.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:                                     # pragma: no cover
    sf = None
    HAVE_SOUNDFILE = False


def load(path) -> tuple[np.ndarray, int]:
    """Return float32 [samples, channels] in -1..1, plus sample rate."""
    path = str(path)
    if HAVE_SOUNDFILE:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data, sr
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise RuntimeError(
            f"{path} is {width * 8}-bit; install soundfile to read anything but 16-bit PCM")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, channels), sr


def write(path, data: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if HAVE_SOUNDFILE:
        sf.write(str(path), data, sr)
        return path
    clipped = np.clip(data, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def describe(path) -> dict:
    data, sr = load(path)
    return {
        "path": str(path),
        "sample_rate": sr,
        "channels": int(data.shape[1]),
        "duration_s": round(len(data) / sr, 3),
        "peak": round(float(np.abs(data).max()) if len(data) else 0.0, 4),
    }


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim == 2 else data


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def apply_pan(data: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan. -1 hard left, +1 hard right."""
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * np.pi / 4.0
    return data * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * np.sqrt(2)


def mix(sources: list[tuple], dest, headroom_db: float = -1.0, sr_hint: int | None = None):
    """Sum (path, gain_db, pan) triples into one file.

    Peak-normalised to leave headroom, because a clipped mix passed back as
    conditioning audio is a corrupted signal, not merely a loud one.
    """
    layers, sr = [], sr_hint
    for path, gain_db, pan in sources:
        data, this_sr = load(path)
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            raise ValueError(f"sample-rate mismatch: {path} is {this_sr}, expected {sr}")
        data = apply_pan(data, pan) * db_to_gain(gain_db)
        layers.append(data)

    if not layers:
        raise ValueError("nothing to mix")

    width = max(l.shape[1] for l in layers)
    length = max(l.shape[0] for l in layers)
    acc = np.zeros((length, width), dtype=np.float64)
    for l in layers:
        if l.shape[1] == 1 and width > 1:
            l = np.repeat(l, width, axis=1)
        acc[: l.shape[0], : l.shape[1]] += l

    peak = float(np.abs(acc).max())
    if peak > 0:
        acc *= db_to_gain(headroom_db) / peak
    return write(dest, acc.astype(np.float32), sr), sr


def null_test(path_a, path_b) -> float:
    """Residual RMS of (A - B) relative to A's RMS. ~0 means B contains A."""
    a, sr_a = load(path_a)
    b, sr_b = load(path_b)
    if sr_a != sr_b:
        raise ValueError(f"sample-rate mismatch: {sr_a} vs {sr_b}")
    n = min(len(a), len(b))
    w = min(a.shape[1], b.shape[1])
    a, b = a[:n, :w], b[:n, :w]
    rms_a = float(np.sqrt(np.mean(a ** 2))) or 1e-12
    return float(np.sqrt(np.mean((a - b) ** 2))) / rms_a


def shift_ms(path, dest, ms: float):
    """Nudge audio in time to correct small drift. Positive delays."""
    data, sr = load(path)
    n = int(round(abs(ms) / 1000.0 * sr))
    if n == 0:
        return write(dest, data, sr)
    pad = np.zeros((n, data.shape[1]), dtype=np.float32)
    out = np.vstack([pad, data])[:len(data)] if ms > 0 else np.vstack([data[n:], pad])
    return write(dest, out, sr)


def slice_seconds(path, dest, start_s: float, end_s: float):
    data, sr = load(path)
    return write(dest, data[int(start_s * sr):int(end_s * sr)], sr)


def concat(paths, dest):
    parts, sr = [], None
    for p in paths:
        data, this_sr = load(p)
        sr = sr or this_sr
        if this_sr != sr:
            raise ValueError("sample-rate mismatch in concat")
        parts.append(data)
    width = max(p.shape[1] for p in parts)
    parts = [np.repeat(p, width, axis=1) if p.shape[1] == 1 and width > 1 else p
             for p in parts]
    return write(dest, np.vstack(parts), sr)


def silence(dest, duration_s: float, sr: int = 48000, channels: int = 2):
    return write(dest, np.zeros((int(duration_s * sr), channels), dtype=np.float32), sr)
