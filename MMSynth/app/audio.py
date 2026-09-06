"""Audio in and out, and the small amount of processing that is ours.

soundfile is used when it is installed and the stdlib `wave` module when it is
not, so the whole service -- including the stub engines and their tests -- runs
on a laptop with nothing compiled. The formats agree: 48 kHz, stereo, and float
in memory, converted only at the file boundary.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48000


def _have_soundfile() -> bool:
    try:
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


def write(path, data: np.ndarray, sr: int = SR) -> Path:
    """Write float audio in [-1, 1]. Mono in, stereo out."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    data = np.clip(data, -1.0, 1.0)

    if _have_soundfile():
        import soundfile as sf
        sf.write(str(path), data, sr, subtype="PCM_24")
        return path

    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((data * 32767.0).astype("<i2").tobytes())
    return path


def read(path) -> tuple[np.ndarray, int]:
    """Read to float32, shape (frames, channels)."""
    path = Path(path)
    if _have_soundfile():
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return data, sr

    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(
            f"{path.name} is {width * 8}-bit; install soundfile to read it")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, channels), sr


def resample(data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Change sample rate.

    scipy's polyphase filter when it is available, linear interpolation when it
    is not. The fallback is honestly worse -- it aliases -- but it is only
    reached on a laptop with no scientific stack, where nothing is being
    listened to critically anyway.
    """
    if source_sr == target_sr or data.size == 0:
        return data
    mono = data if data.ndim == 2 else data[:, None]
    try:
        from math import gcd

        from scipy.signal import resample_poly
        common = gcd(int(source_sr), int(target_sr))
        out = resample_poly(mono, target_sr // common, source_sr // common, axis=0)
    except ImportError:
        n = int(round(mono.shape[0] * target_sr / source_sr))
        source_t = np.linspace(0.0, 1.0, mono.shape[0], endpoint=False)
        target_t = np.linspace(0.0, 1.0, n, endpoint=False)
        out = np.stack([np.interp(target_t, source_t, mono[:, c])
                        for c in range(mono.shape[1])], axis=1)
    out = np.asarray(out, dtype=np.float32)
    return out if data.ndim == 2 else out[:, 0]


def write_resampled(path, data: np.ndarray, source_sr: int, target_sr: int) -> Path:
    """Write audio produced at one rate as a file at the service's rate.

    Every wav MMSynth hands back is the same shape, so a client never has to ask
    which engine made a file before it can play it alongside another.
    """
    return write(path, resample(np.asarray(data, dtype=np.float32),
                                source_sr, target_sr), target_sr)


def peak_db(data: np.ndarray) -> float:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    return -120.0 if peak <= 1e-9 else 20.0 * np.log10(peak)


def rms_db(data: np.ndarray) -> float:
    if not data.size:
        return -120.0
    value = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    return -120.0 if value <= 1e-9 else 20.0 * np.log10(value)


def normalise(data: np.ndarray, peak_dbfs: float = -1.5) -> np.ndarray:
    """Bring the peak to a fixed headroom. Silence is left alone.

    Fixed headroom rather than none: a render that touches 0 dBFS clips the
    moment anything is layered on top of it, and layering is the whole point of
    handing these files to a DAW.
    """
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 1e-6:
        return data
    target = 10.0 ** (peak_dbfs / 20.0)
    return data * (target / peak)


def fade(data: np.ndarray, ms: float = 8.0, sr: int = SR) -> np.ndarray:
    """Short fades at both ends, so a render never starts or stops on a step."""
    n = int(sr * ms / 1000.0)
    if data.size == 0 or n <= 0 or data.shape[0] < 2 * n:
        return data
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = data.copy()
    shape = (n, 1) if out.ndim == 2 else (n,)
    out[:n] *= ramp.reshape(shape)
    out[-n:] *= ramp[::-1].reshape(shape)
    return out


def reverb(data: np.ndarray, mix: float = 0.15, decay_s: float = 1.4,
           sr: int = SR) -> np.ndarray:
    """A synthetic room.

    Real impulse responses are better and are the plan for the quality path --
    OpenAIR has usable ones. This exists so that "dry samples sound dry", the
    loudest tell that something is a MIDI file, is not the first thing anyone
    hears. Exponentially decaying noise is a crude room but it is a room.
    """
    if mix <= 0 or data.size == 0:
        return data
    n = int(sr * decay_s)
    if n < 16:
        return data
    rng = np.random.default_rng(1729)          # fixed: the room does not change
    tail = rng.standard_normal((n, data.shape[1] if data.ndim == 2 else 1))
    tail *= np.exp(-np.linspace(0.0, 6.0, n))[:, None]
    tail[:int(sr * 0.012)] = 0.0               # pre-delay
    tail = tail.astype(np.float32)
    tail /= max(1e-9, float(np.sqrt(np.sum(tail ** 2))))

    mono = data if data.ndim == 2 else data[:, None]
    wet = np.empty_like(mono)
    for c in range(mono.shape[1]):
        wet[:, c] = np.convolve(mono[:, c], tail[:, c % tail.shape[1]],
                                mode="full")[:mono.shape[0]]
    out = (1.0 - mix) * mono + mix * wet
    return out if data.ndim == 2 else out[:, 0]


def match_loudness(data: np.ndarray, target_rms_db: float = -16.0,
                   ceiling_db: float = -1.5) -> np.ndarray:
    """Put the level where a DAW expects it, then guarantee the headroom.

    Matchering does this properly, against a reference master, and is the plan
    for the quality path -- it is pure NumPy/SciPy, deterministic, no weights.
    This is the one-line version so that levels are sane from the first render.
    """
    if data.size == 0:
        return data
    current = rms_db(data)
    if current > -119.0:
        data = data * (10.0 ** ((target_rms_db - current) / 20.0))
    return normalise(data, ceiling_db)
