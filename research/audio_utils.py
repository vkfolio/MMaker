"""Audio helpers shared by the Phase 0 research scripts."""

import numpy as np
import soundfile as sf


def load(path, target_sr=None):
    """Load as float32 [samples, channels]. Resamples only if asked."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if target_sr and sr != target_sr:
        import librosa
        mono_stack = [librosa.resample(data[:, c], orig_sr=sr, target_sr=target_sr)
                      for c in range(data.shape[1])]
        data = np.stack(mono_stack, axis=1)
        sr = target_sr
    return data, sr


def to_mono(data):
    return data.mean(axis=1) if data.ndim == 2 else data


def write(path, data, sr):
    sf.write(str(path), data, sr)
    return path


def mixdown(paths, dest, headroom_db=-1.0):
    """Sum stems to a single file, peak-normalised to leave headroom.

    This mix is what gets passed as src_audio for the next lego call, so it
    must not clip -- a clipped context is a corrupted conditioning signal.
    """
    layers = []
    sr = None
    for p in paths:
        data, this_sr = load(p)
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            raise ValueError(f"sample-rate mismatch: {p} is {this_sr}, expected {sr}")
        layers.append(data)

    width = max(l.shape[1] for l in layers)
    length = max(l.shape[0] for l in layers)
    acc = np.zeros((length, width), dtype=np.float64)
    for l in layers:
        if l.shape[1] == 1 and width > 1:
            l = np.repeat(l, width, axis=1)
        acc[: l.shape[0], : l.shape[1]] += l

    peak = np.abs(acc).max()
    if peak > 0:
        acc *= (10 ** (headroom_db / 20.0)) / peak
    return write(dest, acc.astype(np.float32), sr), sr


def null_test(path_a, path_b):
    """Invert-and-sum two files. Returns residual RMS relative to A's RMS.

    Near 0.0 means B contains A essentially untouched. Near or above 1.0
    means they are unrelated. Used to answer two Phase 0 questions:
    does `lego` return a solo stem or a full mix, and does `repaint`
    genuinely preserve the region outside the mask.
    """
    a, sr_a = load(path_a)
    b, sr_b = load(path_b)
    if sr_a != sr_b:
        raise ValueError(f"sample-rate mismatch: {sr_a} vs {sr_b}")
    n = min(len(a), len(b))
    w = min(a.shape[1], b.shape[1])
    a, b = a[:n, :w], b[:n, :w]
    rms_a = float(np.sqrt(np.mean(a ** 2))) or 1e-12
    residual = float(np.sqrt(np.mean((a - b) ** 2)))
    return residual / rms_a


def describe(path):
    info = sf.info(str(path))
    return {
        "path": str(path),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "duration_s": round(info.frames / info.samplerate, 3),
        "subtype": info.subtype,
    }
