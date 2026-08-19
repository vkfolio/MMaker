"""Tempo/key detection and sync verification.

Degrades honestly: without librosa every result is reported as `unmeasured`
rather than guessed, because a fabricated key would silently corrupt every
stem that locks to it.
"""

from __future__ import annotations

import numpy as np

from . import audio
from .schemas import SyncReport

try:
    import librosa
    HAVE_LIBROSA = True
except ImportError:                                     # pragma: no cover
    librosa = None
    HAVE_LIBROSA = False

# Onsets further than this from a grid line are syncopation or sustain, not drift.
SNAP_WINDOW_S = 0.12

_PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect(path) -> dict:
    """Best-effort bpm/key. Always flagged for human confirmation."""
    if not HAVE_LIBROSA:
        return {"bpm": None, "key_scale": None, "confidence": 0.0,
                "reason": "librosa not installed"}
    data, sr = audio.load(path)
    y = audio.to_mono(data)
    if len(y) < sr:
        return {"bpm": None, "key_scale": None, "confidence": 0.0,
                "reason": "clip too short to analyse"}

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best, best_score = None, -np.inf
    for i in range(12):
        for name, profile in (("Major", _MAJOR), ("Minor", _MINOR)):
            score = float(np.corrcoef(np.roll(chroma, -i), profile)[0, 1])
            if score > best_score:
                best_score, best = score, f"{_PITCHES[i]} {name}"

    return {
        "bpm": int(round(tempo)),
        "key_scale": best,
        "confidence": round(max(0.0, best_score), 3),
        "reason": "",
    }


def sync_report(path, bpm: int, subdivision: int = 4) -> SyncReport:
    """Deviation of onsets from the ideal grid."""
    if not HAVE_LIBROSA:
        return SyncReport(verdict="unmeasured")
    data, sr = audio.load(path)
    y = audio.to_mono(data)
    if len(y) < sr // 2:
        return SyncReport(verdict="unmeasured")

    step = 60.0 / bpm / (subdivision / 4.0)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    devs = [(t - round(t / step) * step) * 1000.0 for t in onsets
            if abs(t - round(t / step) * step) <= SNAP_WINDOW_S]
    if not devs:
        return SyncReport(verdict="unmeasured")

    arr = np.asarray(devs)
    median = float(np.median(np.abs(arr)))
    verdict = "pass" if median < 20 else "marginal" if median < 50 else "fail"
    # A consistently signed offset is latency -- one nudge fixes it. A scattered
    # one is real drift and nudging would not help, so we leave it at 0.
    mean_signed = float(np.mean(arr))
    nudge = -mean_signed if abs(mean_signed) > 5 and abs(mean_signed) > median * 0.6 else 0.0

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm)
    return SyncReport(
        detected_bpm=round(float(np.atleast_1d(tempo)[0]), 2),
        median_dev_ms=round(median, 2),
        p90_dev_ms=round(float(np.percentile(np.abs(arr), 90)), 2),
        verdict=verdict,
        nudge_ms=round(nudge, 2),
    )
