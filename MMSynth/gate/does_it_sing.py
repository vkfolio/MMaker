"""Phase 0, gate 2a: does `cover` sing at all?

Gate 2 measured voice consistency and found none. The diagnostic explained why:
none of the twelve renders contained singing. Every one was statistically
indistinguishable from the instrumental melody carrier --

    zero-crossing rate   carrier 0.020   covers 0.019-0.022   real vocal 0.045
    spectral flatness    carrier 0.0001  covers 0.0001-0.0002 real vocal 0.0013

-- so the lyrics were ignored and gate 2 was comparing instrumentals.

That makes sense mechanically. `cover` conditions on FSQ semantic codes, and
those codes encode *what the source is*, not just its pitches. A monophonic
sampler line says "solo instrument", and at `audio_cover_strength=1.0` the model
honours that for the whole schedule. The same codes that keep the melody also
keep it wordless.

So this asks a narrower question: **is there any setting where the tune survives
AND a voice appears?** Two knobs plausibly buy vocal freedom:

  audio_cover_strength   below 1.0 the model stops attending to the source part
                         way through and finishes on the caption alone -- which
                         is where a vocal could enter
  cover_noise_strength   lower means more room to reinterpret

Pass condition needs both, and they pull against each other:
  * voiced:  zero-crossing and flatness move toward the real-vocal end
  * on-tune: pitch still tracks the MIDI
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_identity import (BASE, LYRICS, MELODY, SR, render_midi,  # noqa: E402
                            submit, write_smf)

# Measured on this pod: the carrier, and a real sung reference the model made.
CARRIER_ZCR, CARRIER_FLAT = 0.020, 0.0001
VOCAL_ZCR, VOCAL_FLAT = 0.045, 0.0013

CAPTION = ("solo female vocal singing, a cappella, clear diction, no instruments, "
           "dry close mic")


def voiced_score(path: Path) -> tuple[float, float, float]:
    """(zcr, flatness, 0..1 how vocal it looks) against the two measured poles."""
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    z = (zcr - CARRIER_ZCR) / (VOCAL_ZCR - CARRIER_ZCR)
    f = (flat - CARRIER_FLAT) / (VOCAL_FLAT - CARRIER_FLAT)
    return zcr, flat, float(np.clip((z + f) / 2.0, 0.0, 1.5))


def pitch_error(path: Path, notes) -> float:
    import librosa
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C7"), frame_length=2048)
    times = librosa.times_like(f0, sr=sr)
    errs, t = [], 0.0
    for pitch, length in notes:
        sel = (times >= t + length * 0.2) & (times <= t + length * 0.8) & voiced
        vals = f0[sel]
        vals = vals[~np.isnan(vals)]
        if vals.size:
            errs.append(abs(float(librosa.hz_to_midi(np.median(vals))) - pitch))
        t += length
    return float(np.mean(errs)) if errs else float("nan")


def run(src: Path, dest: Path, noise: float, strength: float, steps: int = 50,
        seed: int = 7) -> float:
    return submit({
        "task_type": "cover", "prompt": CAPTION, "lyrics": LYRICS,
        "vocal_language": "en",
        "cover_noise_strength": f"{noise:.3f}",
        "audio_cover_strength": f"{strength:.3f}",
        "inference_steps": str(steps), "guidance_scale": "7.0",
        "use_random_seed": "false", "seed": str(seed),
        "audio_format": "wav", "thinking": "false",
    }, {"src_audio": (src.name, src.read_bytes(), "audio/wav")}, dest)


def main():
    out = Path("/workspace/gate/sing")
    out.mkdir(parents=True, exist_ok=True)
    dry = render_midi(write_smf(MELODY, out / "m.mid", 120), out / "00_carrier.wav")

    zc, fl, v = voiced_score(dry)
    print(f"carrier          zcr {zc:.4f} flat {fl:.5f} voiced {v:5.2f}\n")

    grid = [(0.35, 1.0), (0.35, 0.5), (0.35, 0.2),
            (0.15, 0.5), (0.15, 0.2), (0.05, 0.2), (0.05, 0.0)]

    print(f"{'noise':>6} {'strength':>9} {'zcr':>7} {'flat':>8} {'voiced':>7} "
          f"{'pitch_err':>10}")
    print("-" * 52)
    rows = []
    for noise, strength in grid:
        dest = out / f"n{noise}_s{strength}.wav".replace(".", "")
        dest = out / f"n{int(noise*100):03d}_s{int(strength*100):03d}.wav"
        try:
            run(dry, dest, noise, strength)
        except Exception as exc:                                   # noqa: BLE001
            print(f"{noise:6.2f} {strength:9.2f}   FAILED {str(exc)[:30]}")
            continue
        zc, fl, v = voiced_score(dest)
        pe = pitch_error(dest, MELODY)
        rows.append({"noise": noise, "strength": strength, "zcr": zc,
                     "flat": fl, "voiced": v, "pitch_err": pe,
                     "file": dest.name})
        print(f"{noise:6.2f} {strength:9.2f} {zc:7.4f} {fl:8.5f} {v:7.2f} {pe:10.2f}")

    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nvoiced ~0 means it stayed an instrument; ~1 means it looks like the")
    print("model's own sung output. A usable row needs voiced well above 0 AND")
    print("pitch_err at or under about 0.5 semitones.")


if __name__ == "__main__":
    sys.exit(main())
