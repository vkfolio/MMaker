"""Phase 0, gate 3: can ACE-Step XL re-voice a vocal SoulX-Singer already sang?

This is the user's idea, and it inverts the thing that failed.

Gate 2a showed `cover` will not turn an instrument into a voice: the FSQ semantic
codes encode *what the source is*, and a monophonic sampler line says "solo
instrument". The same codes that hold the melody hold it wordless.

But that argument cuts the other way. Give `cover` a source that is **already a
sung vocal** and the codes encode "a voice, singing these words, at these
pitches". Then the thing preserved is the singing, and what the caption is free
to change is the timbre -- which is exactly "same performance, different singer".

If it holds, the two models compose into something neither does alone:

    MIDI + lyrics + singer  ->  SoulX-Singer  ->  a note-exact vocal
                                                       |
                                    ACE-Step XL cover --+-> other voices,
                                                            same performance

Three things have to be true at once, and gate 2a failed the first two:

  voiced     it is still singing, not reduced to an instrument
  on-tune    the melody SoulX sang is still there
  distinct   different captions give measurably different voices
             (this is the test gate 2 failed at a ratio near 1.0)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from does_it_sing import voiced_score  # noqa: E402
from voice_identity import SR, submit, timbre_vector  # noqa: E402

# Twinkle, as SoulX sang it: C C G G A A G.
MELODY = [(60, 0.5), (60, 0.5), (67, 0.5), (67, 0.5), (69, 0.5), (69, 0.5), (67, 1.0)]

VOICES = {
    "warm_male":   "warm male vocal, rich chest voice, intimate, solo a cappella",
    "bright_fem":  "bright female vocal, airy head voice, clear, solo a cappella",
    "raspy_soul":  "raspy soul vocal, gravelly, powerful, solo a cappella",
}
SEEDS = [11, 22]


def pitch_error(path: Path, notes) -> float:
    import librosa
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, sr=sr, fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C7"), frame_length=2048)
    times = librosa.times_like(f0, sr=sr)
    errs, t = [], 0.0
    for pitch, length in notes:
        sel = (times >= t + length * 0.2) & (times <= t + length * 0.8) & voiced
        vals = f0[sel][~np.isnan(f0[sel])]
        if vals.size:
            errs.append(abs(float(librosa.hz_to_midi(np.median(vals))) - pitch))
        t += length
    return float(np.mean(errs)) if errs else float("nan")


def distance(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(1.0 - np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def recover(src: Path, dest: Path, caption: str, noise: float, seed: int) -> float:
    return submit({
        "task_type": "cover", "prompt": caption, "lyrics": "",
        "vocal_language": "en",
        "cover_noise_strength": f"{noise:.3f}",
        "audio_cover_strength": "1.0",
        "inference_steps": "50", "guidance_scale": "7.0",
        "use_random_seed": "false", "seed": str(seed),
        "audio_format": "wav", "thinking": "false",
    }, {"src_audio": (src.name, src.read_bytes(), "audio/wav")}, dest)


def main():
    out = Path("/workspace/gate/revoice")
    out.mkdir(parents=True, exist_ok=True)
    src = Path("/workspace/gate/revoice/00_soulx_vocal.wav")
    if not src.exists():
        raise SystemExit(f"put the SoulX vocal at {src} first")

    zc, fl, v = voiced_score(src)
    pe = pitch_error(src, MELODY)
    print(f"source (SoulX)   voiced {v:5.2f}  pitch_err {pe:5.2f}  zcr {zc:.4f}\n")

    print(f"{'voice':13} {'noise':>6} {'seed':>5} {'voiced':>7} {'pitch_err':>10}")
    print("-" * 46)
    vectors: dict[str, list] = {}
    rows = []
    for noise in (0.35, 0.55):
        for name, caption in VOICES.items():
            key = f"{noise}:{name}"
            vectors[key] = []
            for seed in SEEDS:
                dest = out / f"n{int(noise*100)}_{name}_s{seed}.wav"
                try:
                    recover(src, dest, caption, noise, seed)
                except Exception as exc:                            # noqa: BLE001
                    print(f"{name:13} {noise:6.2f} {seed:5}  FAILED {str(exc)[:28]}")
                    continue
                zc, fl, vv = voiced_score(dest)
                pp = pitch_error(dest, MELODY)
                vectors[key].append(timbre_vector(dest))
                rows.append({"voice": name, "noise": noise, "seed": seed,
                             "voiced": vv, "pitch_err": pp, "file": dest.name})
                print(f"{name:13} {noise:6.2f} {seed:5} {vv:7.2f} {pp:10.2f}")

    print("\n" + "=" * 60)
    print("voice separation -- do different captions give different singers?")
    for noise in (0.35, 0.55):
        keys = [k for k in vectors if k.startswith(f"{noise}:") and len(vectors[k]) > 1]
        if len(keys) < 2:
            continue
        within = [distance(a, b) for k in keys
                  for a, b in itertools.combinations(vectors[k], 2)]
        between = [distance(a, b) for ka, kb in itertools.combinations(keys, 2)
                   for a in vectors[ka] for b in vectors[kb]]
        w, b = float(np.mean(within)), float(np.mean(between))
        print(f"  noise {noise}: within {w:.4f}  between {b:.4f}  ratio {b/w if w else 0:.2f}x")

    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nPass needs all three: voiced stays high, pitch_err stays low, and")
    print("the ratio is clearly above 1.0 (gate 2 got 0.68x, which selected nothing).")


if __name__ == "__main__":
    sys.exit(main())
