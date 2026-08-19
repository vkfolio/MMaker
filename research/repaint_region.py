"""Phase 0: does `repaint` really leave the rest of the layer alone?

The whole editing model assumes "regenerate bars 17-24" touches only bars
17-24. We verify by null-testing the untouched head and tail against the
original. If the residual outside the mask is not near zero, every repaint
silently rewrites the whole layer and the version model has to change.

Usage:
    python research/repaint_region.py research/out/lego_chain_ab12cd 02_bass.wav
"""

import json
import sys
from pathlib import Path

import numpy as np

import acestep
from acestep import run
import audio_utils as au


def bars_to_seconds(bar, bpm, beats_per_bar=4):
    return (bar - 1) * beats_per_bar * 60.0 / bpm


def region_residual(path_a, path_b, start_s, end_s, mode):
    """Null-test only the portion outside (or inside) the repainted region."""
    a, sr = au.load(path_a)
    b, sr_b = au.load(path_b)
    if sr != sr_b:
        raise ValueError(f"sample-rate mismatch: {sr} vs {sr_b}")

    n = min(len(a), len(b))
    w = min(a.shape[1], b.shape[1])
    a, b = a[:n, :w], b[:n, :w]

    i0, i1 = int(start_s * sr), int(end_s * sr)
    mask = np.zeros(n, dtype=bool)
    mask[max(0, i0):min(n, i1)] = True
    if mode == "outside":
        mask = ~mask

    if not mask.any():
        return None
    seg_a, seg_b = a[mask], b[mask]
    rms_a = float(np.sqrt(np.mean(seg_a ** 2))) or 1e-12
    residual = float(np.sqrt(np.mean((seg_a - seg_b) ** 2)))
    return residual / rms_a


def main(run_dir, stem_name, start_bar=17, end_bar=25):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    bpm = grid["bpm"]

    source = run_dir / stem_name
    if not source.exists():
        sys.exit(f"no such stem: {source}")

    start_s = bars_to_seconds(start_bar, bpm)
    end_s = bars_to_seconds(end_bar, bpm)
    duration = au.describe(source)["duration_s"]
    if start_s >= duration:
        sys.exit(f"bar {start_bar} starts at {start_s:.1f}s but the stem is "
                 f"only {duration:.1f}s -- pick earlier bars or a longer render")
    end_s = min(end_s, duration)

    layer = next((l for l in manifest["layers"]
                  if Path(l["audio"]["path"]).name == stem_name), None)
    prompt = layer["params"].get("prompt", "") if layer else ""
    track_name = layer["track_name"] if layer else "unknown"

    print(f"repainting {stem_name} ({track_name}) bars {start_bar}-{end_bar} "
          f"= {start_s:.2f}s..{end_s:.2f}s of {duration:.2f}s\n")

    params = {
        **{k: grid[k] for k in ("bpm", "key_scale", "inference_steps")},
        "task_type": "repaint",
        "prompt": prompt,
        "repainting_start": round(start_s, 3),
        "repainting_end": round(end_s, 3),
        "chunk_mask_mode": "explicit",
        "audio_duration": duration,
        # Deliberately a different seed -- if the region does not change at
        # all, repaint is not doing anything and that is also a finding.
        "seed": grid["seed"] + 1,
    }

    dest = run_dir / f"repaint_{stem_name}"
    path, metas, secs = run(params, dest, src_audio_path=source)

    outside = region_residual(source, path, start_s, end_s, "outside")
    inside = region_residual(source, path, start_s, end_s, "inside")

    print(f"\nresidual OUTSIDE the mask: {outside:.4f}  (want ~0.0 -- context preserved)")
    print(f"residual INSIDE  the mask: {inside:.4f}  (want > 0.2 -- region actually changed)")

    verdict = "PASS"
    if outside is None or outside > 0.05:
        verdict = "FAIL -- repaint rewrites audio outside the requested region"
    elif inside is not None and inside < 0.05:
        verdict = "FAIL -- repaint did not change the requested region"
    print(f"\n{verdict}")

    report = {
        "stem": stem_name, "track_name": track_name,
        "bars": [start_bar, end_bar], "seconds": [start_s, end_s],
        "params": params, "metas": metas, "gpu_seconds": round(secs, 1),
        "residual_outside": outside, "residual_inside": inside,
        "verdict": verdict,
    }
    out = run_dir / f"repaint_report_{Path(stem_name).stem}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    extra = [int(a) for a in sys.argv[3:5]]
    try:
        main(sys.argv[1], sys.argv[2], *extra)
    except acestep.AceStepError as exc:
        sys.exit(f"ACE-Step error: {exc}")
