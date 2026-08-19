"""Phase 0: measure how well the generated layers actually line up.

Cumulative-context conditioning buys musical coherence, not sample-accurate
alignment. This quantifies the gap. For each stem we detect onsets, snap them
to the project's ideal beat grid, and report the deviation. Target is a median
under 20 ms; beyond that we need warp/nudge correction in the backend.

Usage:
    python research/sync_check.py research/out/lego_chain_ab12cd
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np

import audio_utils as au

# Only onsets this close to a grid line are treated as intended hits;
# anything further is syncopation or a sustained note, not evidence of drift.
SNAP_WINDOW_S = 0.12


def analyse(path, bpm, subdivision=4):
    """Deviation of detected onsets from the ideal grid, in milliseconds."""
    data, sr = au.load(path)
    y = au.to_mono(data)

    grid_step = 60.0 / bpm / (subdivision / 4.0)  # seconds per grid line
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)

    deviations = []
    for t in onsets:
        nearest = round(t / grid_step) * grid_step
        delta = t - nearest
        if abs(delta) <= SNAP_WINDOW_S:
            deviations.append(delta * 1000.0)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=bpm)
    tempo = float(np.atleast_1d(tempo)[0])

    if not deviations:
        return {"path": Path(path).name, "onsets": len(onsets),
                "snapped": 0, "detected_bpm": round(tempo, 2),
                "note": "no onsets near the grid -- pad or sustained layer?"}

    dev = np.asarray(deviations)
    return {
        "path": Path(path).name,
        "onsets": len(onsets),
        "snapped": len(dev),
        "detected_bpm": round(tempo, 2),
        "bpm_error": round(tempo - bpm, 2),
        "median_dev_ms": round(float(np.median(np.abs(dev))), 2),
        "p90_dev_ms": round(float(np.percentile(np.abs(dev), 90)), 2),
        "max_dev_ms": round(float(np.max(np.abs(dev))), 2),
        # A consistently signed offset is latency we can fix with a single
        # nudge; a scattered one is real drift and much harder to repair.
        "mean_signed_ms": round(float(np.mean(dev)), 2),
    }


def main(run_dir):
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"no manifest.json in {run_dir} -- run lego_chain.py first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bpm = manifest["grid"]["bpm"]
    print(f"project grid: {bpm} BPM, {manifest['grid']['key_scale']}\n")

    rows = []
    for layer in manifest["layers"]:
        path = run_dir / Path(layer["audio"]["path"]).name
        if not path.exists():
            print(f"  missing: {path.name}")
            continue
        result = analyse(path, bpm)
        result["track_name"] = layer["track_name"]
        rows.append(result)

    header = f"{'layer':<16}{'bpm':>8}{'err':>7}{'median':>9}{'p90':>8}{'max':>8}{'signed':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "median_dev_ms" not in r:
            print(f"{r['track_name']:<16}{r['detected_bpm']:>8}   {r.get('note', '')}")
            continue
        print(f"{r['track_name']:<16}{r['detected_bpm']:>8}{r['bpm_error']:>7}"
              f"{r['median_dev_ms']:>9}{r['p90_dev_ms']:>8}{r['max_dev_ms']:>8}"
              f"{r['mean_signed_ms']:>9}")

    (run_dir / "sync_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    measured = [r["median_dev_ms"] for r in rows if "median_dev_ms" in r]
    if measured:
        worst = max(measured)
        print(f"\nworst median deviation: {worst} ms", end="  ")
        if worst < 20:
            print("PASS -- layers are tight enough to ship as-is")
        elif worst < 50:
            print("MARGINAL -- backend needs a per-layer nudge/quantize step")
        else:
            print("FAIL -- drift correction is a first-class problem, not a polish step")
    print(f"\nreport: {run_dir / 'sync_report.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
