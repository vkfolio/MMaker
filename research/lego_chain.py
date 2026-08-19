"""Phase 0 gate script: build a song layer by layer and listen to it.

drums -> bass -> keys -> vocals, every call locked to the same bpm, key,
seed and duration, and every call conditioned on the mixdown of everything
generated so far. If the result is not musical, the whole product premise
is wrong and we stop before writing any app code.

Usage:
    export ACESTEP_URL=http://<runpod-host>:8001
    python research/lego_chain.py
"""

import json
import sys

import acestep
from acestep import run, new_run_dir
import audio_utils as au

# --- project grid: identical across every call, this is the sync contract ---
GRID = {
    "bpm": 96,
    "key_scale": "A Minor",
    "audio_duration": 30.0,
    "seed": 20260818,
    "inference_steps": 60,
}

STYLE = "warm indie soul, live drums, analog bass, rhodes, intimate"

SEED_LAYER = {
    "task_type": "text2music",
    "prompt": f"{STYLE}, solo drum groove only, no other instruments",
}

# Each subsequent layer hears the mixdown of all previous layers.
LAYERS = [
    {"track_name": "bass",     "prompt": f"{STYLE}, melodic analog bass line locking to the drums"},
    {"track_name": "keyboard", "prompt": f"{STYLE}, rhodes chords, sparse and warm"},
    {"track_name": "vocals",   "prompt": f"{STYLE}, soulful lead vocal",
     "lyrics": "[verse]\nSlow light on the kitchen floor\nI am not in a hurry anymore\n",
     "vocal_language": "en"},
]


def main():
    out = new_run_dir("lego_chain")
    print(f"server : {acestep.BASE_URL}")
    print(f"output : {out}\n")

    manifest = {"grid": GRID, "server": acestep.BASE_URL, "layers": []}
    stems = []

    # --- seed layer -------------------------------------------------------
    print("[1] seed layer (drums) via text2music")
    params = {**GRID, **SEED_LAYER}
    path, metas, secs = run(params, out / "01_drums.wav")
    stems.append(path)
    manifest["layers"].append({
        "index": 1, "track_name": "drums", "params": params, "metas": metas,
        "seconds": round(secs, 1), "audio": au.describe(path),
    })

    # --- lego layers ------------------------------------------------------
    for i, layer in enumerate(LAYERS, start=2):
        name = layer["track_name"]
        print(f"\n[{i}] lego layer: {name}")

        context, sr = au.mixdown(stems, out / f"{i:02d}_context.wav")
        print(f"  context = mix of {len(stems)} stem(s) @ {sr} Hz")

        params = {**GRID, **layer, "task_type": "lego"}
        path, metas, secs = run(params, out / f"{i:02d}_{name}.wav",
                                src_audio_path=context)

        # Diagnostic: is the returned audio a solo stem, or the full mix with
        # the new part added? The docs are ambiguous and it changes the whole
        # data model, so measure it rather than assume.
        residual = au.null_test(context, path)
        contains_context = residual < 0.5
        print(f"  null-test vs context: {residual:.3f} "
              f"-> looks like {'FULL MIX' if contains_context else 'SOLO STEM'}")

        stems.append(path)
        manifest["layers"].append({
            "index": i, "track_name": name, "params": params, "metas": metas,
            "seconds": round(secs, 1), "audio": au.describe(path),
            "null_test_vs_context": round(residual, 4),
            "output_looks_like": "full_mix" if contains_context else "solo_stem",
        })

    # --- final mix --------------------------------------------------------
    final, sr = au.mixdown(stems, out / "99_final_mix.wav")
    manifest["final_mix"] = au.describe(final)
    manifest["total_seconds"] = round(sum(l["seconds"] for l in manifest["layers"]), 1)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\ndone. {manifest['total_seconds']}s of GPU wall-clock for "
          f"{len(manifest['layers'])} layers")
    print(f"listen to: {final}")
    print(f"then run:  python research/sync_check.py {out}")


if __name__ == "__main__":
    try:
        main()
    except acestep.AceStepError as exc:
        print(f"\nACE-Step error: {exc}", file=sys.stderr)
        print("Raw responses saved under research/out/_raw/ -- inspect them to "
              "pin down the real API schema.", file=sys.stderr)
        sys.exit(1)
