"""Phase 0 step one: confirm the pod works and pin down the real API schema.

Run this before anything else. It answers the open items in the plan:
the actual output sample rate, cold-start time, per-render wall clock, and
the true shape of the /release_task and /query_result payloads.

Usage:
    export ACESTEP_URL=http://<runpod-host>:8001
    python research/probe.py
"""

import json
import sys
import time

import requests

import acestep
from acestep import run, new_run_dir
import audio_utils as au


def reachable():
    for path in ("/", "/docs", "/openapi.json"):
        try:
            resp = requests.get(f"{acestep.BASE_URL}{path}", timeout=10)
            print(f"  GET {path:<16} -> {resp.status_code}")
            if path == "/openapi.json" and resp.status_code == 200:
                spec = resp.json()
                paths = sorted(spec.get("paths", {}))
                print(f"  openapi advertises {len(paths)} routes: {paths}")
                out = acestep.OUT / "_raw" / "openapi.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
                print(f"  saved full spec to {out}")
                return True
        except requests.RequestException as exc:
            print(f"  GET {path:<16} -> {type(exc).__name__}")
    return False


def main():
    print(f"probing {acestep.BASE_URL}\n")
    if not reachable():
        print("\nServer did not respond. Check the RunPod port mapping and that "
              "the ACE-Step server is actually listening on 8001.", file=sys.stderr)
        sys.exit(1)

    out = new_run_dir("probe")
    params = {
        "task_type": "text2music",
        "prompt": "simple solo drum groove, dry room, no other instruments",
        "bpm": 96,
        "key_scale": "A Minor",
        "audio_duration": 10.0,
        "seed": 20260818,
        "inference_steps": 30,
    }

    print("\nshortest possible render (10s, 30 steps) -- this is cold start + compute")
    started = time.time()
    path, metas, secs = run(params, out / "probe.wav")
    total = time.time() - started

    info = au.describe(path)
    print("\n--- results -------------------------------------------------")
    print(f"  sample rate  : {info['sample_rate']} Hz   <- open item in the plan")
    print(f"  channels     : {info['channels']}")
    print(f"  duration     : {info['duration_s']}s (asked for {params['audio_duration']}s)")
    print(f"  subtype      : {info['subtype']}")
    print(f"  wall clock   : {total:.1f}s for 10s of audio")
    print(f"  server metas : {metas}")
    print("\nraw API payloads saved under research/out/_raw/ -- read them and "
          "tighten acestep.py before running lego_chain.py")

    (out / "probe_report.json").write_text(
        json.dumps({"audio": info, "metas": metas, "wall_clock_s": round(total, 1),
                    "params": params}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except acestep.AceStepError as exc:
        sys.exit(f"ACE-Step error: {exc}")
