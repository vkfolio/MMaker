"""Phase 0 gate: is the interaction model viable against this backend?

ACE Studio's UX assumes "select a range, hit a tool, get a result" feels
immediate. This backend is a single-GPU queue with renders measured in minutes.
Before building a desktop app around that interaction, measure it.

This script times every tool the app would offer, at every quality tier, and
prints a verdict per tool against thresholds drawn from interaction research:

    < 10s   feels interactive        -- build the tool into the direct loop
    < 60s   tolerable with feedback  -- needs progress and must be cancellable
    < 5min  batch work               -- queue it; the user goes and does
                                        something else
    > 5min  offline                  -- schedule it; do not put it behind a
                                        click that looks live

Usage:
    export MUSICMAKER_URL=https://<pod>-8000.proxy.runpod.net
    export MUSICMAKER_TOKEN=<token>
    python research/latency_probe.py            # warm pod, all tiers
    python research/latency_probe.py --cold     # include first-call cost
    python research/latency_probe.py --quick    # fast tier only, one pass

Writes latency_report.json next to itself so the numbers survive the session.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

BASE = os.environ.get("MUSICMAKER_URL", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.environ.get("MUSICMAKER_TOKEN", "")
OUT = Path(__file__).parent / "latency_report.json"

# Thresholds, in seconds, and what each implies for the UI.
BANDS = [
    (10,   "interactive", "put it in the direct loop"),
    (60,   "tolerable",   "needs progress + cancel"),
    (300,  "batch",       "queue it; user leaves"),
    (10**9, "offline",    "schedule it; never behind a live-looking click"),
]


def band(seconds: float) -> tuple[str, str]:
    for limit, name, advice in BANDS:
        if seconds < limit:
            return name, advice
    return BANDS[-1][1], BANDS[-1][2]


def _params():
    return {"token": TOKEN} if TOKEN else {}


def api(method: str, path: str, **kw):
    url = f"{BASE}{path}"
    headers = kw.pop("headers", {})
    if TOKEN:
        headers["X-API-Token"] = TOKEN
    resp = requests.request(method, url, headers=headers, timeout=kw.pop("timeout", 60), **kw)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def wait(job_id: str, budget_s: float = 1800) -> tuple[float, dict]:
    """Block until a job finishes. Returns (elapsed, body)."""
    started = time.time()
    while True:
        body = api("GET", f"/api/jobs/{job_id}", params=_params())
        if body["status"] in ("done", "error"):
            return time.time() - started, body
        if time.time() - started > budget_s:
            return time.time() - started, {**body, "status": "timeout"}
        time.sleep(2)


def timed(label: str, fn, results: list):
    """Run one operation, record how long the user would actually wait."""
    print(f"  {label:<34} ", end="", flush=True)
    started = time.time()
    try:
        job = fn()
        elapsed, body = wait(job)
        ok = body["status"] == "done"
        err = (body.get("error") or "")[:80]
    except Exception as exc:                                  # noqa: BLE001
        elapsed, ok, err = time.time() - started, False, f"{type(exc).__name__}: {exc}"[:80]

    name, advice = band(elapsed)
    print(f"{elapsed:7.1f}s  {name:<12}" + ("" if ok else f"  FAILED {err}"))
    results.append({"op": label, "seconds": round(elapsed, 2), "band": name,
                    "advice": advice, "ok": ok, "error": err})
    return ok


def submit(path: str, payload: dict) -> str:
    return api("POST", path, json=payload, params=_params())["job"]["id"]


def probe(quality: str, results: list, bars: int = 8):
    print(f"\n=== quality: {quality} ===")

    pid = None

    def create():
        nonlocal pid
        body = api("POST", "/api/projects", params=_params(), json={
            "title": f"probe-{quality}", "prompt": "warm indie soul, brushed drums, rhodes",
            "style": "indie_soul", "bars": bars, "variations": 1, "quality": quality,
        })
        pid = body["project"]["id"]
        return body["job"]["id"]

    if not timed("generate 1 variation", create, results):
        print("  (generation failed; skipping the rest of this tier)")
        return

    project = api("GET", f"/api/projects/{pid}", params=_params())["project"]
    var_id = project["variations"][0]["id"]

    if not timed("split into stems", lambda: submit(
            f"/api/projects/{pid}/split", {"variation_id": var_id, "method": "demucs"}), results):
        return

    stems = api("GET", f"/api/projects/{pid}/stems", params=_params())["stems"]
    if not stems:
        print("  (no stems; skipping per-stem tools)")
        return
    sid = stems[0]["id"]

    # The interaction the whole product rests on: drag a range, regenerate it.
    timed("repaint a 4s region", lambda: submit(
        f"/api/projects/{pid}/stems/{sid}/repaint", {"start_s": 2.0, "end_s": 6.0}), results)
    timed("add a layer (strings)", lambda: submit(
        f"/api/projects/{pid}/layers", {"track_class": "strings", "prompt": "soft pad"}), results)
    timed("another take of a stem", lambda: submit(
        f"/api/projects/{pid}/stems/{sid}/vary", {}), results)
    timed("extend by 8 bars", lambda: submit(
        f"/api/projects/{pid}/stems/{sid}/extend", {"bars": 8}), results)
    timed("export stems", lambda: submit(f"/api/projects/{pid}/export", {}), results)

    # Download cost matters as much as render cost for a desktop client.
    started = time.time()
    try:
        r = requests.get(f"{BASE}/api/projects/{pid}/export/download",
                         params=_params(), timeout=600)
        r.raise_for_status()
        mb = len(r.content) / 1e6
        secs = time.time() - started
        print(f"  {'download export':<34} {secs:7.1f}s  {mb:.1f} MB "
              f"({mb / max(secs, 1e-6):.1f} MB/s)")
        results.append({"op": "download export", "seconds": round(secs, 2),
                        "band": band(secs)[0], "advice": "", "ok": True,
                        "error": "", "mb": round(mb, 1)})
    except Exception as exc:                                  # noqa: BLE001
        print(f"  download failed: {exc}")


def verdict(results: list):
    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)

    ok = [r for r in results if r["ok"]]
    if not ok:
        print("  Nothing completed. No verdict possible.")
        return

    # The selection-driven loop is the product thesis. Judge it specifically.
    core = [r for r in ok if r["op"].startswith(("repaint", "add a layer"))]
    if core:
        worst = max(r["seconds"] for r in core)
        name, advice = band(worst)
        print(f"\n  Selection-driven tools, worst case: {worst:.0f}s ({name})")
        print(f"  -> {advice}")
        if name == "interactive":
            print("  The ACE-Step-Studio interaction model is viable as designed.")
        elif name == "tolerable":
            print("  Viable, but every tool must show progress and be cancellable.")
        else:
            print("  NOT viable as a direct-manipulation loop. What you can build")
            print("  is a timeline with an asynchronous batch renderer attached --")
            print("  a good product, but a different one. Design the UX for that.")

    print(f"\n  {'operation':<34}{'seconds':>9}  band")
    print("  " + "-" * 60)
    for r in results:
        flag = "" if r["ok"] else "  (failed)"
        print(f"  {r['op']:<34}{r['seconds']:>8.1f}s  {r['band']}{flag}")

    total = sum(r["seconds"] for r in ok)
    print(f"\n  Total wall-clock for one full pass: {total / 60:.1f} min")
    print("  A single-GPU queue means concurrent users multiply this directly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast tier only")
    ap.add_argument("--cold", action="store_true",
                    help="do not warm the pod first, so first-call cost is included")
    args = ap.parse_args()

    print(f"engine: {BASE}")
    try:
        health = requests.get(f"{BASE}/health", timeout=30).json()
    except Exception as exc:                                  # noqa: BLE001
        sys.exit(f"cannot reach the engine: {exc}")
    print(f"status: {health.get('status')} | auth: {health.get('auth_required')}")
    gpu = health.get("gpu") or {}
    print(f"gpu:    {gpu.get('name', 'none')} {gpu.get('total_gb', '')}")
    if health.get("stub_models"):
        print("\nWARNING: stub engine. These numbers measure nothing real.\n")

    if not args.cold:
        print("\nwarming the pod (first call loads models)…")
        started = time.time()
        try:
            body = api("POST", "/api/projects", params=_params(), json={
                "title": "warmup", "prompt": "simple drum groove", "bars": 4,
                "variations": 1, "quality": "fast"})
            wait(body["job"]["id"])
            print(f"  warm-up took {time.time() - started:.0f}s "
                  f"(a cold pod pays this once)")
        except Exception as exc:                              # noqa: BLE001
            print(f"  warm-up failed: {exc}")

    results = []
    for quality in (["fast"] if args.quick else ["fast", "high", "ultra"]):
        probe(quality, results)

    verdict(results)
    OUT.write_text(json.dumps({"base": BASE, "gpu": gpu, "results": results},
                              indent=2), encoding="utf-8")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
