"""The null test: does the desktop mixer agree with the server's?

The plan names this as the verification that matters, and names the trap in it:

    `audio.mix` peak-normalises to -1 dBFS, so the app's mixdown will differ
    from the server's by a global gain. The criterion is "nulls after
    gain-matching, to -60 dB", not "cancels to silence".

So this pulls a project's stems, asks the server for its own mixdown of them,
bounces the same stems locally through the desktop mixer, then subtracts the two
after solving for the single scalar that best matches their levels. What is left
is everything the two mixers disagree about.

A residual at -60 dBFS or below means the difference is inaudible and the mixer
is doing arithmetic the server would recognise. Anything louder is a real
disagreement -- a gain law, a pan law, an off-by-one in placement, a resampling
difference -- and the number tells you roughly how bad.

Usage:
    python desktop/tests/null_test.py <project_id>
    MUSICMAKER_URL=... MUSICMAKER_TOKEN=... python desktop/tests/null_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import requests

BASE = os.environ.get("MUSICMAKER_URL", "").rstrip("/")
TOKEN = os.environ.get("MUSICMAKER_TOKEN", "")
MUSICX = Path(__file__).resolve().parents[1] / "build" / "Release" / "musicx.exe"

HEAD = {"X-API-Token": TOKEN} if TOKEN else {}


def api(method: str, path: str, **kw):
    r = requests.request(method, BASE + path, headers=HEAD, timeout=120, **kw)
    r.raise_for_status()
    return r.json() if r.content else {}


def wait(job_id: str, budget: float = 900) -> dict:
    started = time.time()
    while time.time() - started < budget:
        body = api("GET", f"/api/jobs/{job_id}")
        if body["status"] in ("done", "error"):
            return body
        time.sleep(3)
    return {"status": "timeout"}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Reads a WAV as float64 [frames, channels].

    Hand-rolled rather than using `wave`, which rejects format 3 (IEEE float)
    outright -- and float is exactly what a bounce should be written in, so the
    standard library cannot read the thing under test.
    """
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise SystemExit(f"{path.name} is not a RIFF/WAVE file")

    fmt_tag = channels = rate = bits = None
    data = None
    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = int.from_bytes(raw[pos + 4:pos + 8], "little")
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt_tag = int.from_bytes(body[0:2], "little")
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            if fmt_tag == 0xFFFE and len(body) >= 26:      # WAVE_FORMAT_EXTENSIBLE
                fmt_tag = int.from_bytes(body[24:26], "little")
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)                       # chunks are word-aligned

    if data is None or not channels:
        raise SystemExit(f"{path.name}: no data chunk")

    if fmt_tag == 3 and bits == 32:
        samples = np.frombuffer(data, dtype="<f4").astype(np.float64)
    elif fmt_tag == 3 and bits == 64:
        samples = np.frombuffer(data, dtype="<f8").astype(np.float64)
    elif fmt_tag == 1 and bits == 16:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    elif fmt_tag == 1 and bits == 32:
        samples = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
    elif fmt_tag == 1 and bits == 24:
        b3 = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b3[:, 0] | (b3[:, 1] << 8) | (b3[:, 2] << 16)
        samples = np.where(ints & 0x800000, ints - 0x1000000, ints) / 8388608.0
    else:
        raise SystemExit(f"{path.name}: unsupported format {fmt_tag} / {bits}-bit")

    usable = (len(samples) // channels) * channels
    return samples[:usable].reshape(-1, channels), rate


def db(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-12))


def main() -> int:
    if not BASE:
        return fail("set MUSICMAKER_URL (and MUSICMAKER_TOKEN if the pod needs one)")
    if not MUSICX.exists():
        return fail(f"build musicx first -- not found at {MUSICX}")

    project_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not project_id:
        projects = api("GET", "/api/projects")["projects"]
        for p in projects:
            full = api("GET", f"/api/projects/{p['id']}")["project"]
            if len(full.get("stems") or []) >= 2:
                project_id = p["id"]
                print(f"using project {p['title']!r} ({project_id})")
                break
        if not project_id:
            return fail("no project with stems -- split one first, or pass an id")

    project = api("GET", f"/api/projects/{project_id}")["project"]
    stems = project.get("stems") or []
    if len(stems) < 2:
        return fail(f"{project_id} has {len(stems)} stems; need at least 2")

    work = Path(tempfile.mkdtemp(prefix="nulltest_"))
    stem_dir = work / "stems"
    stem_dir.mkdir()

    print(f"\ndownloading {len(stems)} stems…")
    for stem in stems:
        versions = stem.get("versions") or []
        if not versions:
            continue
        current = stem.get("current", 0)
        version = versions[current if isinstance(current, int) else -1]
        rel = version.get("audio")
        if not rel:
            continue
        r = requests.get(f"{BASE}/api/projects/{project_id}/audio/{rel}",
                         headers=HEAD, timeout=300)
        r.raise_for_status()
        # Named by stem id: the local bounce sums whatever is in the folder, and
        # two stems sharing a track_class would otherwise overwrite each other.
        (stem_dir / f"{stem['id']}.wav").write_bytes(r.content)
    got = sorted(stem_dir.glob("*.wav"))
    print(f"  {len(got)} files")

    print("asking the server for its own mixdown…")
    # GET, not POST: the server mixes on demand and streams the result back.
    r = requests.get(f"{BASE}/api/projects/{project_id}/mixdown",
                     headers=HEAD, timeout=300)
    r.raise_for_status()
    reference = work / "server_mix.wav"
    reference.write_bytes(r.content)

    print("bouncing the same stems through the desktop mixer…")
    mine = work / "desktop_mix.wav"
    proc = subprocess.run([str(MUSICX), str(stem_dir), "--bounce", str(mine)],
                          capture_output=True, text=True, timeout=600)
    print("  " + (proc.stdout.strip().replace("\n", "\n  ") or proc.stderr[:200]))
    if not mine.exists():
        return fail("the bounce produced no file")

    a, rate_a = read_wav(reference)
    b, rate_b = read_wav(mine)
    if rate_a != rate_b:
        return fail(f"sample rates differ: server {rate_a}, desktop {rate_b}")

    n = min(len(a), len(b))
    ch = min(a.shape[1], b.shape[1])
    a, b = a[:n, :ch], b[:n, :ch]
    print(f"\ncomparing {n / rate_a:.1f}s, {ch} channels")
    if abs(len(a) - len(b)) > rate_a * 0.02:
        print(f"  note: lengths differ by {abs(len(a) - len(b)) / rate_a:.2f}s "
              f"(trailing tail excluded from the comparison)")

    # The one scalar that best matches the levels -- least squares, which is the
    # honest way to remove a global gain without hand-tuning it toward a pass.
    denom = float(np.sum(b * b))
    if denom <= 0:
        return fail("the desktop mix is silent")
    gain = float(np.sum(a * b) / denom)

    residual = a - gain * b
    peak_ref = float(np.max(np.abs(a)))
    rms_res = float(np.sqrt(np.mean(residual ** 2)))
    peak_res = float(np.max(np.abs(residual)))
    rms_ref = float(np.sqrt(np.mean(a ** 2)))

    print(f"\n  gain applied to desktop mix : {gain:.4f}  ({db(gain):+.2f} dB)")
    print(f"  server mix peak             : {db(peak_ref):.1f} dBFS")
    print(f"  residual RMS                : {db(rms_res):.1f} dBFS")
    print(f"  residual peak               : {db(peak_res):.1f} dBFS")
    print(f"  residual vs signal          : {db(rms_res) - db(rms_ref):.1f} dB down")

    print("\n" + "=" * 62)
    if db(rms_res) <= -60.0:
        print(f"PASS  residual at {db(rms_res):.1f} dBFS, at or below -60 dBFS.")
        print("      The two mixers agree to inaudibility after gain matching.")
        ok = True
    else:
        print(f"FAIL  residual at {db(rms_res):.1f} dBFS, above the -60 dBFS bar.")
        print("      The mixers genuinely disagree. Usual suspects, in order:")
        print("        - pan law (equal-power vs linear) on non-centred stems")
        print("        - stem gain_db applied by one side and not the other")
        print("        - a one-sample placement offset")
        print("        - resampling, if any stem was not already at the device rate")
        ok = False
    print("=" * 62)
    print(f"\nfiles kept for inspection: {work}")
    return 0 if ok else 1


def fail(message: str) -> int:
    print(f"null test: {message}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
