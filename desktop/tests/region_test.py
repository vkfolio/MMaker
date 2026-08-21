"""Did the regenerate touch only what was selected?

The plan states the check plainly: "regenerate bars 9-16, export, confirm in
FL Studio that only those bars changed." This does that without FL Studio, by
comparing the two versions of the stem second by second.

A repaint that bleeds outside its range is the failure mode that matters most,
because it is invisible: the result still sounds like music, it has just quietly
replaced work the user wanted kept. Timing tells you nothing about it, and
listening only catches it if you happen to listen to the right bar.

Usage:
    python desktop/tests/region_test.py <project_id> [stem_id]
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
from null_test import read_wav, db          # noqa: E402

BASE = os.environ.get("MUSICMAKER_URL", "").rstrip("/")
HEAD = {"X-API-Token": os.environ.get("MUSICMAKER_TOKEN", "")}

# Below this, two takes are the same audio -- the residual is the float noise
# of a WAV round-trip, not a difference anyone could hear.
QUIET_DB = -60.0


def main() -> int:
    if not BASE or len(sys.argv) < 2:
        print(__doc__)
        return 2
    project_id = sys.argv[1]
    want_stem = sys.argv[2] if len(sys.argv) > 2 else None

    project = requests.get(f"{BASE}/api/projects/{project_id}",
                           headers=HEAD, timeout=60).json()["project"]

    stem = None
    for s in project.get("stems", []):
        if want_stem and s["id"] != want_stem:
            continue
        repaints = [v for v in s["versions"] if v.get("op") == "repaint"]
        if repaints:
            stem = s
            break
    if not stem:
        print("no stem with a repaint version -- regenerate a range first")
        return 2

    versions = stem["versions"]
    repaint = next(v for v in reversed(versions) if v.get("op") == "repaint")
    index = versions.index(repaint)
    if index == 0:
        print("the repaint has no earlier version to compare against")
        return 2
    before = versions[index - 1]

    params = repaint.get("params") or {}
    start_s = float(params.get("start_s", 0.0))
    end_s = float(params.get("end_s", 0.0))
    print(f"stem {stem['track_class']}  repaint {start_s:.2f}-{end_s:.2f}s")
    print(f"  before {before['id']}\n  after  {repaint['id']}\n")

    tmp = Path(tempfile.mkdtemp(prefix="regiontest_"))
    audio = []
    for v in (before, repaint):
        r = requests.get(f"{BASE}/api/projects/{project_id}/audio/{v['audio']}",
                         headers=HEAD, timeout=300)
        r.raise_for_status()
        path = tmp / f"{v['id']}.wav"
        path.write_bytes(r.content)
        audio.append(path)

    a, rate = read_wav(audio[0])
    b, _ = read_wav(audio[1])
    n = min(len(a), len(b))
    a, b = a[:n, :2], b[:n, :2]

    # A repaint crossfades into its neighbours, so the edge seconds legitimately
    # differ a little. Judge the seconds that are wholly outside the range.
    margin = 1.0
    leaked, quiet_inside = [], []
    print("   sec   diff dBFS")
    for second in range(n // rate):
        seg_a = a[second * rate:(second + 1) * rate]
        seg_b = b[second * rate:(second + 1) * rate]
        diff = db(float(np.sqrt(np.mean((seg_a - seg_b) ** 2))))
        inside = start_s <= second < end_s
        outside_clear = (second + 1 <= start_s - margin) or (second >= end_s + margin)

        tag = ""
        if inside and diff <= QUIET_DB:
            quiet_inside.append(second)
            tag = "  selected but unchanged"
        elif inside:
            tag = "  selected, regenerated"
        elif outside_clear and diff > QUIET_DB:
            leaked.append((second, diff))
            tag = "  LEAKED OUTSIDE SELECTION"
        print(f"   {second:3}   {diff:8.1f}{tag}")

    print("\n" + "=" * 60)
    if leaked:
        print(f"FAIL  {len(leaked)} second(s) changed outside the selection:")
        for second, diff in leaked[:8]:
            print(f"        {second}s at {diff:.1f} dBFS")
        print("      A repaint is overwriting audio the user asked to keep.")
        return 1
    if len(quiet_inside) >= max(1, int(end_s - start_s)):
        print("FAIL  nothing inside the selection actually changed.")
        return 1
    print(f"PASS  only {start_s:.2f}-{end_s:.2f}s changed; everything else is")
    print(f"      unchanged to within {QUIET_DB:.0f} dBFS.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
