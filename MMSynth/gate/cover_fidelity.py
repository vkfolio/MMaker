"""Phase 0, gate 1: can ACE-Step XL be driven by a melody?

ACE-Step has no note input. The only way a melody reaches it is as audio: render
the MIDI with a sampler, hand that to `cover`, and hope the model keeps the tune
while replacing the timbre. This script measures whether that hope is justified,
because if it is not, feature 1 does not exist and everything built on it is
wasted.

Two knobs, easy to confuse, both swept here:

  cover_noise_strength   SDEdit. 0 = start from pure noise and ignore the source
                         waveform, 1 = hug it. Their UI calls this "Cover
                         Strength (Melody Retention)" and suggests 0.1-0.25 for
                         SFT checkpoints. Note the inversion versus the usual
                         `strength` convention.
  audio_cover_strength   Not a blend. cover_steps = int(infer_steps * this);
                         after that many steps the model drops the source
                         conditioning entirely and continues on caption alone.

The pass condition is a conjunction, and both halves matter:

  * the melody survives -- rendered pitch tracks the MIDI within half a semitone
  * the timbre actually changed -- the output is not just the soundfont back

A setting that scores perfectly on pitch by ignoring the caption has failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

BASE = "http://127.0.0.1:8011"
SR = 48000

# Twinkle Twinkle: C C G G A A G. Chosen because the tune is unmistakable by
# ear, the leap C->G is big enough that a drifting model is obvious, and the
# repeated notes catch a model that slurs onsets together.
MELODY = [(60, 0.5), (60, 0.5), (67, 0.5), (67, 0.5), (69, 0.5), (69, 0.5), (67, 1.0)]
BPM = 120
CAPTION = "solo violin, expressive vibrato, rosin and bow noise, concert hall"

TICKS = 480


def write_smf(notes, dest: Path, bpm: float = 120.0, program: int = 40) -> Path:
    """Minimal format-0 SMF. Same encoder MMSynth uses, inlined so the gate has
    no dependency on the service being installed."""
    def vlq(v: int) -> bytes:
        out = bytearray([v & 0x7F])
        v >>= 7
        while v:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(reversed(out))

    tps = TICKS * bpm / 60.0
    events = []
    t = 0.0
    for pitch, length in notes:
        start = round(t * tps)
        events.append((start, 1, pitch, 96))
        events.append((start + max(1, round(length * tps)), 0, pitch, 0))
        t += length
    events.sort(key=lambda e: (e[0], e[1]))

    body = bytearray()
    body += vlq(0) + b"\xff\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big")
    body += vlq(0) + bytes([0xC0, program])
    prev = 0
    for tick, kind, pitch, vel in events:
        body += vlq(tick - prev)
        prev = tick
        body += bytes([(0x90 if kind else 0x80), pitch, vel])
    body += vlq(0) + b"\xff\x2f\x00"

    head = (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + TICKS.to_bytes(2, "big")
    dest.write_bytes(b"MThd" + len(head).to_bytes(4, "big") + head
                     + b"MTrk" + len(body).to_bytes(4, "big") + bytes(body))
    return dest


def soundfont() -> str:
    for p in ("/usr/share/sounds/sf2/FluidR3_GM.sf2",
              "/usr/share/sounds/sf2/default-GM.sf2"):
        if Path(p).exists():
            return p
    raise SystemExit("no soundfont; apt-get install fluid-soundfont-gm")


def render_midi(midi: Path, dest: Path) -> Path:
    subprocess.run(["fluidsynth", "-ni", "-F", str(dest), "-r", str(SR), "-g", "0.8",
                    soundfont(), str(midi)], check=True, capture_output=True)
    return dest


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def note_pitches(path: Path, notes) -> list[float]:
    """Median detected pitch (in MIDI numbers) inside each note's own window.

    pyin rather than a spectral peak: this is monophonic melodic audio, which is
    exactly what pyin is for, and it reports voicing so silence does not become
    a confident wrong answer.
    """
    import librosa
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(
        y, sr=sr, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        frame_length=2048)
    times = librosa.times_like(f0, sr=sr)

    out = []
    t = 0.0
    for _pitch, length in notes:
        # Trim 20% off each end: attacks and releases are where a pitch tracker
        # is least trustworthy, and they are not what "is the note right" means.
        lo, hi = t + length * 0.2, t + length * 0.8
        sel = (times >= lo) & (times <= hi) & voiced
        vals = f0[sel]
        vals = vals[~np.isnan(vals)]
        out.append(float(librosa.hz_to_midi(np.median(vals))) if vals.size else float("nan"))
        t += length
    return out


def pitch_error(path: Path, notes) -> tuple[float, float, int]:
    """(mean abs error in semitones, % of notes within 0.5, notes tracked)."""
    got = note_pitches(path, notes)
    want = [p for p, _ in notes]
    errs = [abs(g - w) for g, w in zip(got, want) if not np.isnan(g)]
    if not errs:
        return float("nan"), 0.0, 0
    within = 100.0 * sum(1 for e in errs if e <= 0.5) / len(errs)
    return float(np.mean(errs)), within, len(errs)


def spectral_distance(a: Path, b: Path) -> float:
    """How different two renders sound, as mean abs difference of log-mel.

    Stands in for "did the timbre actually change". Crude, but it separates
    'the model returned the soundfont' from 'the model played something' without
    anyone having to listen to thirty files.
    """
    import librosa
    ya, _ = librosa.load(str(a), sr=SR, mono=True)
    yb, _ = librosa.load(str(b), sr=SR, mono=True)
    n = min(len(ya), len(yb))
    ma = librosa.power_to_db(librosa.feature.melspectrogram(y=ya[:n], sr=SR, n_mels=64))
    mb = librosa.power_to_db(librosa.feature.melspectrogram(y=yb[:n], sr=SR, n_mels=64))
    m = min(ma.shape[1], mb.shape[1])
    return float(np.mean(np.abs(ma[:, :m] - mb[:, :m])))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

def cover(src: Path, dest: Path, *, task_type: str, noise: float, strength: float,
          steps: int, caption: str, seed: int = 12345, timeout: int = 900) -> dict:
    with open(src, "rb") as fh:
        data = {
            "task_type": task_type,
            "prompt": caption,
            "lyrics": "",
            "cover_noise_strength": str(noise),
            "audio_cover_strength": str(strength),
            "inference_steps": str(steps),
            "guidance_scale": "7.0",
            "use_random_seed": "false",
            "seed": str(seed),
            "audio_format": "wav",
            "thinking": "false",
        }
        r = requests.post(f"{BASE}/release_task", data=data,
                          files={"src_audio": (src.name, fh, "audio/wav")}, timeout=120)
    r.raise_for_status()
    task_id = r.json()["data"]["task_id"]

    started = time.time()
    while time.time() - started < timeout:
        time.sleep(3)
        q = requests.post(f"{BASE}/query_result", json={"task_id_list": [task_id]},
                          timeout=60)
        q.raise_for_status()
        payload = q.json()["data"][0]

        # The shape is nested twice and neither layer is obvious: `data` is a
        # list of task envelopes, and each envelope's `result` is a JSON *string*
        # holding a *list* of results. The status that matters is the inner one;
        # the envelope has none until the task is done.
        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            result = json.loads(result)
        if isinstance(result, list):
            result = result[0] if result else None
        if not isinstance(result, dict):
            continue

        status = result.get("status")
        if status == 1:
            url = result["file"]
            audio = requests.get(url if url.startswith("http") else BASE + url, timeout=300)
            audio.raise_for_status()
            dest.write_bytes(audio.content)
            return {"seconds": round(time.time() - started, 1)}
        if status == 2:
            raise RuntimeError(f"task failed: {str(result)[:300]}")
    raise RuntimeError(f"task timed out after {timeout}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/gate/cover")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    midi = write_smf(MELODY, out / "melody.mid", BPM)
    dry = render_midi(midi, out / "00_sampler.wav")
    err, within, tracked = pitch_error(dry, MELODY)
    print(f"\nsampler reference: pitch err {err:.2f} st, {within:.0f}% within 0.5, "
          f"{tracked}/{len(MELODY)} notes tracked")
    print("(this is the floor -- the melody carrier, before any model)\n")

    # cover_noise_strength is the melody-retention knob; sweep it widely because
    # their recommended 0.1-0.25 is for whole songs, not a bare sampler line.
    grid = []
    for task in ("cover", "cover-nofsq"):
        for noise in (0.15, 0.35, 0.55, 0.75, 0.9):
            grid.append((task, noise, 1.0))
    grid.append(("cover", 0.55, 0.6))     # does the early switchover matter?

    print(f"{'task':12} {'noise':>6} {'strength':>8} {'pitch_err':>10} "
          f"{'within.5':>9} {'tracked':>8} {'timbreΔ':>8} {'secs':>6}")
    print("-" * 76)

    rows = []
    for task, noise, strength in grid:
        name = f"{task}_n{noise}_s{strength}".replace(".", "")
        dest = out / f"{name}.wav"
        try:
            info = cover(dry, dest, task_type=task, noise=noise, strength=strength,
                         steps=args.steps, caption=CAPTION)
        except Exception as exc:                                  # noqa: BLE001
            print(f"{task:12} {noise:6.2f} {strength:8.2f}   FAILED: {str(exc)[:40]}")
            continue
        e, w, t = pitch_error(dest, MELODY)
        d = spectral_distance(dry, dest)
        rows.append({"task": task, "noise": noise, "strength": strength,
                     "pitch_err": e, "within": w, "tracked": t, "timbre": d,
                     "seconds": info["seconds"], "file": dest.name})
        print(f"{task:12} {noise:6.2f} {strength:8.2f} {e:10.2f} {w:8.0f}% "
              f"{t:>4}/{len(MELODY)} {d:8.1f} {info['seconds']:6.1f}")

    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/results.json and {len(rows)} wavs")
    print("\nA row passes only if BOTH hold: pitch_err <= 0.5 (the tune survived)")
    print("and timbreΔ is clearly above the noise floor (the timbre changed).")


if __name__ == "__main__":
    sys.exit(main())
