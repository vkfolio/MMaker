"""Phase 0, gate 2: can a singer be *picked* on ACE-Step XL?

ACE-Step has no voice bank. musicmaker's README calls this "the one real gap",
and it is the thing that decides whether "select a singer preset" can honestly
exist. Three mechanisms, tried cheapest first, stopping at whatever works:

  A  caption tags only        "female pop vocal, breathy, bright"
  B  caption + reference_audio  a 30 s clip through AceStepTimbreEncoder,
                                documented as global and as *removing* melody and
                                rhythm -- exactly right when the melody is
                                already coming from src_audio
  C  a trained LoRA           not here; it trains only the DiT decoder's four
                              attention projections with the timbre and lyric
                              encoders frozen, and is one-adapter-at-a-time

The measurement is consistency, because consistency is the entire point of a
preset. For each mechanism we render the same lyrics with **different seeds** and
ask how far apart the voices are. Then we compare that to the distance between
two *deliberately different* voices. A mechanism works only if:

    within(same preset, different seeds)  <<  between(different presets)

If within is as large as between, the "preset" is not selecting anything and a
user asking for the same singer twice will not get it.

Timbre distance is MFCC-statistics based -- means and standard deviations of 20
MFCCs. That is a proxy for a speaker embedding, not a speaker embedding, and it
is honest about voice colour without dragging a speaker-ID model onto the pod.
Take the ratio seriously and the absolute numbers lightly.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

BASE = "http://127.0.0.1:8011"
SR = 48000

# A singable line, deliberately plain so the words are not the variable.
MELODY = [(67, 0.5), (69, 0.5), (71, 0.5), (72, 1.0),
          (71, 0.5), (69, 0.5), (67, 1.5)]
LYRICS = "hold me close and don't let go"
BPM = 120
TICKS = 480

PRESETS = {
    "female_bright": "female pop vocal, bright, clear, close mic, solo voice",
    "male_raspy": "male soul vocal, raspy, warm, gravelly, solo voice",
}


def write_smf(notes, dest: Path, bpm: float = 120.0, program: int = 53) -> Path:
    def vlq(v: int) -> bytes:
        out = bytearray([v & 0x7F]); v >>= 7
        while v:
            out.append((v & 0x7F) | 0x80); v >>= 7
        return bytes(reversed(out))
    tps = TICKS * bpm / 60.0
    ev, t = [], 0.0
    for pitch, length in notes:
        s = round(t * tps)
        ev.append((s, 1, pitch, 96))
        ev.append((s + max(1, round(length * tps)), 0, pitch, 0))
        t += length
    ev.sort(key=lambda e: (e[0], e[1]))
    body = bytearray()
    body += vlq(0) + b"\xff\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big")
    body += vlq(0) + bytes([0xC0, program])
    prev = 0
    for tick, kind, pitch, vel in ev:
        body += vlq(tick - prev); prev = tick
        body += bytes([(0x90 if kind else 0x80), pitch, vel])
    body += vlq(0) + b"\xff\x2f\x00"
    head = (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + TICKS.to_bytes(2, "big")
    dest.write_bytes(b"MThd" + len(head).to_bytes(4, "big") + head
                     + b"MTrk" + len(body).to_bytes(4, "big") + bytes(body))
    return dest


def render_midi(midi: Path, dest: Path) -> Path:
    sf2 = next((p for p in ("/usr/share/sounds/sf2/FluidR3_GM.sf2",
                            "/usr/share/sounds/sf2/default-GM.sf2") if Path(p).exists()), None)
    if not sf2:
        raise SystemExit("no soundfont")
    subprocess.run(["fluidsynth", "-ni", "-F", str(dest), "-r", str(SR), "-g", "0.8",
                    sf2, str(midi)], check=True, capture_output=True)
    return dest


def timbre_vector(path: Path) -> np.ndarray:
    """A crude voice fingerprint: MFCC means and stds, on the voiced parts."""
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    # Drop near-silence so a long tail does not dominate the statistics.
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    keep = rms > max(1e-4, float(np.median(rms)) * 0.5)
    m = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20, hop_length=512)
    if keep.sum() > 4:
        m = m[:, keep[:m.shape[1]]] if keep.shape[0] >= m.shape[1] else m
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance, so overall level does not masquerade as voice colour."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def submit(data: dict, files: dict | None, dest: Path, timeout: int = 1200) -> float:
    started = time.time()
    r = requests.post(f"{BASE}/release_task", data=data, files=files, timeout=180)
    r.raise_for_status()
    tid = r.json()["data"]["task_id"]
    while time.time() - started < timeout:
        time.sleep(3)
        q = requests.post(f"{BASE}/query_result", json={"task_id_list": [tid]}, timeout=60)
        q.raise_for_status()
        payload = q.json()["data"][0]
        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            result = json.loads(result)
        if isinstance(result, list):
            result = result[0] if result else None
        if not isinstance(result, dict):
            continue
        if result.get("status") == 1:
            url = result["file"]
            got = requests.get(url if url.startswith("http") else BASE + url, timeout=300)
            got.raise_for_status()
            dest.write_bytes(got.content)
            return round(time.time() - started, 1)
        if result.get("status") == 2:
            raise RuntimeError(str(result)[:200])
    raise RuntimeError("timed out")


def cover_sing(src: Path, dest: Path, caption: str, seed: int,
               reference: Path | None = None) -> float:
    data = {
        "task_type": "cover", "prompt": caption, "lyrics": LYRICS,
        "vocal_language": "en",
        "cover_noise_strength": "0.35", "audio_cover_strength": "1.0",
        "inference_steps": "50", "guidance_scale": "7.0",
        "use_random_seed": "false", "seed": str(seed),
        "audio_format": "wav", "thinking": "false",
    }
    files = {}
    with open(src, "rb") as fh:
        files["src_audio"] = (src.name, fh.read(), "audio/wav")
    if reference is not None:
        with open(reference, "rb") as fh:
            files["ref_audio"] = (reference.name, fh.read(), "audio/wav")
    return submit(data, files, dest)


def make_reference(dest: Path, caption: str, seed: int) -> float:
    """A 30 s vocal clip generated by the model itself.

    Self-contained and rights-clean: it is the model's own output, so nobody's
    voice is being used without consent to test whether voices can be held
    steady at all.
    """
    return submit({
        "task_type": "text2music", "prompt": caption,
        "lyrics": "[verse]\n" + LYRICS + "\n" + LYRICS,
        "vocal_language": "en", "audio_duration": "30",
        "inference_steps": "50", "guidance_scale": "7.0",
        "use_random_seed": "false", "seed": str(seed),
        "audio_format": "wav", "thinking": "false",
    }, None, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/gate/voice")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    midi = write_smf(MELODY, out / "vocal.mid", BPM)
    dry = render_midi(midi, out / "00_melody.wav")
    print(f"melody carrier: {dry.name}\n")

    seeds = [11, 22, 33]
    vectors: dict[str, list[np.ndarray]] = {}

    # --- A: caption tags only -------------------------------------------
    print("A. caption tags only")
    for name, caption in PRESETS.items():
        vectors[f"A:{name}"] = []
        for seed in seeds:
            dest = out / f"A_{name}_s{seed}.wav"
            secs = cover_sing(dry, dest, caption, seed)
            vectors[f"A:{name}"].append(timbre_vector(dest))
            print(f"   {name:14} seed {seed:3}  {secs:5.1f}s  {dest.name}")

    # --- B: caption + a reference timbre clip ---------------------------
    print("\nB. caption + 30s reference_audio")
    refs = {}
    for name, caption in PRESETS.items():
        ref = out / f"ref_{name}.wav"
        secs = make_reference(ref, caption, seed=777)
        refs[name] = ref
        print(f"   built reference {ref.name} ({secs:.1f}s)")
    for name, caption in PRESETS.items():
        vectors[f"B:{name}"] = []
        for seed in seeds:
            dest = out / f"B_{name}_s{seed}.wav"
            secs = cover_sing(dry, dest, caption, seed, reference=refs[name])
            vectors[f"B:{name}"].append(timbre_vector(dest))
            print(f"   {name:14} seed {seed:3}  {secs:5.1f}s  {dest.name}")

    # --- verdict ---------------------------------------------------------
    print("\n" + "=" * 66)
    print(f"{'mechanism':12} {'within (same preset)':>22} {'between (different)':>22}")
    print("-" * 66)
    verdict = {}
    for mech in ("A", "B"):
        keys = [k for k in vectors if k.startswith(mech + ":")]
        within = []
        for k in keys:
            within += [distance(a, b) for a, b in itertools.combinations(vectors[k], 2)]
        between = []
        for ka, kb in itertools.combinations(keys, 2):
            between += [distance(a, b) for a in vectors[ka] for b in vectors[kb]]
        w, b = float(np.mean(within)), float(np.mean(between))
        verdict[mech] = {"within": w, "between": b, "ratio": b / w if w else 0.0}
        print(f"{mech:12} {w:22.4f} {b:22.4f}    ratio {b / w if w else 0:.2f}x")

    (out / "results.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("\nA mechanism selects a voice only if `between` is clearly larger than")
    print("`within` -- i.e. two seeds of one preset sound more alike than two")
    print("different presets do. Ratio near 1.0 means the preset selects nothing.")


if __name__ == "__main__":
    sys.exit(main())
