# Phase 0 — Prove the premise before building the app

These scripts answer one question: **can we build a song layer by layer with
ACE-Step 1.5 and have the layers stay in sync and stay editable?**

No app code exists yet and none should until this passes.

## Setup

On a RunPod pod with ACE-Step 1.5 running its REST server:

```bash
pip install -r research/requirements.txt
export ACESTEP_URL=http://<runpod-host>:8001
```

## Run in order

| # | Command | Answers |
|---|---------|---------|
| 1 | `python research/probe.py` | Is the pod up? What's the real sample rate, cold-start time, and API schema? |
| 2 | `python research/lego_chain.py` | Does drums → bass → keys → vocals sound musical? Does `lego` return a solo stem or a full mix? |
| 3 | `python research/sync_check.py research/out/lego_chain_<id>` | How many milliseconds do the layers drift? |
| 4 | `python research/repaint_region.py research/out/lego_chain_<id> 02_bass.wav 9 13` | Does regenerating bars 9–12 leave the rest untouched? |

Run them from the `research/` directory, or add it to `PYTHONPATH` — the
scripts import each other as flat modules.

## Gates

- **lego_chain** — listen to `99_final_mix.wav`. If it isn't musical, stop.
- **sync_check** — worst median deviation under 20 ms passes. 20–50 ms means
  the backend needs a nudge/quantize step. Over 50 ms makes drift correction a
  first-class problem rather than polish.
- **repaint_region** — residual outside the mask must be near zero. If it
  isn't, every edit silently rewrites the whole layer and the version model in
  the plan has to change.

The bar range is the last two arguments and defaults to 17–25. That default
only fits a longer render — `lego_chain.py` makes 30s clips, which at 96 BPM is
12 bars, so pass `9 13` (or raise `audio_duration` in the grid).

## Known unknowns

`acestep.py` is deliberately tolerant about response field names because the
public docs don't pin them down. Every raw response lands in `out/_raw/`.
After `probe.py` runs once against a real pod, read those payloads and tighten
`_find_task_id` / `_find_audio_ref` to the actual schema.
