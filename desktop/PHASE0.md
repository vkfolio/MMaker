# Phase 0 — verdict

Phase 0 asked two questions. Both now have measured answers.

## 1. Does the UX premise survive the backend's latency?

**Yes — but the measurement has a caveat that must be resolved before it is
trusted at the top tier.**

`research/latency_probe.py` timed every tool the desktop app will offer, against
the live RunPod engine, at all three quality tiers. Every operation landed in the
**interactive** band (<10 s):

| operation              | seconds | band        |
|------------------------|---------|-------------|
| generate 1 variation   | 3.0–3.1 | interactive |
| split into stems       | 3.0–6.0 | interactive |
| **repaint a 4s region**| 3.4–6.5 | interactive |
| add a layer (strings)  | 6.1–6.7 | interactive |
| another take of a stem | 5.7–5.9 | interactive |
| extend by 8 bars       | 3.4–5.6 | interactive |
| export + download      | 5.4–5.8 | interactive |

The selection-driven loop — *drag a range, hit a tool, hear the result* — is the
product thesis, and its worst case is **6.5 s**. That is slower than a local
plugin and faster than a coffee. It supports building the app as **an instrument
with an async edge**, not as a render queue with a timeline bolted on.

This overturns the plan's stated risk #1, which assumed minutes-long renders and
warned the product might have to become "a timeline with an asynchronous batch
renderer" — a different, lesser product. It doesn't. Build the direct loop.

**What still follows even at 6 s:** every tool needs visible progress and a
working cancel. Six seconds with no feedback still feels broken.

### The caveat, resolved

All three tiers originally returned near-identical timings, and I recorded that
these numbers were "a floor, honest for `fast`, unproven for `ultra`". That
caveat is now closed, and the reason was worse than a measurement artefact: all
three tiers were rendering on the same model.

`ACE-Step/Ace-Step1.5` ships only `acestep-v15-turbo`. The XL checkpoints are
separate ~20 GB repos. And downloading them is not sufficient either -- ACE-Step's
model registry is fixed at startup and does not scan the checkpoint directory, so
a request naming an unregistered model gets

    Model 'acestep-v15-xl-turbo' not found in ['acestep-v15-turbo'],
    using primary: acestep-v15-turbo

and renders at turbo quality **without failing**. Nothing in the result says so.
Both checkpoints are now fetched and registered as model slots (see
`server/start.sh` and `tools/fetch-quality-weights.sh`), and the log confirms
each tier reaching its own model.

Re-measured on the pod's RTX A6000, warm, 8 bars:

| tier | model | steps | seconds |
|------|-------|-------|---------|
| fast  | `acestep-v15-turbo` 0.6B    | 8  | ~4.5 |
| high  | `acestep-v15-xl-turbo` 4B   | 8  | ~5.1 |
| ultra | `acestep-v15-xl-base` 4B    | 50 + CFG 7.0 | ~10.4 |

At full length, `ultra` renders 32 bars in 23.9 s and **64 bars -- about two
minutes of music -- in 41.4 s**, at 48 kHz stereo and sensible levels
(-1.6 dBFS peak, -14 dBFS RMS).

**This strengthens the verdict rather than weakening it.** The selection-driven
loop stays interactive even on the largest model: a regional repaint operates on
a few seconds of audio, not a whole song. Two things do follow:

- Each model costs **49-81 s to load, once**, on first use after a restart. That
  is a real cold-start the UI must not present as a normal render.
- All three loaded together sit at **40 GB of 48 GB VRAM**. There is headroom for
  two-minute renders but not much else; batch sizes above 1 or a fourth model
  would not fit. Dropping the `fast` slot frees ~10 GB if that is ever needed.

## 2. Are the three technical seams alive?

`src/main.cpp` is a self-driving spike: it starts audio, dispatches a worker, and
animates zoom without anyone touching it, printing a machine-readable `SEAM` line
every second. Release build, 10 s run:

```
SEAM audio_ok=1 backend=WASAPI heard=9.73s latency_ms=30.0 peak=0.250
     | worker_started=1 worker_landed=1 woke=1
     | columns=10000 paint_ms=0.15 worst_ms=0.48 frames=602 ticks=599
```

- **Audio** — miniaudio → WASAPI, playing, no xruns. The playhead is
  latency-compensated (30.0 ms of device latency subtracted), which is the defect
  the plan calls the most-noticed sync problem in a DAW.
- **Async** — a worker posted back through the `UiMessage` queue and
  `platform->wake()` pulled the UI out of an idle block. `woke=1` is that seam
  proving itself, not an assertion.
- **Render** — 10 000 columns / 20 000 points, **zoom rebuilt every single
  frame** (the true wheel-zoom cost, not a static draw): **60 fps locked**
  (602 frames in 10.0 s), 0.15 ms typical, 0.48 ms worst. Against a 16.6 ms
  budget that is ~35× headroom.

### One finding worth carrying forward

The **Debug** build runs the same scene at **10 fps** — `ticks` tracks `frames`
1:1, so the loss is not in paint (0.55 ms) but in the un-instrumented rest of the
frame: Yoga layout and the full scene rebuild that `Scene::clear()` forces every
frame. Release erases it entirely.

Consequences:
- **Never judge performance in Debug**, and never let a teammate do so either.
- The plan's concern that `Scene::clear()` + `paint_skia` re-execution would
  eventually demand an `SkPicture`/tile cache is **real but not yet binding**.
  Debug is the early warning: the per-frame rebuild is genuinely expensive, it is
  simply cheap enough optimised. Profile again at 40 tracks, and on Intel
  integrated graphics.

## Status

Both Phase 0 gates pass. The premise survives, and the seams hold with room to
spare. Phase 0.5 (the server fixes) is already done — non-destructive split,
second-precision ranges, cooperative cancel, idempotency keys, honest progress.

Next is **Phase 1 — transport and timeline**, which the plan marks the framework
gate at 10–14 weeks. That is a large enough commitment to confirm before starting.
