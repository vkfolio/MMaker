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

### The caveat

All three tiers returned near-identical timings. Either the tiers are genuinely
close, or **`ultra` is silently falling back to the turbo weights** because the
XL checkpoints were never fetched. `/api/engine` was added to make this
checkable. Until it is checked, treat these numbers as **a floor, honest for
`fast`, unproven for `ultra`**. If `ultra` really runs 4B at 50 steps, its true
latency is higher and the band assignment for the top tier may change.

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
