# musicX Studio (desktop)

A Windows desktop timeline on `vikui_cpp`, with the RunPod engine as its
backend. This is the Phase 1 slice: **local audio on a real timeline, playing in
sync, editable while it rolls.** No server connection yet.

## Build

```
cmake -S . -B build
cmake --build build --config Release --target musicx
```

Needs the `vikui_cpp` checkout (`-DVIKUI_DIR=` to point elsewhere).
**Build Release to judge anything about speed** — see the note at the bottom.

## Run

```
build\Release\musicx.exe                      # synthesised demo, always works
build\Release\musicx.exe D:\path\to\stems     # every .wav/.mp3/.flac in a folder
build\Release\musicx.exe <folder> --selftest  # plays, reports, exits
build\Release\musicx.exe <folder> --render out.png   # draws a PNG, no window
```

Point it at a folder of exported stems. Filenames sharing a render-id prefix get
that prefix stripped, so tracks read `vocals` and `drums` rather than
`283a6411-cef6-...-vocals`.

### What you can do

| | |
|---|---|
| Space, or the transport | play / stop |
| Click the ruler | seek; drag to scrub |
| Drag a clip | move it — **including while playing** |
| Ctrl + wheel | zoom about the cursor |
| Wheel | scroll |
| `L` | loop |
| M / S on a track | mute / solo |

Every run prints a heartbeat line each second: device, decode times, clip count,
playhead, xruns, and paint cost split into our arithmetic versus Skia's. A GUI
that can only be diagnosed by looking at it cannot be diagnosed over a terminal,
which is where this actually gets debugged.

## The gate

```
build\Release\mixer_test.exe
```

The plan is explicit that what kills a DAW is not draw throughput but "the click
when you drag [a clip] during playback", and that an xrun counter does not catch
it. So the mixer has no UI dependency and is exercised offline through the events
that actually click — start, stop, seek, loop wrap, mute, a fader slammed, and a
60-step clip drag — with the output checked for sample-to-sample steps.

It found three real defects on first run, one of which was a design claim of mine
that was wrong: a position-derived read cursor does **not** make editing during
playback click-free. It stops a cursor being *reset*; it does nothing about the
content jumping, and moving a clip changes which source sample lands at time *t*.
Seek, loop wrap and clip moves are all that same defect, and share one repair —
hold the last output value and decay it out *while* the new material fades in.
Both halves are needed; doing only the first relocates the step rather than
removing it.

Every case now comes out limited by the source material: worst step 0.0144,
which is exactly the test sine's own slope.

## Measured

Real song, 11 MP3 stems, Release:

```
device=WASAPI rate=48000 latency_ms=30.0 | sources=11 decode_ms=339
tracks=11 clips=11 | xruns=0 | paint_ms=3.7 (build 0.10 skia 3.5) | 60 fps
```

- **Playhead is latency-compensated** — the render position minus the device's
  30 ms, measured rather than assumed. Drawing the render position puts the line
  visibly ahead of the sound, which is the sync error people notice first.
- **Paint is fill-rate bound, not overhead bound.** Collapsing all 11
  `drawPoints` calls into one changed nothing (3.3–3.6 ms vs 2.9–3.3 ms), so the
  cost is per-*point*. Grouping draws by paint type — every fill, then every
  waveform, then every border — still won ~30%, because interleaving antialiased
  round-rects with non-AA point batches makes Skia flush at each switch.
- **Debug runs the same scene at 10 fps** where Release does 60, and `ticks`
  tracks `frames` 1:1, so the loss is not paint (0.55 ms) but Yoga layout plus
  the rebuild `Scene::clear()` forces every frame. Never judge this framework's
  performance in a Debug build.

At 11 tracks there is comfortable headroom. The plan's stress target is 40
tracks; that is where the `SkPicture`/tile cache it warns about becomes real, and
the numbers above are the baseline to profile against.

## Not here yet

No server connection, undo, trimming, fades UI, recording, or `.mmproj`. Those
are Phases 2–4 and are absent rather than half-present.
