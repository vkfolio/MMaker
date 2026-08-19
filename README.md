# musicmaker — an ACE Studio–style engine, headless, on your own GPU

**What this is:** a single self-hosted API that takes a rough idea (MIDI, a hummed or
sung recording, or just a prompt), gives you variations in a chosen style, splits the one
you like into stems, and then lets you edit, extend, improvise on, and re-voice each stem
independently — everything staying in sync.

**What this is not:** a DAW. No timeline, no mixer, no effects. Stems export to FL Studio
and you finish there. That deliberate omission is what makes this buildable.

---

## The pipeline

```
  idea            variations          pick one         per-stem work         export
  ────            ──────────          ────────         ─────────────         ──────
  MIDI file  ─┐                                    ┌─ edit region
  voice rec  ─┼─→  N renders    ─→   split to  ─→  ├─ extend            ─→  stems.zip
  prompt     ─┤    (same grid,       stems         ├─ improvise/vary        → FL Studio
  lyrics     ─┤     N seeds)                       ├─ change voice
  style      ─┘                                    └─ isolate vocals
```

## Component map

| Pipeline step | How it's done | Confidence |
|---|---|---|
| MIDI input | FluidSynth renders MIDI → audio, fed to ACE-Step as `src_audio` | Solid — ACE-Step has **no** native MIDI input (roadmap for 2.0) |
| Voice/hum input | ACE-Step `cover` with your recording as `src_audio` | Documented |
| Style presets | Prompt template library — our code, not a model feature | Ours |
| Variations | Same bpm/key/duration, N different seeds | Trivial |
| Split to stems | Demucs `htdemucs_ft` (deterministic) or ACE-Step `extract` | Demucs is proven (~9.19 dB vocal SDR) |
| Edit a region | ACE-Step `repaint` with `repainting_start/end` | **Unverified — Phase 0 gate** |
| Extend | ACE-Step `complete` | Documented |
| Improvise / vary | `repaint` with a new seed, or per-stem `cover` | Weakest link — see Known Risks |
| Isolate vocals | Demucs `htdemucs_ft` | Proven |
| Lyrics | ACE-Step `lyrics` + `vocal_language` params | Supported |
| **Change voice** | **Seed-VC / RVC singing voice conversion — see below** | **The gap we have to fill ourselves** |

---

## The one real gap: voice selection

This is the part of the ACE Studio experience that ACE-Step **does not give you**.

ACE Studio's core value is 140+ curated voice models, voice cloning, and VoiceMix timbre
blending. ACE-Step has no voice bank at all — you get whatever timbre the model invents
from your prompt, and you cannot ask for the same singer twice.

**The fix is a conversion layer, not a bigger generator.** Generate the vocal with
ACE-Step, then convert its timbre with zero-shot singing voice conversion:

```
ACE-Step vocal stem ──→ Seed-VC ──→ vocal in the selected voice
                          ▲
                    voice library
                    (reference clips, one per voice)
```

Because [Seed-VC](https://github.com/Plachtaa/seed-vc) is zero-shot, a "voice" is just a
reference clip plus an embedding — no per-voice training. That gets us three things at
once:

- **Predefined voice selection** — curate a library of reference clips, expose as `voice_id`
- **Voice cloning** — user uploads ~10s of singing, becomes a new `voice_id`
- **Consistency** — the same voice across every song, which ACE-Step alone cannot do

Licensing note: every reference clip in the shipped library needs clear rights. Cloning a
real singer without permission is not something to build in as a default.

---

## Architecture

One Docker image, one FastAPI service, one RunPod pod.

```
FastAPI  (single container, GPU)
   ├── ACE-Step 1.5   cover · lego · repaint · complete · extract · text2music
   ├── Demucs         deterministic stem split + vocal isolation
   ├── Seed-VC        voice library, voice change, voice cloning
   ├── FluidSynth     MIDI → audio (the MIDI input path)
   └── librosa        bpm/key detection + sync verification
        │
        ▼
   local disk or S3 — every version is an immutable file
```

**Data model.** A *project* holds a grid (`bpm`, `key`, `time_sig`) and a list of *stems*;
each stem holds a stack of *versions* (immutable audio + the exact params that made it).
Nothing is ever mutated — editing appends a version. Undo is free, and every stem stays
genuinely independent.

### API surface

```
POST /projects                  idea (midi|audio|prompt) + style + lyrics → N variations
GET  /projects/{id}
POST /projects/{id}/split       chosen variation → stems
POST /stems/{id}/repaint        {start_bar, end_bar, prompt} → new version
POST /stems/{id}/extend         {bars} → new version
POST /stems/{id}/vary           {seed} → new version
POST /stems/{id}/voice          {voice_id} → new version (Seed-VC)
GET  /voices                    library
POST /voices                    clone from uploaded clip
GET  /projects/{id}/export      stems zip, tempo-matched, for FL Studio
GET  /jobs/{id}/events          SSE progress — renders take minutes, never block
```

A minimal audition page (play variations, A/B stem versions, hit export) is worth building
eventually, but it is not a DAW and not on the critical path.

---

## Known risks

**1. Layer/edit quality is unproven.** ACE Studio 2.0 ships this exact feature set, and
MusicTech called its "Add a Layer" *"unusable for professional projects"* due to artefacts;
Sound On Sound called the generative output *"fairly generic."* The same team trained the
model we're using. **Phase 0 exists to find out whether we clear that bar** before any API
code is written.

**2. "Improvise" is the weakest verb.** `cover` is style re-performance, not variation —
it won't develop your melody, it re-renders it in different clothes. `repaint` with a new
seed gives alternate takes, not developed ideas. Real melodic development is a symbolic
problem; if it matters, that's a score-native path, not an audio-latent one.

**3. Bad key/BPM detection poisons everything.** A rough hum or sloppy MIDI can be
misdetected, and since every stem locks to that grid, one bad reading corrupts the whole
project. **Always let the user confirm or override detected bpm/key before layering.**

**4. Render latency is a product problem, not just an engineering one.** Both ACE Studio
reviews name it as the flow-killer. Async jobs + SSE are necessary but not sufficient —
the client must let you keep working while renders run.

---

---

## Setup goal

The whole deployment story, start to finish:

```
1. Spin up a RunPod pod from the template (GPU + network volume)
2. It boots, downloads weights once to the volume, reports ready
3. Open the pod URL in a browser  ← the UI is served by the pod itself
4. Make music
```

No local install, no build step, no separate frontend to host. The UI also has a
"backend URL" field, so you *can* point a hosted copy at your pod instead — but the
default path is: open the URL, it works.

Model weights live on the **network volume**, not baked into the image. First boot
downloads them (~20–40 GB) and reports progress on `/health`; every later boot is
instant, and the image stays small enough to pull quickly.

---

## Interface principle

**One screen. No timeline.** Easy by default, powerful when you go looking.

```
┌──────────────────────────────────────────────────────────────┐
│  idea:  [ prompt ▾ ]  [ drop MIDI / audio ]   style: [ ▾ ]   │  ← always visible
│         bpm 96 ▾   key A Minor ▾   32 bars ▾      [ Generate ]│
├──────────────────────────────────────────────────────────────┤
│  variations   ▶ 1    ▶ 2    ▶ 3    ▶ 4       [ split this ]  │
├──────────────────────────────────────────────────────────────┤
│  ▶ drums     ■■■■■■■■■■  solo mute   ⋯ edit extend vary      │  ← stem rack
│  ▶ bass      ■■■■■■■■■■  solo mute   ⋯                       │
│  ▶ vocals    ■■■■■■■■■■  solo mute   ⋯ voice: Aria ▾         │
│                                          [ export stems → FL ]│
└──────────────────────────────────────────────────────────────┘
```

Every stem row expands to reveal bar-range selection, version history, and per-stem
prompt. Nothing modal, nothing blocking — renders run in the background and the row shows
a progress bar while you keep working on other stems.

---

## Phases

Each phase ends in something you can actually use.

### Phase 0 — Prove the quality *(gate — needs a pod)*
Scripts already written in `research/`. Run `probe.py`, `lego_chain.py`, `sync_check.py`,
`repaint_region.py`. **Deliverable:** a yes/no on whether layer and edit quality beats
what ACE Studio already ships. Everything below is wasted effort if this fails.

### Phase 1 — One-command pod
Dockerfile, model bootstrap onto the network volume, FastAPI skeleton with `/health`
reporting download progress, CORS, static SPA mount. RunPod template JSON.
**Deliverable:** spin up a pod, open its URL, see a page that says the engine is ready.
*Buildable now, without a GPU.*

### Phase 2 — Idea → variations
`POST /projects` accepting prompt / MIDI / audio. FluidSynth for the MIDI path. BPM+key
detection **with user confirmation before anything locks**. Style preset library. N seeds
in parallel. UI: idea bar + variation cards you can play and compare.
**Deliverable:** type an idea, get four takes, pick one.

### Phase 3 — Stems + export
Demucs `htdemucs_ft` split. Stem rack UI with solo/mute/gain. Tempo-matched zip export.
**Deliverable:** the full round trip — idea → stems open in FL Studio, on the grid.

### Phase 4 — Stem editing
`repaint` (bar range), `complete` (extend), vary (new seed). Immutable version stack per
stem with A/B compare and revert. Sync verification on every returned stem, auto-nudge
small drift.
**Deliverable:** fix the one bar you don't like without touching anything else.

### Phase 5 — Vocals + voice library
Lyrics editor. `lego` with `track_name="vocals"` over the instrumental. Seed-VC voice
library (curated, rights-cleared reference clips) and cloning from a ~10s upload.
**Deliverable:** your lyrics, sung over your track, in a voice you picked.

### Phase 6 — SFX
[Stable Audio 3](https://stableaudio3.com/stable-audio-3-explained) (May 2026, open
weights, small + medium, trained on licensed data) for one-shots, textures, and risers,
as a separate generator alongside the music path.
**Deliverable:** same app makes sound effects, not just songs.

### Phase 7 — Polish
Project save/load, preset management, batch export, prompt library.

---

## Status

| Phase | State |
|---|---|
| 0 — Prove the quality | Scripts written, math verified offline. **Awaiting a pod.** |
| 1 — One-command pod | Done. Boots, downloads to the volume, reports health, serves the UI. |
| 2 — Idea → variations | Done. Prompt / MIDI / audio in, N takes out, grid confirmation guard. |
| 3 — Stems + export | Done. Split, mix state, tempo-matched zip for FL Studio. |
| 4 — Stem editing | Done. Repaint a bar range, extend, alternate takes, version stack. |
| 5 — Vocals + voices | Done, except the Seed-VC adapter — see Known gaps. |
| 6 — SFX | Done, except the Stable Audio adapter — see Known gaps. |
| 7 — Polish | Done. Projects persist, presets, version history, export. |

Deployment: `server/start.sh` brings up ACE-Step on `127.0.0.1:8001` and the API + UI on
`:8000`; the Dockerfile installs both. Set `MUSICMAKER_API_TOKEN` on any pod with a
public proxy URL — see `RUNPOD.md`.

**29/29 tests pass** against the stub engine, and the full flow was driven in a real
browser: idea → 3 takes → split → drag bars 5–9 → regenerate → new version, plus SFX.

```
research/    Phase 0 gate scripts (probe, lego_chain, sync_check, repaint_region)
server/      the engine — FastAPI, engines/, Dockerfile, model registry, UI
  app/       schemas, storage, jobs, service, analysis, audio, routers
  static/    the single-screen UI
```

## Known gaps

These are honest holes, not oversights. Each fails loudly with a message saying what to do.

1. **Nothing has run against a real model.** Every test uses the stub engine, which
   produces diagnostic tones. Phase 0 on a pod is still the gate.
2. **The Seed-VC adapter is a stub that raises.** The voice library, cloning, storage and
   UI are all built and tested; only `_load()` needs wiring once the checkpoint is
   verified on a pod. The whole voice path runs on the stub today.
3. **The Stable Audio 3 adapter is written but unverified** — the repo id in
   `models.yaml` is a placeholder.
4. **`repo_id` values for Seed-VC and Stable Audio are unverified placeholders.**
   Confirm on Hugging Face before enabling them.
5. **Sync verification needs librosa.** Without it every stem reports `unmeasured`
   rather than a guessed number. The pod image installs it.
6. **`lego` may return a full mix rather than a solo stem.** The stub assumes a solo
   stem; `research/lego_chain.py` measures which it actually is. If it is a full mix,
   `service.add_layer` needs a Demucs pass after generation.

`QUICKSTART.md` runs it locally. `RUNPOD.md` runs it on a pod — the short
version is one line: `bash <(curl -sL .../install.sh)`.
