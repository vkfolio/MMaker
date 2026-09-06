# Phase 0 — measured, on ACE-Step 1.5 XL

Measured on a **RunPod RTX 6000 Ada Generation (48 GB)**, US-WA-1, with a 100 GB
network volume. Checkpoint `acestep-v15-xl-sft` (4B DiT) plus `acestep-5Hz-lm-1.7B`.

## Gate 1 — can ACE-Step XL be driven by a melody? **Yes.**

ACE-Step has no note input, so the melody has to arrive as audio: FluidSynth
renders the MIDI, `cover` re-timbres it. The question was whether the tune
survives that. `gate/cover_fidelity.py` sweeps both cover knobs and measures
rendered pitch against the source MIDI with `librosa.pyin`.

Melody: Twinkle Twinkle (C C G G A A G), 4.0 s, caption *"solo violin,
expressive vibrato, rosin and bow noise, concert hall"*, 50 steps, fixed seed.

```
                                pitch_err  within.5  tracked  timbreΔ   secs
sampler reference (the floor)        0.00      100%      7/7        -      -
cover          noise 0.15            0.03      100%      7/7      9.1    6.3
cover          noise 0.35            0.01      100%      7/7     10.8    6.5
cover          noise 0.55            0.00      100%      7/7     10.9    3.0
cover          noise 0.75            0.00      100%      7/7     10.3    3.0
cover          noise 0.90            0.00      100%      7/7     10.6    3.0
cover          noise 0.55, str 0.6   0.00      100%      7/7     10.8    3.0
cover-nofsq    noise 0.15           28.64        0%      6/7     11.0    3.0
cover-nofsq    noise 0.35            5.13       57%      7/7     10.9    3.0
cover-nofsq    noise 0.55            0.01      100%      7/7     12.4    3.0
cover-nofsq    noise 0.75            0.00      100%      7/7     11.2    3.0
cover-nofsq    noise 0.90            0.00      100%      7/7     10.8    3.0
```

`pitch_err` is mean absolute error in semitones over each note's middle 60%.
`timbreΔ` is mean absolute log-mel difference from the sampler render — a proxy
for "the timbre actually changed", not a quality score.

### The finding that decides the design

**`cover` holds the melody at every noise level; `cover-nofsq` does not.**

That is not a quirk, it follows from the mechanism. `cover` replaces the source
latents with **FSQ semantic codes**, and those codes encode melody, rhythm and
harmony explicitly — so the tune is carried by the conditioning regardless of how
much the waveform is re-noised. `cover-nofsq` feeds the raw continuous VAE
latents instead, so melody survives only while the waveform itself does, and it
collapses below `cover_noise_strength` 0.55 (28.6 semitones of error at 0.15 —
i.e. unrecognisable).

**So: use `cover`, and treat `cover_noise_strength` as a timbre-freedom dial
rather than a melody-safety dial.** Lower values give the model more room to
reinterpret while the codes keep the notes. That is the opposite of the intuition
their UI copy suggests ("Melody Retention"), and it is only true because the FSQ
codes are doing the retaining.

`audio_cover_strength=0.6` — dropping source conditioning after 60% of steps —
also kept the melody perfectly, so there is headroom there too.

### Numbers

| | |
|---|---|
| checkpoint | `acestep-v15-xl-sft`, 4B DiT, 19 GB on disk |
| VRAM | **16.9 GB of 48 GB** with DiT + 1.7B LM both resident |
| render | **3.0 s for 4.0 s of audio** at 50 steps, warm |
| first render | 6.3 s, including lazy model load |
| output | 48 kHz stereo, padded 4.00 s → 5.12 s |
| GPU tier | ACE-Step self-reports "unlimited": 600 s max duration, batch 8 |

16.9 GB means the 48 GB card is not the constraint — an `xl-base` could be loaded
alongside for `lego`/`extract`/`complete`, which `xl-sft` cannot do.

### What this does not tell us

**Nobody has listened yet.** `timbreΔ` says the output differs from the soundfont
by 9–12 dB of log-mel distance; it does not say it sounds like a violin. Pitch
accuracy is objective and passes cleanly; timbre quality is a listening judgement
and is still open. Files are in `.work/gate1/`.

## Gate 4 — do singer presets actually select a singer? **Yes, decisively.**

The question gates 2 and 3 both failed. A preset selects a voice only if two
seeds of one preset sound more alike than two different presets do:

| mechanism | within | between | ratio |
|---|---|---|---|
| caption tags, on `cover` | 0.0075 | 0.0051 | **0.68x** |
| caption + `reference_audio`, on `cover` | 0.0096 | 0.0074 | 0.77x |
| caption on a sung source (re-voicing) | 0.0068 | 0.0041 | 0.60x |
| **SoulX reference clip** | **0.0037** | **0.0362** | **9.78x** |

Everything ACE-Step offers scored *below* 1.0 — different presets came out no
further apart than two seeds of the same one, meaning the preset selected
nothing at all. SoulX's reference clip scores **9.78x**, because it copies a
timbre from audio rather than describing one in words.

**So the voice knob is the reference clip, and nothing else.** Captions do not
steer vocal identity on either engine, on any source. `revoice` stays an
off-by-default "another take" button and must never be labelled as a voice
control.

### Building a preset without SoulX's preprocessing stack

Their official route wants torch 2.10, numpy 2.x, NeMo and funasr — about 10 GB,
all conflicting with the two environments this pod already runs. The cheaper
route works: **the user says what was sung**, notes come from `pyin`
(`app/transcribe.py`) and phonemes from the G2P already here (`app/g2p.py`), and
`app/voicebuild.py` assembles SoulX's prompt metadata from those. No third
environment, no extra weights.

Three starter presets — Aria, Juno, Wren — are generated by ACE-Step
`text2music`, so they are model output and no person's voice is used to prove
that voice selection works. 12–15 s to generate each; SoulX then sings at
**1.6–1.7 s per phrase** once warm.

## Gate 2 — singer identity (superseded by gate 4)

Not yet run. Order is cheapest-first, stopping at whatever works:

1. fixed seed + vocal caption tags (`female vocal, breathy`) and inline lyric tags
2. a 30 s `reference_audio` timbre clip — documented as global, and as *removing*
   melody and rhythm, which is exactly right when the melody comes from `src_audio`
3. a trained LoRA — last, because it trains only the DiT decoder's four attention
   projections with the timbre and lyric encoders frozen, and is one-at-a-time

## Operational notes

- **RunPod's nginx squats on 8001**, 8081, 7861, 9091 and 3001. ACE-Step runs on
  **8011** here. `musicmaker/server/start.sh` hits the same thing.
- **`uv sync` fails on the network volume** — "Stale file handle (os error 116)"
  from MooseFS during hardlinking. Code and venv live on the container disk at
  `/opt/ACE-Step-1.5`; only `checkpoints/` is symlinked to the volume. Set
  `UV_LINK_MODE=copy`.
- `/query_result` nests twice: `data` is a list of envelopes, and each envelope's
  `result` is a **JSON string containing a list**. The status that matters is the
  inner one.
- Changing env in a supervisor script needs the **supervisor** restarted, not just
  the child.
