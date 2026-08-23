# musicX AI sidecar

A Node process spawned hidden by the desktop app (`desktop/src/ai/sidecar.cpp`),
bridging the Inspire Me panel to Claude over ndjson stdio. One JSON object per
line in, one per line out.

It turns a plain-language idea into an ACE-Step caption, and into lyrics when
the song has vocals.

## Why

ACE-Step's own documentation is blunt about the problem: *"As ordinary people,
our music vocabulary is impoverished"*, and it recommends having an LLM rewrite
your caption. The model responds to nine dimensions — genre, emotion,
instruments, timbre, era, production, vocals, tempo, structure — and a caption
naming several of them anchors the result far better than a sentence naming
one. Turning "something sad for a rainy evening" into that is a language
problem, not a music problem.

## Prerequisites

- Node.js 18+ on PATH, or `MUSICX_NODE` pointing at a node.exe
- One-time: `npm install` in this directory
- An Anthropic key, entered in the app (Settings) or via `ANTHROPIC_API_KEY`.
  Neither is required if this machine already has a Claude Code login — the SDK
  will use it.

## Protocol

```
in : {"type":"start","api_key":"…","model":"claude-sonnet-5"}
     {"type":"compose","id":"c1","mode":"idea"|"lyrics","text":"…",
      "instrumental":false,"advanced":{"bpm":124,"key_scale":"C Major","bars":32}}
     {"type":"refine","id":"c2","text":"make the chorus shorter"}
     {"type":"cancel"}
out: {"type":"ready","model":"…"}
     {"type":"status","id":"c1","text":"thinking"}
     {"type":"result","id":"c1","title":"…","caption":"…","lyrics":"…",
      "bpm":68,"key_scale":"D Minor","notes":"…"}
     {"type":"error","id":"c1","message":"…"}
```

`refine` continues the same session, so "make the chorus shorter" lands against
what was just proposed. Every reply is a whole proposal rather than a patch: the
app shows it for editing before anything is generated, and a partial update
would be harder to review than a complete one.

## Debugging

Sidecar log: stderr, which the app redirects to `%TEMP%\musicx-sidecar.log`.

Standalone smoke test — note stdin must stay open, or the process exits before
the reply arrives:

```
(printf '%s\n' '{"type":"start"}' \
  '{"type":"compose","id":"c1","mode":"idea","text":"lofi beat for studying","instrumental":true}'; \
 sleep 90) | node sidecar.mjs
```
