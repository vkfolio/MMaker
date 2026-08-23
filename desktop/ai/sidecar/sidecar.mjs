// musicX AI sidecar: plain language in, an ACE-Step caption out.
//
// Spawned hidden by the desktop app (desktop/src/ai/sidecar.cpp) and spoken to
// in ndjson over stdio: one JSON object per line in, one per line out.
//
// Why this exists. ACE-Step's own documentation is blunt about it: "As ordinary
// people, our music vocabulary is impoverished", and it recommends having an
// LLM rewrite your caption. The model responds to nine dimensions -- genre,
// emotion, instruments, timbre, era, production, vocals, tempo, structure --
// and a caption naming several of them anchors the result far better than a
// sentence naming one. Turning "something sad for a rainy evening" into that
// is a language problem, not a music problem, which is what this is for.
//
// Protocol
//   in : {type:"start", api_key?, model?}
//        {type:"compose", id, mode:"idea"|"lyrics", text, instrumental, advanced?}
//        {type:"refine",  id, text}
//        {type:"cancel"}
//   out: {type:"ready"} | {type:"result", id, caption, lyrics, bpm, key_scale,
//        title, notes} | {type:"error", id?, message} | {type:"status", id, text}
//
// Every reply is a full proposal, never a patch: the app shows it for editing
// before anything is generated, so a partial update would be a worse thing to
// review than a whole one.

import readline from "node:readline";

const state = {
  model: "claude-sonnet-5",
  apiKey: "",
  sessionId: null,
  busy: false,
};

let sdk = null;
async function ensureSdk() {
  if (!sdk) sdk = await import("@anthropic-ai/claude-agent-sdk");
  return sdk;
}

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}
function log(...a) {
  process.stderr.write("[sidecar] " + a.join(" ") + "\n");
}

// --- the domain knowledge -------------------------------------------------

const SYSTEM = `You turn a musician's rough idea into a caption for ACE-Step, a
text-to-music model, and into song lyrics when the song has vocals.

ACE-Step responds to these dimensions. A caption naming several of them anchors
the result; one naming a single dimension leaves the model to invent the rest:

  Style/Genre       pop, lo-fi, synthwave, bossa nova, drum and bass
  Emotion           melancholic, uplifting, dreamy, menacing, nostalgic
  Instruments       rhodes, 808 drums, nylon guitar, string section, analog bass
  Timbre            warm, crisp, airy, punchy, lush, raw, saturated
  Era               80s synth-pop, 90s boom bap, 60s soul, modern trap
  Production        lo-fi, studio-polished, live room, bedroom pop, wall of sound
  Vocals            female vocal, breathy, falsetto, choir, spoken word
  Tempo/Rhythm      slow, mid-tempo, driving, halftime, swung, 96 BPM
  Structure         building intro, big chorus, sparse bridge, fade-out

Rules that come from the model's own behaviour:
- Specific beats vague. "sad piano ballad, female breathy vocal, sparse" beats
  "a sad song".
- Combine dimensions. Aim for six to ten comma-separated tags.
- References work well: "in the style of 80s synthwave", "reminiscent of Bon Iver".
- Never combine conflicting styles. If the user asks for two that fight, either
  pick the dominant one or express it as a progression in the Structure tag --
  "starts as sparse folk, opens into full band" -- rather than fusing them.
- Do not pad. Every tag should change what the model does.
- Write the caption as comma-separated tags in the order above. No sentences.

Lyrics, when the song is not instrumental:
- Use ACE-Step's section tags on their own lines: [verse] [chorus] [bridge]
  [intro] [outro].
- Keep them singable: short lines, plain words, a repeated chorus.
- Match the language the user wrote in unless they asked otherwise.
- Do not write lyrics for an instrumental. Return an empty string.

If the user gives you lyrics instead of an idea, infer the caption from them --
their subject, mood and register -- and return those lyrics unchanged unless
asked to change them.

Estimate bpm and key from the style you chose. Be conventional: a lo-fi beat is
70-90, house is 120-128, a ballad is 60-80.

Reply with a single JSON object and nothing else. No prose, no code fence:
{"title": "...", "caption": "...", "lyrics": "...", "bpm": 96,
 "key_scale": "A Minor", "notes": "one short sentence on what you chose and why"}`;

// --- session --------------------------------------------------------------
//
// One long-lived query() so "make the chorus shorter" lands against what was
// just proposed. The prompt stream stays open between turns; a finished query
// cannot be added to.

let claudeQuery = null;
let userQueue = [];
let wake = null;

async function* promptStream() {
  for (;;) {
    if (userQueue.length === 0) {
      await new Promise((r) => (wake = r));
      continue;
    }
    yield userQueue.shift();
  }
}

async function startClaude(resumeId) {
  const { query } = await ensureSdk();
  if (state.apiKey) process.env.ANTHROPIC_API_KEY = state.apiKey;
  if (!process.env.ANTHROPIC_API_KEY)
    log("no ANTHROPIC_API_KEY -- relying on an existing Claude Code login, if any");

  userQueue = [];
  claudeQuery = query({
    prompt: promptStream(),
    options: {
      model: state.model,
      systemPrompt: SYSTEM,
      // This writes words, nothing else. Every tool is off: there is no file
      // to read and no command to run that could make a caption better, and a
      // sidecar that can touch the disk is a sidecar that has to be trusted.
      allowedTools: [],
      disallowedTools: [
        "Bash", "Write", "Edit", "Read", "Glob", "Grep", "WebFetch", "WebSearch",
        "NotebookEdit", "TodoWrite", "Task", "KillShell", "BashOutput",
      ],
      settingSources: [],
      maxTurns: 4,
      ...(resumeId ? { resume: resumeId } : {}),
    },
  });
  consume(claudeQuery);
}

let pending = null;   // id of the request in flight

async function consume(q) {
  let text = "";
  try {
    for await (const m of q) {
      if (m.type === "system" && m.subtype === "init") {
        state.sessionId = m.session_id;
        continue;
      }
      if (m.type === "assistant") {
        for (const block of m.message?.content ?? [])
          if (block.type === "text") text += block.text;
        continue;
      }
      if (m.type === "result") {
        state.sessionId = m.session_id ?? state.sessionId;
        deliver(text.trim());
        text = "";
        state.busy = false;
      }
    }
  } catch (err) {
    log("query failed:", err?.message ?? String(err));
    send({ type: "error", id: pending, message: String(err?.message ?? err) });
    state.busy = false;
    claudeQuery = null;   // next request restarts, resuming the session
  }
}

/// Pull the JSON object out of the reply.
///
/// Asking for "JSON and nothing else" is a request, not a guarantee, and a
/// stray sentence before the brace would otherwise throw away a good answer.
function parseProposal(text) {
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    return JSON.parse(text.slice(start, end + 1));
  } catch {
    return null;
  }
}

function deliver(text) {
  const p = parseProposal(text);
  if (!p) {
    send({ type: "error", id: pending, message: "the model did not return a proposal" });
    return;
  }
  send({
    type: "result",
    id: pending,
    title: String(p.title ?? "").slice(0, 120),
    caption: String(p.caption ?? "").slice(0, 1000),
    lyrics: String(p.lyrics ?? "").slice(0, 8000),
    bpm: Number.isFinite(p.bpm) ? Math.round(p.bpm) : 0,
    key_scale: String(p.key_scale ?? ""),
    notes: String(p.notes ?? "").slice(0, 400),
  });
}

function ask(id, body) {
  pending = id;
  state.busy = true;
  userQueue.push({
    type: "user",
    message: { role: "user", content: [{ type: "text", text: body }] },
    parent_tool_use_id: null,
    session_id: state.sessionId ?? "",
  });
  if (wake) { const w = wake; wake = null; w(); }
}

function composeBody(msg) {
  const advanced = msg.advanced ?? {};
  const lines = [];
  if (msg.mode === "lyrics") {
    lines.push("Here are my lyrics. Infer the caption from them and return them unchanged:");
    lines.push("");
    lines.push(String(msg.text ?? ""));
  } else {
    lines.push("My idea: " + String(msg.text ?? ""));
  }
  lines.push("");
  lines.push(msg.instrumental
    ? "This is an INSTRUMENTAL. Return an empty lyrics string."
    : "This has vocals. Write lyrics with [verse]/[chorus] section tags.");
  // Only what the user actually set. Sending "bpm: 0" would read as an
  // instruction to write something at zero beats per minute.
  if (advanced.bpm) lines.push(`Use ${advanced.bpm} BPM.`);
  if (advanced.key_scale) lines.push(`Use the key ${advanced.key_scale}.`);
  if (advanced.bars) lines.push(`The piece is about ${advanced.bars} bars long.`);
  return lines.join("\n");
}

// --- stdio loop -----------------------------------------------------------

const rl = readline.createInterface({ input: process.stdin });

rl.on("line", async (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let msg;
  try {
    msg = JSON.parse(trimmed);
  } catch {
    return;
  }

  try {
    if (msg.type === "start") {
      state.model = msg.model || state.model;
      state.apiKey = msg.api_key || state.apiKey;
      send({ type: "ready", model: state.model });
      return;
    }

    if (msg.type === "compose" || msg.type === "refine") {
      if (state.busy) {
        send({ type: "error", id: msg.id, message: "still working on the previous request" });
        return;
      }
      if (!claudeQuery) await startClaude(state.sessionId);
      send({ type: "status", id: msg.id, text: "thinking" });
      ask(msg.id, msg.type === "compose" ? composeBody(msg) : String(msg.text ?? ""));
      return;
    }

    if (msg.type === "cancel") {
      try { claudeQuery?.interrupt?.(); } catch { /* already gone */ }
      state.busy = false;
      return;
    }
  } catch (err) {
    log("handler failed:", err?.message ?? String(err));
    send({ type: "error", id: msg?.id, message: String(err?.message ?? err) });
    state.busy = false;
  }
});

rl.on("close", () => process.exit(0));
log("started");
