"""What this pod can actually do.

One entry per tool on the home page. The server owns this list rather than the
page, for two reasons: the page should not claim a capability the engine on
*this* pod cannot serve, and a tool whose gate has not passed should be visibly
absent rather than silently broken.

`state` is the honest one:

  ready       measured, and the numbers are in PHASE0.md
  untested    the mechanism is proven but this particular shape is not
  gated       there is a measurement outstanding, or one that went against it

`signal_in` / `signal_out` drive the colour of the rule on the home page --
notes, audio or voice -- so the page shows at a glance which tools take a score
and which take a recording.
"""

from __future__ import annotations

TOOLS = [
    {
        "id": "play",
        "name": "Play a melody",
        "takes": "a .mid file",
        "gives": "a real instrument",
        "blurb": "Eleven instruments, or describe your own. Keeps your timing and pitch exactly.",
        "signal_in": "notes",
        "signal_out": "audio",
        "endpoint": "/render",
        "state": "ready",
        "evidence": "0.00 semitones of drift, measured across every setting",
    },
    {
        "id": "sing",
        "name": "Sing a melody",
        "takes": "a .mid file and words",
        "gives": "a sung vocal",
        "blurb": "Note-exact singing, with the pronunciation of every syllable open to editing.",
        "signal_in": "notes",
        "signal_out": "voice",
        "endpoint": "/sing",
        "state": "ready",
        "evidence": "SoulX-Singer, score-conditioned, 1.4 s per phrase",
    },
    {
        "id": "improve",
        "name": "Improve a recording",
        "takes": "an instrument recording",
        "gives": "the same take, better",
        "blurb": "Re-voices what you played without moving a note or bending the pitch.",
        "signal_in": "audio",
        "signal_out": "audio",
        "endpoint": "/enhance",
        "state": "ready",
        "evidence": "the same cover pass as Play a melody, minus the sampler",
    },
    {
        "id": "backing",
        "name": "Produce a backing track",
        "takes": "your layers, no vocal",
        "gives": "the same arrangement, produced",
        "blurb": "Needs proving first: on a full mix the model may rewrite the arrangement, not just the sound.",
        "signal_in": "audio",
        "signal_out": "audio",
        "endpoint": "/enhance",
        "state": "gated",
        "evidence": "every measurement so far used a single melodic line",
    },
    {
        "id": "sing_over",
        "name": "Sing over a track",
        "takes": "a recording and words",
        "gives": "a vocal on your tune",
        "blurb": "Reads the melody out of the audio, then sings your words on it.",
        "signal_in": "audio",
        "signal_out": "voice",
        "endpoint": "/sing",
        "state": "ready",
        "evidence": "pitch tracking reads the tune, then SoulX sings your words on it",
    },
    {
        "id": "voices",
        "name": "Voices",
        "takes": "a clip and the words sung",
        "gives": "a singer you can reuse",
        "blurb": "Your voices live on the pod's volume and are still here next time you switch it on.",
        "signal_in": "voice",
        "signal_out": "voice",
        "endpoint": "/voices",
        "state": "ready",
        "evidence": "zero-shot: a voice is a reference clip, no training",
    },
]

READY = {t["id"] for t in TOOLS if t["state"] == "ready"}


def public() -> dict:
    return {"tools": TOOLS}
