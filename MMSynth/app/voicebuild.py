"""Turning a clip and its words into a singer preset.

SoulX-Singer is zero-shot, so a voice is a reference clip. But it conditions on
that clip's *score* as well as its audio -- the prompt metadata carries the same
duration / phoneme / note_pitch / note_type fields a target does. A bare
recording is therefore not yet a voice.

Their own answer is a preprocessing pipeline with automatic vocal separation,
lyric transcription and note transcription. It wants torch 2.10, numpy 2.x, NeMo
and funasr -- all in direct conflict with the two Python environments this pod
already runs, for about 10 GB.

So this takes the cheaper route the user chose: **you say what was sung, and we
work out the rest.** Notes come from pitch tracking (app/transcribe.py) and
phonemes from the G2P already here (app/g2p.py). No third environment, no extra
weights, and the syllable alignment is the same code the singing path uses.

The alignment will sometimes be wrong. That matters less here than it would for
a target: the prompt's job is to carry timbre, and the melody it was sung on is
not the melody being generated.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from . import lyrics, soulx_score, transcribe, voices
from .errors import NotReady


def metadata_for(clip: Path, words: str) -> tuple[dict, str]:
    """The prompt metadata SoulX needs, derived from the clip and the words."""
    notes, note = transcribe.melody(clip)
    if not notes:
        raise NotReady("no melody could be read from that clip")

    if words.strip():
        spread = lyrics.distribute(notes, words)
        for n, a in zip(notes, spread.notes):
            n.lyric, n.phonemes = a.lyric, a.phonemes
        detail = f"{note} {spread.note}"
    else:
        lyrics.ensure_defaults(notes)
        detail = f"{note} No words given, so every note was set to 'da'."

    segments = soulx_score.build_target(notes)
    if not segments:
        raise NotReady("that clip produced no singable segments")

    # The prompt is one segment. Their loader takes the first entry of the list
    # and ignores the rest, so sending several would silently drop most of the
    # clip -- better to hand over the longest one deliberately.
    best = max(segments, key=lambda s: s["time"][1] - s["time"][0])
    best["index"] = "prompt_0"
    return best, detail


def add(clip_path, label: str, words: str, *, description: str = "",
        language: str = "en", agreement: str = "", recorded: bool = False,
        builtin: bool = False, voice_id: str | None = None) -> dict:
    """Register a voice: the clip, its derived score, and its provenance.

    Consent is a parameter with a default of "no" so the call site has to say
    something about it, and `voices.reference()` refuses anything without it.
    """
    voice_id = voice_id or uuid.uuid4().hex[:12]
    folder = voices.library_dir() / voice_id
    folder.mkdir(parents=True, exist_ok=True)

    clip = folder / "reference.wav"
    shutil.copyfile(str(clip_path), clip)

    meta, detail = metadata_for(clip, words)
    (folder / "reference.json").write_text(json.dumps([meta], indent=2),
                                           encoding="utf-8")

    data = {
        "id": voice_id,
        "label": label or "Untitled voice",
        "description": description or detail,
        "language": language,
        "clip": "reference.wav",
        "metadata": "reference.json",
        "builtin": builtin,
        "added_at": time.time(),
        "words": words,
        "derived": detail,
        "consent": {"agreement": agreement, "recorded": bool(recorded)},
    }
    (folder / "voice.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["_dir"] = str(folder)
    return voices.public(data)
