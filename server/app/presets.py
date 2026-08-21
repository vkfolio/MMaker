"""Style presets and prompt construction.

ACE-Step reads the prompt as a tag list, not prose. Its guidance: 3-7 tags,
ordered genre/era, then instruments, then mood, then tempo -- and never mix
contradictory descriptors, which produce muddled output. Presets are therefore
stored as ordered tag groups rather than sentences.
"""

STYLES = [
    {"id": "indie_soul",   "label": "Indie Soul",      "prompt": "warm indie soul, live drums, analog bass, rhodes, intimate", "bpm": 96,  "key_scale": "A Minor"},
    {"id": "lofi",         "label": "Lo-fi",           "prompt": "lo-fi hip hop, dusty drums, mellow keys, vinyl noise, relaxed", "bpm": 82, "key_scale": "F Major"},
    {"id": "cinematic",    "label": "Cinematic",       "prompt": "cinematic orchestral, swelling strings, deep percussion, epic", "bpm": 90, "key_scale": "D Minor"},
    {"id": "house",        "label": "House",           "prompt": "deep house, four on the floor, warm sub bass, filtered chords", "bpm": 124, "key_scale": "G Minor"},
    {"id": "drill",        "label": "Drill",           "prompt": "uk drill, sliding 808s, dark piano, sparse hats", "bpm": 142, "key_scale": "C# Minor"},
    {"id": "folk",         "label": "Folk",            "prompt": "acoustic folk, fingerpicked guitar, brushed drums, upright bass", "bpm": 104, "key_scale": "G Major"},
    {"id": "synthwave",    "label": "Synthwave",       "prompt": "synthwave, analog pads, gated drums, arpeggiated bass, neon", "bpm": 110, "key_scale": "F# Minor"},
    {"id": "ambient",      "label": "Ambient",         "prompt": "ambient texture, evolving pads, no drums, wide reverb", "bpm": 70, "key_scale": "C Major"},
    {"id": "rock",         "label": "Rock",            "prompt": "alt rock, driving drums, overdriven guitars, melodic bass", "bpm": 128, "key_scale": "E Minor"},
    {"id": "trap",         "label": "Trap",            "prompt": "trap, booming 808, crisp hats, dark bells", "bpm": 140, "key_scale": "A# Minor"},
]

SFX_PRESETS = [
    {"id": "impact",   "label": "Impact",       "prompt": "deep cinematic impact hit, sub rumble, short tail"},
    {"id": "riser",    "label": "Riser",        "prompt": "tension riser sweeping upward, building to a peak"},
    {"id": "whoosh",   "label": "Whoosh",       "prompt": "fast air whoosh transition, stereo movement"},
    {"id": "ui_click", "label": "UI Click",     "prompt": "clean minimal interface click, short and dry"},
    {"id": "ambience", "label": "Ambience",     "prompt": "quiet room tone with distant traffic, looping"},
    {"id": "foley",    "label": "Foley",        "prompt": "footsteps on gravel, close mic, natural"},
]

_BY_ID = {s["id"]: s for s in STYLES}

# Tags beyond this many dilute the prompt rather than sharpen it.
MAX_TAGS = 7


def build_prompt(user_text: str, style_id: str = "", *, instrumental: bool = False,
                 want_vocals: bool = False, bpm: int | None = None) -> str:
    """Compose a tag list: style tags first, the user's words next, then tempo.

    Order matters to the model (genre, instruments, mood, tempo) and the preset
    tags are already written in that order, so the user's own words slot in
    before the tempo tag rather than being appended at the end.
    """
    tags: list[str] = []

    preset = _BY_ID.get(style_id)
    if preset:
        tags.extend(t.strip() for t in preset["prompt"].split(",") if t.strip())

    for part in (user_text or "").split(","):
        part = part.strip()
        if part and part.lower() not in (t.lower() for t in tags):
            tags.append(part)

    # Vocal intent belongs in the tags; the lyrics field carries the words.
    if instrumental:
        if not any("instrumental" in t.lower() for t in tags):
            tags.append("instrumental")
    elif want_vocals:
        if not any(("vocal" in t.lower() or "sung" in t.lower()) for t in tags):
            tags.append("lead vocals")

    if bpm and not any("bpm" in t.lower() for t in tags):
        tags.append(f"{bpm} BPM")

    return ", ".join(tags[:MAX_TAGS])


def lyric_budget(duration_s: float) -> int:
    """Roughly how many words fit. ACE-Step sings ~2.5 words per second."""
    return max(8, int(duration_s * 2.5))


def style(style_id):
    return _BY_ID.get(style_id)


def compose_prompt(prompt: str, style_id: str = "") -> str:
    """Style preset supplies the bed; the user's words take precedence."""
    preset = _BY_ID.get(style_id)
    if not preset:
        return prompt.strip()
    if not prompt.strip():
        return preset["prompt"]
    return f"{preset['prompt']}, {prompt.strip()}"
