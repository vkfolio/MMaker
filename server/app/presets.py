"""Style presets -- prompt templates, our code rather than a model feature.

Kept as data so they can be edited without touching the engine, and so the UI
can show the same list it sends.
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
