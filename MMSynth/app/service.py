"""The two pipelines. Everything above this is plumbing; everything below is a model.

    render:  notes -> melody -> transforms -> humanise -> sampler -> refine -> space
    sing:    notes -> melody -> words already on them -> singer -> space

Both stages of `render` are optional to the caller and both are meaningful
alone. The sampler by itself is offline, deterministic and immediate, which is
what you want while drawing notes; the refine pass is what turns a triggered
sample into something that sounds played. Keeping them separable is also what
makes a bad render diagnosable -- if the dry version is right and the refined
one is not, you know which half to blame.
"""

from __future__ import annotations

import random
from pathlib import Path

from . import audio, g2p, lyrics, presets, storage, transcribe, uploads, voices
from .config import settings
from .engines import music, sampler, singer
from .errors import NotReady
from .score import (Note, duration, flatten_to_melody, from_wire, humanise,
                    quantise, scale_time, swing, transpose)


def _seed(seed: int | None) -> int:
    return int(seed) if seed is not None else random.randint(1, 2_000_000_000)


def _apply_transforms(notes: list[Note], names, bpm: float) -> list[Note]:
    """The melody edits a quick action is allowed to make.

    Deliberately small and deliberately deterministic. A prompt reaches these by
    choosing among them, never by writing notes -- Libretto (arXiv 2606.22708)
    measured LLM MIDI editing at a 12% pass rate unaided, and editing an
    existing piece is the harder half of what it tested.
    """
    for name in names:
        if name == "up_octave":
            notes = transpose(notes, 12)
        elif name == "down_octave":
            notes = transpose(notes, -12)
        elif name == "swing":
            notes = swing(notes, bpm)
        elif name == "straighten":
            notes = quantise(notes, bpm)
        elif name == "half_time":
            notes = scale_time(notes, 2.0)
        elif name == "double_time":
            notes = scale_time(notes, 0.5)
    return notes


def _finish(path: Path, mix: float, decay_s: float) -> None:
    """Space and level, applied in place.

    Dry samples sounding dry is the single loudest tell that something came out
    of a MIDI file, and a render that arrives at a random level makes every
    later comparison a loudness comparison instead of a musical one.
    """
    data, sr = audio.read(path)
    data = audio.reverb(data, mix=mix, decay_s=decay_s, sr=sr)
    data = audio.match_loudness(data)
    audio.write(path, audio.fade(data, sr=sr), sr)


def render(request, report=None) -> dict:
    """Play a melody on an AI instrument."""
    notes = from_wire(request.notes)
    notes, dropped = flatten_to_melody(notes)
    if not notes:
        raise NotReady("there are no notes to play")

    resolved = presets.resolve(request.instrument_id, request.actions,
                              request.prompt)
    seed = _seed(request.seed)

    if report:
        report(0.05, "shaping the performance")
    notes = _apply_transforms(notes, resolved.transforms, request.bpm)
    notes = humanise(notes, request.bpm, resolved.humanise, seed=seed)

    name, path = storage.new_name("render")
    work = storage.work_dir() / (Path(name).stem + "_dry.wav")

    sampler().render(work, notes, request.bpm, resolved, report=report)

    wants_refine = (not request.dry and settings.refine_enabled
                    and bool(resolved.refine.prompt))
    if wants_refine:
        # `cover`, not text2music: the sampler render is the melody carrier and
        # ACE-Step's FSQ codes are what keep the tune while the caption changes
        # the instrument. Measured at 0.00 semitones of drift (PHASE0.md).
        music().cover(path, work, resolved.refine.prompt,
                      noise=resolved.refine.noise,
                      strength=resolved.refine.strength,
                      steps=settings.acestep_steps,
                      guidance=settings.acestep_guidance,
                      seed=seed, report=report)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(work).read_bytes())

    if report:
        report(0.95, "room and level")
    _finish(path, resolved.space.mix, resolved.space.decay_s)

    said = []
    if dropped:
        said.append(f"kept the top line, dropping {dropped} note(s) underneath")
    if not wants_refine:
        said.append("sampler only" if request.dry else
                    "sampler only: the refine engine is off")

    result = {
        "audio": name,
        "duration_s": round(duration(notes), 3),
        "instrument": resolved.instrument.public(),
        "notes": len(notes),
        "bpm": request.bpm,
        "seed": seed,
        "refined": wants_refine,
        "prompt": resolved.refine.prompt,
        "noise": resolved.refine.noise,
        "strength": resolved.refine.strength,
        "actions": list(request.actions),
        "note": "; ".join(said),
    }
    storage.write_sidecar(name, result)
    return result


def distribute_lyrics(request) -> dict:
    """Spread a typed line across the notes, rendering nothing.

    Pure, and deliberately so. The document lives in the client, so this returns
    what the notes *would* become together with the number of hand-set notes it
    would replace; the client warns and then keeps or discards the result. That
    is the whole "warn before overwriting" mechanism, and it needs no state on
    this side to work.
    """
    notes = from_wire(request.notes)
    spread = lyrics.distribute(notes, request.line)
    assigned = [{"lyric": a.lyric, "phonemes": a.phonemes,
                 "display": g2p.display(a.phonemes)} for a in spread.notes]
    # What the lyric bar should read afterwards, rebuilt from the notes rather
    # than echoed back, so a client that trusts it is trusting the same
    # round trip the next edit will make.
    for note, a in zip(notes, spread.notes):
        note.lyric, note.phonemes = a.lyric, a.phonemes
    return {
        "notes": assigned,
        "pending": spread.pending,
        "held": spread.held,
        "replaced": spread.replaced,
        "line": lyrics.current_line(notes),
        "note": spread.note,
    }


def phonemes_for_words(request) -> dict:
    """ARPAbet per word, split into syllables. For the row above the notes."""
    out = []
    for word in request.words:
        for label, sounds in g2p.syllables(word):
            out.append({"word": word, "lyric": label, "phonemes": sounds,
                        "display": g2p.display(sounds)})
    return {"syllables": out}


def sing(request, report=None) -> dict:
    """Sing a melody.

    The words are already on the notes -- POST /lyrics is what puts them there.
    Anything still lyric-less is given `da`, so this cannot fail for want of
    words; a melody someone drew and hit sing on is a melody that sings.
    """
    notes = from_wire(request.notes)
    notes, dropped = flatten_to_melody(notes)
    if not notes:
        raise NotReady("there are no notes to sing")

    filled = lyrics.ensure_defaults(notes)
    parts = lyrics.segments(notes)
    if not parts:
        raise NotReady("every note is a hold, so there is no syllable to start on")

    reference = voices.reference(request.voice_id) if request.voice_id else None
    seed = _seed(request.seed)

    name, path = storage.new_name("sing")
    work = storage.work_dir() / (Path(name).stem + "_raw.wav")
    singer().sing(work, notes, request.bpm, reference,
                  language=request.language, seed=seed, report=report)

    # Optionally run the sung take back through ACE-Step. Measured in PHASE0.md
    # gate 3: the singing survives (voiced 0.56 -> 0.85) and the melody is held
    # exactly (2.08 -> 2.04 semitones). What it does NOT do is change who is
    # singing -- captions gave no voice separation at all, so this is "another
    # take, richer" and never "make it a different singer". That knob is the
    # SoulX reference clip.
    if request.revoice and settings.refine_enabled:
        music().cover(path, work, request.revoice_prompt or "solo vocal, natural",
                      noise=settings.revoice_noise, strength=1.0,
                      steps=settings.acestep_steps,
                      guidance=settings.acestep_guidance,
                      seed=seed, report=report)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(work).read_bytes())

    if report:
        report(0.95, "room and level")
    # A voice wants less room than an instrument: reverb on a lead vocal is a
    # mix decision, and this one is being handed to a DAW to make properly.
    _finish(path, mix=0.12, decay_s=1.4)

    said = []
    if dropped:
        said.append(f"kept the top line, dropping {dropped} note(s) underneath")
    if filled:
        said.append(f"{filled} note(s) had no words and were given '{lyrics.DEFAULT_LYRIC}'")
    held = sum(p.count - 1 for p in parts)
    if held:
        said.append(f"{held} note(s) held a syllable rather than starting one")

    result = {
        "audio": name,
        "duration_s": round(duration(notes), 3),
        "voice_id": request.voice_id,
        "notes": len(notes),
        "syllables": len(parts),
        "bpm": request.bpm,
        "seed": seed,
        "line": lyrics.current_line(notes),
        "revoiced": bool(request.revoice and settings.refine_enabled),
        "note": "; ".join(said),
    }
    storage.write_sidecar(name, result)
    return result


def enhance(request, report=None) -> dict:
    """Re-voice audio the user already has.

    The same `cover` call the instrument path ends with, minus the sampler --
    the melody carrier is unnecessary when the source is already a recording.
    Measured to hold pitch to 0.00-0.03 semitones (PHASE0.md), which is what
    makes "without changing timing or scale" a claim rather than a hope.
    """
    src = uploads.path(request.upload_id)
    seed = _seed(request.seed)

    if request.instrument_id:
        resolved = presets.resolve(request.instrument_id, request.actions,
                                   request.prompt)
        caption, noise = resolved.refine.prompt, resolved.refine.noise
        label = resolved.instrument.label
    else:
        caption = request.prompt.strip()
        if not caption:
            raise NotReady(
                "say what it should sound like, or pick an instrument -- with "
                "no caption the model has nothing to change it towards")
        noise = 0.35
        label = caption.split(",")[0][:40]

    # A full arrangement carries far more in its semantic codes than a single
    # line, and the model may reinterpret harmony rather than just timbre.
    # Holding the source harder is the only lever we have against that.
    if request.keep_arrangement:
        noise = max(noise, 0.75)

    name, path = storage.new_name("enhance")
    music().cover(path, src, caption, noise=noise, strength=1.0,
                  steps=settings.acestep_steps,
                  guidance=settings.acestep_guidance,
                  seed=seed, report=report)

    if report:
        report(0.95, "levelling")
    # No added reverb: the source has its own space, and putting a room on a
    # room is the fastest way to make a mix sound worse than it started.
    data, sr = audio.read(path)
    audio.write(path, audio.fade(audio.match_loudness(data), sr=sr), sr)

    info = uploads.meta(request.upload_id) or {}
    result = {
        "audio": name,
        "source": info.get("name", request.upload_id),
        "duration_s": info.get("seconds"),
        "label": label,
        "prompt": caption,
        "noise": noise,
        "seed": seed,
        "kept_arrangement": bool(request.keep_arrangement),
        "note": "held hard to the source" if request.keep_arrangement else "",
    }
    storage.write_sidecar(name, result)
    return result


def transcribe_audio(request) -> dict:
    """Read a melody out of a recording. Immediate, no GPU, no job."""
    src = uploads.path(request.upload_id)
    notes, note = transcribe.melody(src)
    lyrics.ensure_defaults(notes)
    return {
        "notes": [{"start": round(n.start, 4), "length": round(n.length, 4),
                   "pitch": n.pitch, "velocity": n.velocity,
                   "lyric": n.lyric, "phonemes": n.phonemes} for n in notes],
        "bpm": 120.0,
        "note": note,
    }
