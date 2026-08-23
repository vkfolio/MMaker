"""Operations. Routers stay thin; everything that touches audio lives here.

Every mutating operation follows the same shape: build the params, call an
engine, verify the result against the project grid, append an immutable version.
Nothing is ever overwritten.
"""

from __future__ import annotations

import random
import uuid
import zipfile
from pathlib import Path

import math

import time

from . import analysis, audio, engines, midi, presets, storage, voices
from .base_errors import NotReady, RenderError
from .schemas import (CoverRequest, Grid, Project, Stem, StemVersion, SyncReport,
                      TrackClass, Variation)

# Layers are generated against the mixdown of what already exists, so the model
# hears the whole arrangement rather than one track.
MIX_HEADROOM_DB = -1.0


def _rel(project: Project, path: Path) -> str:
    return str(Path(path).relative_to(storage.project_dir(project.id))).replace("\\", "/")


def _abs(project: Project, rel: str) -> Path:
    return storage.project_dir(project.id) / rel


def _seed(explicit: int | None = None) -> int:
    return explicit if explicit is not None else random.randint(1, 2_000_000_000)


def _quality(project: Project) -> dict:
    """Only the real engine takes a quality tier; the stub has no such notion."""
    if engines.music().name != "acestep":
        return {}
    return {"quality": project.quality, "thinking": project.thinking}


def _engine_meta() -> dict:
    """What the engine reported rendering with, if it said anything.

    Asking for a model the pod lacks makes ACE-Step fall back silently; this
    is how that becomes visible instead of just looking fast.
    """
    engine = engines.music()
    meta = getattr(engine, "last_meta", None)
    return {"engine": dict(meta)} if meta else {}


def _verify(path: Path, grid: Grid, auto_nudge: bool = True) -> SyncReport:
    """Beat-check a generated stem and correct small, consistent latency.

    Scattered drift is left alone -- nudging cannot fix it, and pretending
    otherwise would hide a real quality problem behind a plausible number.
    """
    report = analysis.sync_report(path, grid.bpm)
    if auto_nudge and report.nudge_ms and abs(report.nudge_ms) < 120:
        audio.shift_ms(path, path, report.nudge_ms)
        rechecked = analysis.sync_report(path, grid.bpm)
        rechecked.nudge_ms = report.nudge_ms
        return rechecked
    return report


# ---------------------------------------------------------------------------
# Phase 2 -- idea to variations
# ---------------------------------------------------------------------------

def detect_grid(project: Project, source: Path) -> dict:
    """Detect bpm/key from the idea. Never auto-applied without confirmation."""
    found = analysis.detect(source)
    if found.get("bpm"):
        project.grid.bpm = found["bpm"]
    if found.get("key_scale"):
        project.grid.key_scale = found["key_scale"]
    project.grid.confirmed = False          # a human still has to agree
    storage.save(project)
    return found


def generate_variations(project: Project, count: int, report=None) -> list[dict]:
    """N renders on one grid, differing only by seed.

    ACE-Step sings whenever lyrics are supplied, and writes its own words when
    the prompt implies vocals. Instrumental is therefore an explicit choice,
    not the default -- we say so in the prompt and withhold lyrics.
    """
    engine = engines.music()
    lyrics = (project.lyrics or "").strip()

    # ACE-Step reads "[instrumental]" in the LYRICS field as the instruction to
    # leave vocals out -- that is the documented control, not prompt wording.
    if project.instrumental:
        lyrics = "[instrumental]"
    prompt = presets.build_prompt(
        project.prompt, project.style,
        instrumental=project.instrumental,
        want_vocals=not project.instrumental and not lyrics,
        bpm=project.grid.bpm,
    )
    src = _abs(project, project.source_audio) if project.source_audio else None

    out_dir = storage.project_dir(project.id) / "variations"
    out_dir.mkdir(parents=True, exist_ok=True)

    made = []
    for i in range(count):
        if report:
            report(i / count, f"variation {i + 1} of {count}")
        seed = _seed()
        dest = out_dir / f"var_{i + 1}_{seed}.wav"
        # report is threaded in so the engine's poll loop is a cancellation
        # point -- without it a cancelled render runs to completion unnoticed.
        engine.generate(dest, prompt=prompt, grid=project.grid, seed=seed,
                        lyrics=lyrics, vocal_language=project.vocal_language,
                        src_audio=src, report=report, **_quality(project))
        variation = Variation(seed=seed, audio=_rel(project, dest), prompt=prompt,
                              engine=_engine_meta().get("engine", {}))
        project.variations.append(variation)
        made.append(variation.model_dump())

    storage.save(project)
    if report:
        report(1.0, f"{count} variations ready")
    return made


# ---------------------------------------------------------------------------
# Phase 3 -- stems
# ---------------------------------------------------------------------------

def split_variation(project: Project, variation_id: str, method: str = "demucs",
                    report=None, tier: str = "professional",
                    remove_reverb: bool = False) -> list[dict]:
    """Separate a variation into stems, appending a new set.

    Splitting used to clear project.stems and write flat filenames like
    stems/bass.wav, so a second split destroyed every stem id and overwrote the
    audio a saved client project referenced. Each run now gets its own id and
    its own directory, and older sets are kept.
    """
    variation = project.variation(variation_id)
    if not variation:
        raise NotReady(f"no such variation: {variation_id}")

    split_id = f"split_{uuid.uuid4().hex[:8]}"
    src = _abs(project, variation.audio)
    out_dir = storage.project_dir(project.id) / "stems" / split_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if report:
        report(0.1, f"separating with {method}")

    if method == "extract":
        engine = engines.music()
        wanted = (["vocals", "keyboard"] if tier == "basic"
                  else ["vocals", "drums", "bass", "keyboard"])
        parts = {}
        for i, track in enumerate(wanted):
            if report:
                report(0.1 + 0.8 * i / len(wanted), f"extracting {track}")
            dest = out_dir / f"{track}.wav"
            parts[track] = engine.extract(dest, src, track, project.grid)
    else:
        parts = engines.separator().separate(src, out_dir, tier=tier)
        if tier == "basic":
            parts = _fold_to_basic(parts, out_dir)

    # Demucs calls its catch-all "other"; the rest of the app speaks track classes.
    rename = {"other": "keyboard", "piano": "keyboard", "guitar": "guitar"}
    project.chosen_variation = variation_id
    project.active_split = split_id

    unfulfilled = []
    if remove_reverb:
        # Say so rather than silently ignoring it: we host no de-reverb model.
        unfulfilled.append("remove_reverb")

    made = []
    for name, path in parts.items():
        track_class = rename.get(name, name)
        if track_class not in TrackClass.__args__:
            track_class = "keyboard"
        stem = Stem(split_id=split_id, track_class=track_class,
                    label=name.title(), prompt=project.prompt)
        stem.add(StemVersion(
            audio=_rel(project, Path(path)), op="import",
            params={"from": variation_id, "method": method,
                    "split_id": split_id, "tier": tier,
                    "unfulfilled": unfulfilled},
            sync=analysis.sync_report(path, project.grid.bpm),
        ))
        project.stems.append(stem)
        made.append(stem.model_dump())

    storage.save(project)
    if report:
        report(1.0, f"{len(made)} stems")
    return made


def _fold_to_basic(parts: dict, out_dir: Path) -> dict:
    """Basic is vocals + everything else, which we sum rather than re-separate."""
    vocals = parts.get("vocals")
    others = [(name, p) for name, p in parts.items() if name != "vocals"]
    if not vocals or not others:
        return parts
    inst = out_dir / "instrumental.wav"
    audio.mix([(p, 0.0, 0.0) for _, p in others], inst)
    for name, p in others:
        Path(p).unlink(missing_ok=True)
    return {"vocals": vocals, "keyboard": inst}


def mixdown(project: Project, dest: Path | None = None,
            exclude_stem: str | None = None) -> Path:
    """Sum the audible stems. This is the conditioning context for new layers."""
    sources = []
    for stem in project.audible_stems():
        if exclude_stem and stem.id == exclude_stem:
            continue
        sources.append((_abs(project, stem.version.audio), stem.gain_db, stem.pan))
    if not sources:
        raise NotReady("no audible stems to mix")
    dest = dest or (storage.project_dir(project.id) / "mixdown.wav")
    path, _ = audio.mix(sources, dest, headroom_db=MIX_HEADROOM_DB)
    return path


def export_zip(project: Project, report=None) -> Path:
    """Stems plus a manifest, ready to drop into FL Studio."""
    d = storage.project_dir(project.id)
    out = d / f"{project.id}_stems.zip"

    lines = [
        f"project: {project.title}",
        f"bpm: {project.grid.bpm}",
        f"key: {project.grid.key_scale}",
        f"time signature: {project.grid.time_sig}",
        f"bars: {project.grid.bars}",
        f"duration: {project.grid.duration_s:.2f}s",
        "",
        "stems:",
    ]

    stems = project.active_stems()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, stem in enumerate(stems):
            if not stem.version:
                continue
            if report:
                report(i / max(len(stems), 1), f"packing {stem.label}")
            src = _abs(project, stem.version.audio)
            name = f"{stem.track_class}_{stem.id[-4:]}.wav"
            zf.write(src, name)
            sync = stem.version.sync
            lines.append(
                f"  {name}  gain {stem.gain_db:+.1f} dB  pan {stem.pan:+.2f}"
                f"  sync {sync.verdict}"
                + (f" ({sync.median_dev_ms} ms)" if sync.median_dev_ms is not None else ""))
        try:
            mix_path = mixdown(project)
            zf.write(mix_path, "_mixdown.wav")
            lines.append("")
            lines.append("  _mixdown.wav  reference mix")
        except NotReady:
            pass
        zf.writestr("README.txt", "\n".join(lines) + "\n")

    if report:
        report(1.0, "export ready")
    return out


# ---------------------------------------------------------------------------
# Phase 4 -- per-stem editing
# ---------------------------------------------------------------------------

def _require_confirmed(project: Project):
    if not project.grid.confirmed:
        raise NotReady(
            "confirm bpm and key before layering -- a wrong grid corrupts every "
            "stem that locks to it (POST /projects/{id}/grid)")


def _next_path(project: Project, stem: Stem, op: str) -> Path:
    d = storage.project_dir(project.id) / "stems"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem.id}_{op}_{len(stem.versions)}.wav"


def repaint_stem(project: Project, stem: Stem, start_s: float, end_s: float,
                 prompt: str | None = None, seed: int | None = None,
                 report=None, strength: float | None = None) -> dict:
    """Regenerate a time region. Callers pass seconds; bars are resolved above."""
    _require_confirmed(project)
    if end_s <= start_s:
        raise NotReady("the end of the region must come after its start")

    src = _abs(project, stem.version.audio)
    duration = audio.describe(src)["duration_s"]
    end_s = min(end_s, duration)
    if start_s >= duration:
        raise NotReady(
            f"the region starts at {start_s:.2f}s but the stem is only "
            f"{duration:.2f}s long")

    seed = _seed(seed)
    dest = _next_path(project, stem, "repaint")
    if report:
        report(0.15, f"repainting {start_s:.2f}s-{end_s:.2f}s")

    # Render to a scratch file, then splice only the requested window back into
    # the original.
    #
    # ACE-Step re-encodes the whole clip, so the parts nobody asked to change
    # come back *re-synthesised* -- close to the original but not it. Measured,
    # that was ~-42 dBFS of drift across untouched seconds: quiet, audible on a
    # sparse stem, and squarely audio the user asked to keep. Splicing makes
    # outside the region bit-identical instead of merely similar, which is what
    # the voice path already does a few functions below.
    rendered = dest.with_suffix(".render.wav")
    engines.music().repaint(rendered, src, start_s, end_s,
                            prompt or stem.prompt, project.grid, seed,
                            report=report, strength=strength,
                            **_quality(project))
    try:
        audio.splice(src, rendered, dest, start_s, end_s)
    finally:
        rendered.unlink(missing_ok=True)

    # Prove the untouched region really was untouched, rather than trusting it.
    outside_residual = _outside_residual(src, dest, start_s, end_s)
    version = stem.add(StemVersion(
        audio=_rel(project, dest), op="repaint",
        params={"start_s": round(start_s, 3), "end_s": round(end_s, 3),
                "seed": seed, "prompt": prompt or stem.prompt,
                **({"strength": strength} if strength is not None else {}),
                "outside_residual": outside_residual},
        sync=_verify(dest, project.grid),
        # 0.05 relative RMS is about -26 dB -- it accepted plainly audible
        # drift. With the splice, anything outside the window should be
        # identical, so this now only tolerates float noise.
        note=("context preserved" if outside_residual is not None
              and outside_residual < 0.001 else
              f"WARNING: audio outside the region changed (residual {outside_residual})"),
    ))
    storage.save(project)
    if report:
        report(1.0, "repainted")
    return version.model_dump()


def _outside_residual(before: Path, after: Path, start_s: float, end_s: float):
    """RMS difference outside the repainted window, relative to the original."""
    try:
        import numpy as np
        a, sr = audio.load(before)
        b, sr_b = audio.load(after)
        if sr != sr_b:
            return None
        n = min(len(a), len(b))
        w = min(a.shape[1], b.shape[1])
        a, b = a[:n, :w], b[:n, :w]
        mask = np.ones(n, dtype=bool)
        mask[int(start_s * sr):min(int(end_s * sr), n)] = False
        if not mask.any():
            return None
        rms = float(np.sqrt(np.mean(a[mask] ** 2))) or 1e-12
        return round(float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) / rms, 4)
    except Exception:                                    # noqa: BLE001
        return None


def extend_stem(project: Project, stem: Stem, bars: int,
                prompt: str | None = None, report=None) -> dict:
    _require_confirmed(project)
    added_s = bars * project.grid.beats_per_bar * 60.0 / project.grid.bpm
    seed = _seed()
    dest = _next_path(project, stem, "extend")
    if report:
        report(0.15, f"extending by {bars} bars")

    engines.music().extend(dest, _abs(project, stem.version.audio), added_s,
                           prompt or stem.prompt, project.grid, seed,
                           report=report, **_quality(project))

    version = stem.add(StemVersion(
        audio=_rel(project, dest), op="extend",
        params={"bars": bars, "seed": seed, "prompt": prompt or stem.prompt},
        sync=_verify(dest, project.grid),
    ))
    project.grid.bars += bars
    storage.save(project)
    if report:
        report(1.0, "extended")
    return version.model_dump()


def vary_stem(project: Project, stem: Stem, seed: int | None = None,
              prompt: str | None = None, report=None) -> dict:
    """An alternate take of the whole stem, still fitting the other layers.

    This is the honest limit of "improvise": a different performance, not a
    developed idea. Real melodic development is a symbolic problem.
    """
    _require_confirmed(project)
    seed = _seed(seed)
    dest = _next_path(project, stem, "vary")
    if report:
        report(0.15, "generating an alternate take")

    context = mixdown(project, storage.project_dir(project.id) / "_context.wav",
                      exclude_stem=stem.id)
    engines.music().layer(dest, stem.track_class, context,
                          prompt or stem.prompt, project.grid, seed,
                          lyrics=project.lyrics if stem.track_class == "vocals" else "",
                          vocal_language=project.vocal_language,
                          report=report, **_quality(project))

    version = stem.add(StemVersion(
        audio=_rel(project, dest), op="vary",
        params={"seed": seed, "prompt": prompt or stem.prompt},
        sync=_verify(dest, project.grid),
    ))
    storage.save(project)
    if report:
        report(1.0, "alternate take ready")
    return version.model_dump()


def add_layer(project: Project, track_class: str, prompt: str = "",
              seed: int | None = None, report=None,
              start_s: float | None = None, end_s: float | None = None) -> dict:
    """Generate a new stem that fits everything already in the project.

    With a region, only that slice is generated and it is placed at the right
    offset inside a silent bed, so a layer can answer "just here" rather than
    always spanning the whole arrangement.
    """
    _require_confirmed(project)
    seed = _seed(seed)
    stem = Stem(split_id=project.active_split or "",
                track_class=track_class,
                label=track_class.replace("_", " ").title(),
                prompt=prompt or project.prompt)

    if report:
        report(0.1, "mixing context")
    try:
        context = mixdown(project, storage.project_dir(project.id) / "_context.wav")
    except NotReady:
        context = None

    region = None
    if context is not None and start_s is not None and end_s is not None and end_s > start_s:
        region = (float(start_s), float(end_s))
        sliced = storage.project_dir(project.id) / "_context_region.wav"
        audio.slice_seconds(context, sliced, region[0], region[1])
        context = sliced

    dest = _next_path(project, stem, "layer")
    if report:
        report(0.25, f"generating {track_class}")

    composed = presets.build_prompt(prompt or project.prompt, project.style,
                                    bpm=project.grid.bpm)
    engine = engines.music()
    if context is None:
        engine.generate(dest, composed, project.grid, seed,
                        lyrics=project.lyrics if track_class == "vocals" else "",
                        vocal_language=project.vocal_language,
                        report=report, **_quality(project))
    else:
        engine.layer(dest, track_class, context, composed, project.grid, seed,
                     lyrics=project.lyrics if track_class == "vocals" else "",
                     vocal_language=project.vocal_language,
                     report=report, **_quality(project))

    if region is not None:
        # Put the generated slice back where it belongs on the timeline.
        placed = _next_path(project, stem, "placed")
        audio.place(dest, placed, region[0], project.grid.duration_s)
        dest = placed

    stem.add(StemVersion(
        audio=_rel(project, dest), op="generate",
        params={"seed": seed, "prompt": composed, "track_class": track_class,
                **({"start_s": region[0], "end_s": region[1]} if region else {})},
        sync=_verify(dest, project.grid),
    ))
    project.stems.append(stem)
    storage.save(project)
    if report:
        report(1.0, f"{track_class} added")
    return stem.model_dump()


# ---------------------------------------------------------------------------
# Phase 5 -- vocals and voices
# ---------------------------------------------------------------------------

def generate_vocals(project: Project, lyrics: str | None = None,
                    vocal_language: str | None = None, voice_id: str | None = None,
                    seed: int | None = None, report=None) -> dict:
    """Sing the lyrics over the current instrumental, then apply a voice."""
    _require_confirmed(project)
    if lyrics is not None:
        project.lyrics = lyrics
    if vocal_language:
        project.vocal_language = vocal_language
    if not project.lyrics.strip():
        raise NotReady("lyrics are required to generate vocals")

    stem_dict = add_layer(project, "vocals", project.prompt, seed, report=report)
    stem = project.stem(stem_dict["id"])

    if voice_id:
        if report:
            report(0.85, "applying voice")
        apply_voice(project, stem, voice_id)
    return stem.model_dump()


def render_score(project: Project, notes, bpm: int | None = None,
                 program: int = 0, label: str = "Score",
                 track_class: TrackClass = "keyboard",
                 prompt: str | None = None, seed: int | None = None,
                 report=None) -> dict:
    """Render notes to audio, and optionally reimagine that render.

    Two stages, both optional to the caller. The sampler alone is offline,
    deterministic and immediate -- it is what makes a piano roll audible at all,
    and it is the honest floor of quality. Given a prompt, the General MIDI
    render then becomes `src_audio` for a cover, which is how a triggered patch
    turns into something that sounds played. That second stage is the same path
    a hummed take takes, and it needs no note conditioning from the model --
    which is fortunate, because ACE-Step has none.

    The stem is a normal stem afterwards. Repaint, cover, voice and the version
    history all work on it, because the audio is what they act on and the score
    is the desktop's business.
    """
    if not notes:
        raise NotReady("there are no notes to render")

    grid_bpm = bpm or project.grid.bpm
    out_dir = storage.project_dir(project.id) / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)

    if report:
        report(0.1, "writing the score")
    smf_path = out_dir / f"score_{stamp}.mid"
    midi.write_smf(notes, smf_path, bpm=grid_bpm, program=program)

    if report:
        report(0.3, "playing it")
    rendered = out_dir / f"score_{stamp}.wav"
    try:
        midi.render(smf_path, rendered)
    except RenderError as exc:
        # Say which of the two things is missing rather than "render failed".
        raise NotReady(f"cannot play the score: {exc}") from exc

    final = rendered
    op = "score"
    params: dict = {"bpm": grid_bpm, "program": program, "notes": len(notes)}

    if prompt:
        if report:
            report(0.5, "reimagining it")
        seed = _seed(seed)
        composed = presets.build_prompt(prompt, project.style, bpm=grid_bpm)
        covered = out_dir / f"score_{stamp}_cover.wav"
        engines.music().generate(
            covered, prompt=composed, grid=project.grid, seed=seed,
            lyrics="", vocal_language=project.vocal_language,
            src_audio=rendered, report=report, **_quality(project))
        final = covered
        op = "score-cover"
        params.update({"prompt": composed, "seed": seed})

    stem = Stem(split_id=project.active_split or "",
                track_class=track_class,
                label=label or "Score",
                prompt=prompt or "")
    stem.add(StemVersion(audio=_rel(project, final), op=op, params=params,
                         sync=_verify(final, project.grid)))
    project.stems.append(stem)
    storage.save(project)
    if report:
        report(1.0, "score rendered")
    return stem.model_dump()


def cover_stem(project: Project, stem: Stem, prompt: str | None = None,
               strength: float = 0.5, seed: int | None = None,
               region: tuple[float, float] | None = None,
               report=None) -> dict:
    """Reimagine a stem -- or one region of it -- against a prompt.

    The difference between this and vary_stem is which question is being
    answered. vary_stem asks the model for another performance of the same
    part, conditioned on the *other* layers, and needs `lego`. This conditions
    on the part itself and needs only `cover`, which every checkpoint
    implements -- so it is the one form of "play with my idea" that works on a
    laptop install.

    It is also the path an imported recording or a hummed melody takes: seed
    audio in, a prompt, and a dial for how much of the original survives.
    """
    _require_confirmed(project)
    seed = _seed(seed)
    src = _abs(project, stem.version.audio)
    dest = _next_path(project, stem, "cover")
    if report:
        report(0.15, "reimagining" + (" the selection" if region else ""))

    composed = presets.build_prompt(prompt or stem.prompt or project.prompt,
                                    project.style, bpm=project.grid.bpm)
    cover_params = CoverRequest(strength=strength).engine_params()
    engine = engines.music()

    if region:
        # Cover the slice alone and splice it back, so the rest of the stem is
        # the same file it was. Regenerating the whole stem and trusting the
        # model to leave the other bars alone is what made repaint bleed.
        piece = storage.project_dir(project.id) / "_cover_region.wav"
        audio.slice_seconds(src, piece, region[0], region[1])

        # The model is told how long to render, and the grid it inherits
        # describes the whole project -- so without this a four-bar selection
        # asks for a four-minute cover. Round the region up to whole bars:
        # splice truncates a long patch, but zero-pads a short one, and silence
        # in the middle of a take is worse than a slightly early cut.
        bar_s = project.grid.beats_per_bar * 60.0 / project.grid.bpm
        region_bars = max(1, math.ceil((region[1] - region[0]) / bar_s))
        grid = project.grid.model_copy(update={"bars": region_bars})

        covered = storage.project_dir(project.id) / "_cover_result.wav"
        engine.generate(covered, composed, grid, seed,
                        lyrics=project.lyrics if stem.track_class == "vocals" else "",
                        vocal_language=project.vocal_language,
                        src_audio=piece, report=report, cover=cover_params,
                        **_quality(project))
        audio.splice(src, covered, dest, region[0], region[1])
    else:
        engine.generate(dest, composed, project.grid, seed,
                        lyrics=project.lyrics if stem.track_class == "vocals" else "",
                        vocal_language=project.vocal_language,
                        src_audio=src, report=report, cover=cover_params,
                        **_quality(project))

    version = stem.add(StemVersion(
        audio=_rel(project, dest), op="cover",
        params={"seed": seed, "prompt": prompt or stem.prompt,
                "strength": strength, **cover_params,
                **({"start_s": region[0], "end_s": region[1]} if region else {})},
        sync=_verify(dest, project.grid),
    ))
    storage.save(project)
    if report:
        report(1.0, "ready")
    return version.model_dump()



def apply_voice(project: Project, stem: Stem, voice_id: str, report=None,
                start_s: float | None = None, end_s: float | None = None) -> dict:
    """Convert a vocal stem onto a library voice, optionally only a region."""
    reference = voices.reference_path(voice_id)
    if not reference:
        raise NotReady(f"no such voice: {voice_id}")

    src = _abs(project, stem.version.audio)
    dest = _next_path(project, stem, "voice")
    if report:
        report(0.3, "converting voice")

    region = None
    if start_s is not None and end_s is not None and end_s > start_s:
        region = (float(start_s), float(end_s))
        piece = storage.project_dir(project.id) / "_voice_region.wav"
        audio.slice_seconds(src, piece, region[0], region[1])
        converted = storage.project_dir(project.id) / "_voice_converted.wav"
        engines.voice().convert_voice(converted, piece, reference)
        audio.splice(src, converted, dest, region[0], region[1])
    else:
        engines.voice().convert_voice(dest, src, reference)

    stem.voice_id = voice_id
    version = stem.add(StemVersion(
        audio=_rel(project, dest), op="voice",
        params={"voice_id": voice_id,
                "label": (voices.get(voice_id) or {}).get("label"),
                **({"start_s": region[0], "end_s": region[1]} if region else {})},
        sync=_verify(dest, project.grid, auto_nudge=False),
    ))
    storage.save(project)
    if report:
        report(1.0, "voice applied")
    return version.model_dump()


def isolate_vocals(project: Project, stem: Stem, report=None) -> dict:
    """Pull a clean vocal out of a stem that still has backing in it."""
    src = _abs(project, stem.version.audio)
    out_dir = storage.project_dir(project.id) / "stems" / f"{stem.id}_isolate"
    if report:
        report(0.2, "isolating vocals")
    parts = engines.separator().separate(src, out_dir)
    if "vocals" not in parts:
        raise RenderError("separator returned no vocals track")

    dest = _next_path(project, stem, "isolate")
    audio.write(dest, *audio.load(parts["vocals"]))
    version = stem.add(StemVersion(
        audio=_rel(project, dest), op="isolate",
        params={"method": engines.separator().name},
        sync=_verify(dest, project.grid, auto_nudge=False),
    ))
    storage.save(project)
    if report:
        report(1.0, "vocals isolated")
    return version.model_dump()


# ---------------------------------------------------------------------------
# Phase 6 -- SFX
# ---------------------------------------------------------------------------

def generate_sfx(prompt: str, duration_s: float, count: int,
                 seed: int | None = None, report=None) -> list[dict]:
    """One-shots and textures. Separate from the music path by design."""
    from .config import settings
    out_dir = settings.data_dir / "sfx"
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = engines.sfx()
    made = []
    for i in range(count):
        if report:
            report(i / count, f"sfx {i + 1} of {count}")
        this_seed = _seed(seed if seed is None else seed + i)
        dest = out_dir / f"sfx_{this_seed}.wav"
        engine.sfx(dest, prompt, duration_s, this_seed)
        made.append({
            "seed": this_seed,
            "prompt": prompt,
            "audio": f"sfx/{dest.name}",
            **audio.describe(dest),
        })
    if report:
        report(1.0, f"{count} sounds")
    return made
