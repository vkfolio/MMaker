"""Project lifecycle: create from an idea, confirm the grid, split, export."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import analysis, jobs, midi, presets, service, storage
from ..base_errors import MusicmakerError
from ..schemas import (AddLayerRequest, ConfirmGrid, CreateProject, Grid,
                       Project, SfxRequest, SplitRequest, VocalRequest)

router = APIRouter(prefix="/api", tags=["projects"])


def _get(project_id: str) -> Project:
    project = storage.load(project_id)
    if not project:
        raise HTTPException(404, f"no such project: {project_id}")
    return project


def _guard(fn):
    """Turn deliberate domain errors into 409s rather than 500s."""
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except MusicmakerError as exc:
            raise HTTPException(409, str(exc)) from exc
    return wrapped


# ---- presets --------------------------------------------------------------

@router.get("/styles")
def styles():
    return {"styles": presets.STYLES, "sfx": presets.SFX_PRESETS}


# ---- create ---------------------------------------------------------------

@router.get("/projects")
def list_projects():
    return {"projects": storage.list_projects()}


@router.post("/projects")
def create_project(body: CreateProject,
                   idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    project = Project(
        title=body.title,
        prompt=body.prompt,
        style=body.style,
        lyrics=body.lyrics,
        instrumental=body.instrumental,
        quality=body.quality,
        thinking=body.thinking,
        vocal_language=body.vocal_language,
        idea_kind="prompt",
    )
    preset = presets.style(body.style)
    project.grid = Grid(
        bpm=body.bpm or (preset or {}).get("bpm", 96),
        key_scale=body.key_scale or (preset or {}).get("key_scale", "A Minor"),
        bars=body.bars,
        # A prompt-only project has nothing to mis-detect, so the grid is
        # exactly what was asked for and needs no second opinion.
        confirmed=True,
    )
    storage.save(project)

    job = jobs.queue.submit(
        "variations",
        lambda report: service.generate_variations(_get(project.id), body.variations, report),
        project_id=project.id,
        idempotency_key=idempotency_key,
    )
    return {"project": project.model_dump(), "job": job.public()}


@router.post("/projects/upload")
async def create_from_upload(
    file: UploadFile = File(...),
    title: str = Form("Untitled"),
    prompt: str = Form(""),
    style: str = Form(""),
    lyrics: str = Form(""),
    bars: int = Form(32),
    variations: int = Form(4),
):
    """Start from a MIDI file or an audio idea (a hum, a loop, a voice memo)."""
    suffix = Path(file.filename or "idea").suffix.lower()
    is_midi = suffix in (".mid", ".midi")

    project = Project(title=title, prompt=prompt, style=style, lyrics=lyrics,
                      idea_kind="midi" if is_midi else "audio")
    project.grid.bars = bars
    storage.save(project)

    d = storage.project_dir(project.id) / "source"
    d.mkdir(parents=True, exist_ok=True)
    raw = d / f"idea{suffix or '.wav'}"
    with raw.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    if is_midi:
        # ACE-Step takes no MIDI, so render it and condition on the audio.
        try:
            rendered = midi.render(raw, d / "idea_rendered.wav")
        except MusicmakerError as exc:
            raise HTTPException(409, f"MIDI render failed: {exc}") from exc
        project.source_audio = f"source/{rendered.name}"
        tempo = midi.read_tempo(raw)
        if tempo:
            project.grid.bpm = tempo
    else:
        project.source_audio = f"source/{raw.name}"

    storage.save(project)

    detected = service.detect_grid(project, storage.project_dir(project.id) / project.source_audio)
    if is_midi and midi.read_tempo(raw):
        project.grid.bpm = midi.read_tempo(raw)
        detected["bpm"] = project.grid.bpm
        detected["reason"] = "tempo read from the MIDI file itself"
        storage.save(project)

    return {
        "project": storage.load(project.id).model_dump(),
        "detected": detected,
        # Deliberately no generation yet: a wrong key here would corrupt every
        # stem downstream, so a human confirms the grid first.
        "next": "POST /api/projects/{id}/grid to confirm, then /variations",
    }


@router.post("/projects/{project_id}/grid")
def confirm_grid(project_id: str, body: ConfirmGrid):
    project = _get(project_id)
    project.grid = Grid(bpm=body.bpm, key_scale=body.key_scale, bars=body.bars,
                        time_sig=body.time_sig, confirmed=True)
    storage.save(project)
    return {"grid": project.grid.model_dump()}


@router.post("/projects/{project_id}/variations")
def make_variations(project_id: str, count: int = 4):
    project = _get(project_id)
    job = jobs.queue.submit(
        "variations",
        lambda report: service.generate_variations(_get(project_id), count, report),
        project_id=project_id,
    )
    return {"job": job.public()}


@router.get("/projects/{project_id}")
def get_project(project_id: str):
    return {"project": _get(project_id).model_dump()}


@router.delete("/projects/{project_id}")
def delete_project(project_id: str):
    if not storage.delete(project_id):
        raise HTTPException(404, "no such project")
    return {"deleted": project_id}


# ---- stems ----------------------------------------------------------------

@router.post("/projects/{project_id}/split")
def split(project_id: str, body: SplitRequest,
          idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    _get(project_id)
    job = jobs.queue.submit(
        "split",
        lambda report: service.split_variation(
            _get(project_id), body.variation_id, body.method, report,
            tier=body.tier, remove_reverb=body.remove_reverb),
        project_id=project_id,
        idempotency_key=idempotency_key,
    )
    return {"job": job.public()}


@router.post("/projects/{project_id}/layers")
def add_layer(project_id: str, body: AddLayerRequest):
    _get(project_id)
    job = jobs.queue.submit(
        "layer",
        lambda report: service.add_layer(
            _get(project_id), body.track_class, body.prompt, body.seed, report,
            start_s=body.start_s, end_s=body.end_s),
        project_id=project_id,
    )
    return {"job": job.public()}


@router.post("/projects/{project_id}/vocals")
def vocals(project_id: str, body: VocalRequest):
    _get(project_id)
    job = jobs.queue.submit(
        "vocals",
        lambda report: service.generate_vocals(
            _get(project_id), body.lyrics, body.vocal_language,
            body.voice_id, body.seed, report),
        project_id=project_id,
    )
    return {"job": job.public()}


# ---- output ---------------------------------------------------------------

@router.get("/projects/{project_id}/mixdown")
def get_mixdown(project_id: str):
    project = _get(project_id)
    path = _guard(service.mixdown)(project)
    return FileResponse(str(path), media_type="audio/wav", filename="mixdown.wav")


@router.post("/projects/{project_id}/export")
def export(project_id: str):
    _get(project_id)
    job = jobs.queue.submit(
        "export",
        lambda report: {"file": str(service.export_zip(_get(project_id), report))},
        project_id=project_id,
    )
    return {"job": job.public()}


@router.get("/projects/{project_id}/export/download")
def download_export(project_id: str):
    path = storage.project_dir(project_id) / f"{project_id}_stems.zip"
    if not path.exists():
        raise HTTPException(404, "no export yet -- POST /export first")
    return FileResponse(str(path), media_type="application/zip",
                        filename=f"{project_id}_stems.zip")


@router.get("/projects/{project_id}/audio/{path:path}")
def get_audio(project_id: str, path: str):
    """Serve any audio file belonging to a project."""
    base = storage.project_dir(project_id).resolve()
    target = (base / path).resolve()
    # Path traversal guard: the resolved target must stay inside the project.
    if not str(target).startswith(str(base)) or not target.exists():
        raise HTTPException(404, "no such audio")
    return FileResponse(str(target), media_type="audio/wav")


# ---- sfx ------------------------------------------------------------------

@router.post("/sfx")
def sfx(body: SfxRequest):
    job = jobs.queue.submit(
        "sfx",
        lambda report: service.generate_sfx(
            body.prompt, body.duration_s, body.variations, body.seed, report),
    )
    return {"job": job.public()}


@router.get("/sfx/{name}")
def get_sfx(name: str):
    from ..config import settings
    path = (settings.data_dir / "sfx" / name).resolve()
    root = (settings.data_dir / "sfx").resolve()
    if not str(path).startswith(str(root)) or not path.exists():
        raise HTTPException(404, "no such sound")
    return FileResponse(str(path), media_type="audio/wav")
