"""Per-stem editing: repaint, extend, vary, voice, mix state, version history."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import jobs, service, storage
from ..schemas import (ExtendRequest, MixState, Project, RepaintRequest,
                       VaryRequest, VoiceRequest)

router = APIRouter(prefix="/api/projects/{project_id}/stems", tags=["stems"])


def _get(project_id: str) -> Project:
    project = storage.load(project_id)
    if not project:
        raise HTTPException(404, f"no such project: {project_id}")
    return project


def _stem(project: Project, stem_id: str):
    stem = project.stem(stem_id)
    if not stem:
        raise HTTPException(404, f"no such stem: {stem_id}")
    if not stem.version:
        raise HTTPException(409, "stem has no audio yet")
    return stem


@router.get("")
def list_stems(project_id: str):
    return {"stems": [s.model_dump() for s in _get(project_id).stems]}


@router.post("/{stem_id}/repaint")
def repaint(project_id: str, stem_id: str, body: RepaintRequest):
    project = _get(project_id)
    _stem(project, stem_id)
    try:
        start_s, end_s = body.resolve(project.grid)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    def work(report):
        proj = _get(project_id)
        return service.repaint_stem(proj, proj.stem(stem_id), start_s, end_s,
                                    body.prompt, body.seed, report)

    return {"job": jobs.queue.submit("repaint", work, project_id).public()}


@router.post("/{stem_id}/extend")
def extend(project_id: str, stem_id: str, body: ExtendRequest):
    _stem(_get(project_id), stem_id)

    def work(report):
        project = _get(project_id)
        return service.extend_stem(project, project.stem(stem_id),
                                   body.bars, body.prompt, report)

    return {"job": jobs.queue.submit("extend", work, project_id).public()}


@router.post("/{stem_id}/vary")
def vary(project_id: str, stem_id: str, body: VaryRequest):
    _stem(_get(project_id), stem_id)

    def work(report):
        project = _get(project_id)
        return service.vary_stem(project, project.stem(stem_id),
                                 body.seed, body.prompt, report)

    return {"job": jobs.queue.submit("vary", work, project_id).public()}


@router.post("/{stem_id}/voice")
def voice(project_id: str, stem_id: str, body: VoiceRequest):
    _stem(_get(project_id), stem_id)

    def work(report):
        project = _get(project_id)
        return service.apply_voice(project, project.stem(stem_id),
                                   body.voice_id, report)

    return {"job": jobs.queue.submit("voice", work, project_id).public()}


@router.post("/{stem_id}/isolate")
def isolate(project_id: str, stem_id: str):
    _stem(_get(project_id), stem_id)

    def work(report):
        project = _get(project_id)
        return service.isolate_vocals(project, project.stem(stem_id), report)

    return {"job": jobs.queue.submit("isolate", work, project_id).public()}


# ---- non-generative state (instant, no job) -------------------------------

@router.patch("/{stem_id}/mix")
def set_mix(project_id: str, stem_id: str, body: MixState):
    project = _get(project_id)
    stem = project.stem(stem_id)
    if not stem:
        raise HTTPException(404, "no such stem")
    for field in ("gain_db", "pan", "muted", "soloed"):
        value = getattr(body, field)
        if value is not None:
            setattr(stem, field, value)
    storage.save(project)
    return {"stem": stem.model_dump()}


@router.post("/{stem_id}/version/{index}")
def select_version(project_id: str, stem_id: str, index: int):
    """Switch which take is current. Nothing is deleted -- undo is free."""
    project = _get(project_id)
    stem = project.stem(stem_id)
    if not stem:
        raise HTTPException(404, "no such stem")
    if not 0 <= index < len(stem.versions):
        raise HTTPException(400, f"version index out of range (0..{len(stem.versions) - 1})")
    stem.current = index
    storage.save(project)
    return {"stem": stem.model_dump()}


@router.delete("/{stem_id}")
def delete_stem(project_id: str, stem_id: str):
    project = _get(project_id)
    before = len(project.stems)
    project.stems = [s for s in project.stems if s.id != stem_id]
    if len(project.stems) == before:
        raise HTTPException(404, "no such stem")
    storage.save(project)
    return {"deleted": stem_id}
