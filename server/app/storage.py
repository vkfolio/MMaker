"""Project persistence: one directory per project, JSON + audio files.

Deliberately filesystem-based. On RunPod this sits on the network volume, so a
pod restart loses nothing, and a project is a directory you can zip and move.
"""

import json
import shutil
import threading
from pathlib import Path

from .config import settings
from .schemas import Project

_lock = threading.RLock()


def project_dir(project_id: str) -> Path:
    return settings.data_dir / "projects" / project_id


def audio_path(project_id: str, rel: str) -> Path:
    return project_dir(project_id) / rel


def save(project: Project) -> Project:
    import time
    with _lock:
        project.updated_at = time.time()
        d = project_dir(project.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "project.json.tmp"
        tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(d / "project.json")     # atomic -- never a half-written file
    return project


def load(project_id: str) -> Project | None:
    path = project_dir(project_id) / "project.json"
    if not path.exists():
        return None
    with _lock:
        return Project.model_validate_json(path.read_text(encoding="utf-8"))


def delete(project_id: str) -> bool:
    d = project_dir(project_id)
    if not d.exists():
        return False
    with _lock:
        shutil.rmtree(d)
    return True


def list_projects() -> list[dict]:
    root = settings.data_dir / "projects"
    if not root.exists():
        return []
    out = []
    for path in root.glob("*/project.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out.append({
            "id": data.get("id"),
            "title": data.get("title"),
            "updated_at": data.get("updated_at", 0),
            "bpm": data.get("grid", {}).get("bpm"),
            "key_scale": data.get("grid", {}).get("key_scale"),
            "stems": len(data.get("stems", [])),
        })
    return sorted(out, key=lambda p: p["updated_at"], reverse=True)
