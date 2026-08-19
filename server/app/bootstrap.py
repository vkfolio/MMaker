"""First-boot model download, with progress that /health can report.

Runs in a background thread at startup so the HTTP server answers immediately --
the UI polls /health and shows a progress bar rather than hanging on a request
that takes twenty minutes.
"""

import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .config import settings


@dataclass
class ModelState:
    name: str
    label: str
    kind: str
    repo_id: str
    approx_gb: float
    phase: int
    purpose: str
    enabled: bool
    status: str = "pending"       # pending | downloading | ready | skipped | error
    downloaded_gb: float = 0.0
    error: str | None = None

    @property
    def percent(self):
        if self.status == "ready":
            return 100.0
        if not self.approx_gb:
            return 0.0
        return min(99.0, round(self.downloaded_gb / self.approx_gb * 100, 1))

    def public(self):
        d = asdict(self)
        d["percent"] = self.percent
        return d


@dataclass
class BootstrapState:
    status: str = "starting"       # starting | downloading | ready | error
    models: list = field(default_factory=list)
    message: str = ""
    started_at: float = field(default_factory=time.time)

    def public(self):
        return {
            "status": self.status,
            "message": self.message,
            "uptime_s": round(time.time() - self.started_at, 1),
            "models": [m.public() for m in self.models],
        }


state = BootstrapState()
_lock = threading.Lock()


def load_registry(path=None):
    path = Path(path or settings.registry_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    models = []
    for entry in raw.get("models", []):
        models.append(ModelState(
            name=entry["name"],
            label=entry.get("label", entry["name"]),
            kind=entry.get("kind", "huggingface"),
            repo_id=entry.get("repo_id", ""),
            approx_gb=float(entry.get("approx_gb", 0)),
            phase=int(entry.get("phase", 0)),
            purpose=entry.get("purpose", ""),
            enabled=bool(entry.get("enabled", False)),
        ))
    return models


def _target_dir(model):
    return settings.models_dir / model.name


def _download_stub(model):
    """Fake a download so the boot flow can be exercised without a GPU."""
    steps = 10
    for i in range(steps):
        time.sleep(0.05)
        with _lock:
            model.downloaded_gb = model.approx_gb * (i + 1) / steps
    _target_dir(model).mkdir(parents=True, exist_ok=True)
    (_target_dir(model) / "STUB").write_text("stub\n", encoding="utf-8")


def _download_huggingface(model):
    from huggingface_hub import snapshot_download

    dest = _target_dir(model)
    dest.mkdir(parents=True, exist_ok=True)

    # snapshot_download does not stream progress back to us, so we watch the
    # directory grow from a side thread. Approximate, but it is all the UI needs.
    done = threading.Event()

    def watch():
        while not done.wait(1.0):
            total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
            with _lock:
                model.downloaded_gb = total / 1e9

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        snapshot_download(repo_id=model.repo_id, local_dir=str(dest),
                          local_dir_use_symlinks=False)
    finally:
        done.set()
        watcher.join(timeout=2)


def _download_torchhub(model):
    """Demucs and friends fetch their own weights through their own APIs."""
    if model.name == "demucs":
        from demucs.pretrained import get_model
        get_model(model.repo_id)
    else:
        raise RuntimeError(f"no torchhub handler for {model.name}")


KNOWN_KINDS = {"huggingface", "torchhub"}


def _is_present(model):
    dest = _target_dir(model)
    return dest.exists() and any(dest.iterdir())


def run():
    """Download every enabled model. Safe to call once at startup."""
    settings.ensure_dirs()
    with _lock:
        state.models = load_registry()
        state.status = "downloading"

    for model in state.models:
        if not model.enabled:
            model.status = "skipped"
            continue

        if model.kind == "huggingface" and _is_present(model):
            model.status = "ready"
            model.downloaded_gb = model.approx_gb
            continue

        with _lock:
            model.status = "downloading"
            state.message = f"downloading {model.label}"
        try:
            # Validate the kind even in stub mode -- otherwise a typo in
            # models.yaml sails through local testing and only fails on a pod.
            if model.kind not in KNOWN_KINDS:
                raise RuntimeError(f"unknown model kind: {model.kind}")
            if settings.stub_models:
                _download_stub(model)
            elif model.kind == "huggingface":
                _download_huggingface(model)
            elif model.kind == "torchhub":
                _download_torchhub(model)
        except Exception as exc:                      # noqa: BLE001
            model.status = "error"
            model.error = f"{type(exc).__name__}: {exc}"
            continue
        model.status = "ready"
        model.downloaded_gb = model.approx_gb

    required = [m for m in state.models if m.enabled]
    failed = [m for m in required if m.status != "ready"]
    with _lock:
        if failed:
            state.status = "error"
            state.message = "failed: " + ", ".join(m.label for m in failed)
        else:
            state.status = "ready"
            state.message = "engine ready"


def start_background():
    # Reset synchronously, before the thread starts. Otherwise a caller polling
    # /health can observe the previous run's terminal status and conclude that
    # boot is already finished.
    with _lock:
        state.status = "starting"
        state.message = ""
        state.models = []
    thread = threading.Thread(target=run, name="bootstrap", daemon=True)
    thread.start()
    return thread
