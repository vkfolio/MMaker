"""Fetching model weights on first boot.

The image carries code and virtualenvs; the weights live on the network volume.
That split is deliberate. Baked-in weights would make the image ~35 GB, and the
volume has to be mounted at /workspace anyway -- where it would *hide* anything
the image had put there. So: code at /opt (never hidden), weights at /workspace
(persistent across pods, fetched once).

`/health` reports progress while this runs, and the UI holds a boot screen until
it says ready. Every later boot on the same volume is instant.

Progress is measured by watching the directory grow, because neither
huggingface_hub nor the ACE-Step downloader offers a byte callback. It is an
estimate against a hardcoded expected size, which is why `percent` refuses to
reach 100 until the download actually returns -- a bar that sits at 100% while
work continues is worse than one that sits at 97%.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings


@dataclass
class Item:
    name: str
    label: str
    repo: str
    dest: Path
    approx_gb: float
    # A file that only exists when the download finished. Directory presence is
    # not enough: an interrupted fetch leaves a partial tree that looks done.
    marker: str
    state: str = "pending"        # pending | downloading | ready | error
    message: str = ""
    bytes_now: int = 0

    @property
    def percent(self) -> int:
        if self.state == "ready":
            return 100
        if self.approx_gb <= 0:
            return 0
        got = self.bytes_now / (self.approx_gb * 1e9)
        return max(0, min(97, int(got * 100)))

    def public(self) -> dict:
        return {"name": self.name, "label": self.label, "state": self.state,
                "percent": self.percent, "approx_gb": self.approx_gb,
                "message": self.message}


@dataclass
class State:
    items: list[Item] = field(default_factory=list)
    status: str = "starting"      # starting | downloading | ready | error
    message: str = ""

    def public(self) -> dict:
        return {"status": self.status, "message": self.message,
                "models": [i.public() for i in self.items]}


state = State()
_started = False


def _items() -> list[Item]:
    ace = settings.acestep_checkpoints
    soulx = settings.soulx_weights
    return [
        Item("acestep-core", "ACE-Step core (VAE, text encoder, LM)",
             "ACE-Step/Ace-Step1.5", ace, 10.0, marker="vae/config.json"),
        Item("acestep-xl-sft", "ACE-Step 1.5 XL (4B)",
             "ACE-Step/acestep-v15-xl-sft", ace / "acestep-v15-xl-sft", 19.5,
             marker="config.json"),
        Item("soulx", "SoulX-Singer",
             "Soul-AILab/SoulX-Singer", soulx / "SoulX-Singer", 5.7,
             marker="model.pt"),
    ]


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _watch(item: Item, stop: threading.Event) -> None:
    while not stop.is_set():
        item.bytes_now = _dir_bytes(item.dest)
        stop.wait(2.0)


def _download(item: Item) -> None:
    if (item.dest / item.marker).exists():
        item.state = "ready"
        return

    item.state = "downloading"
    item.dest.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(item, stop), daemon=True)
    watcher.start()
    try:
        # The CLI rather than the Python API: it handles resume, retries and
        # xet transport itself, and a subprocess that dies cannot take the
        # service down with it.
        proc = subprocess.run(
            ["hf", "download", item.repo, "--local-dir", str(item.dest)],
            capture_output=True, text=True, timeout=7200,
            env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
        )
        if proc.returncode != 0 or not (item.dest / item.marker).exists():
            item.state = "error"
            item.message = (proc.stderr or proc.stdout or "download failed")[-300:]
            return
        item.state = "ready"
    except subprocess.TimeoutExpired:
        item.state = "error"
        item.message = "timed out after 2 hours"
    except Exception as exc:                                    # noqa: BLE001
        item.state = "error"
        item.message = str(exc)[:300]
    finally:
        stop.set()
        item.bytes_now = _dir_bytes(item.dest)


def _run() -> None:
    for item in state.items:
        _download(item)
    bad = [i for i in state.items if i.state == "error"]
    if bad:
        state.status = "error"
        state.message = f"{bad[0].label}: {bad[0].message}"
    else:
        state.status = "ready"
        state.message = ""


def start_background() -> None:
    """Kick the fetch off on a daemon thread and return immediately.

    Reset synchronously, before the thread starts: a client polling /health in
    the meantime must not see the previous run's terminal status and conclude
    boot is already finished.
    """
    global _started
    if _started:
        return
    _started = True

    state.items = _items()
    if settings.stub_models:
        for item in state.items:
            item.state = "ready"
        state.status = "ready"
        return

    missing = [i for i in state.items if not (i.dest / i.marker).exists()]
    state.status = "ready" if not missing else "downloading"
    for item in state.items:
        if (item.dest / item.marker).exists():
            item.state = "ready"

    if missing:
        threading.Thread(target=_run, daemon=True).start()


def ready() -> bool:
    return state.status == "ready"
