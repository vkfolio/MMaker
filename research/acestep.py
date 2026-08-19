"""Thin client for the ACE-Step 1.5 REST server.

The server is documented as exposing three endpoints on :8001 --
POST /release_task, POST /query_result, GET /v1/audio -- but the exact
response field names are not fully pinned down in the public docs. This
client is therefore deliberately tolerant: it dumps every raw response to
out/_raw/ so the first real run against a pod tells us the true schema
instead of us guessing. Tighten it once probe.py has run.
"""

import json
import os
import time
import uuid
from pathlib import Path

import requests

BASE_URL = os.environ.get("ACESTEP_URL", "http://localhost:8001").rstrip("/")
OUT = Path(__file__).parent / "out"
RAW = OUT / "_raw"

# Track classes ACE-Step 1.5 recognises, per the model docs.
TRACK_CLASSES = [
    "vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard",
    "percussion", "strings", "synth", "fx", "brass", "woodwinds",
]

TASK_TYPES = ["text2music", "cover", "repaint", "extract", "lego", "complete"]


class AceStepError(RuntimeError):
    pass


def _dump(tag, payload):
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{int(time.time()*1000)}_{tag}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _find_task_id(resp):
    """The docs don't name the task-id field. Accept the usual suspects."""
    if isinstance(resp, str):
        return resp
    for key in ("task_id", "taskId", "id", "task", "uuid", "request_id"):
        if isinstance(resp, dict) and resp.get(key):
            return resp[key]
    if isinstance(resp, dict) and len(resp) == 1:
        only = next(iter(resp.values()))
        if isinstance(only, (str, int)):
            return only
    raise AceStepError(f"could not locate a task id in response: {resp!r}")


def _find_audio_ref(resp):
    """Locate the generated-audio reference in a /query_result payload."""
    if not isinstance(resp, dict):
        return None
    for key in ("file", "audio", "url", "audio_path", "output"):
        val = resp.get(key)
        if isinstance(val, str) and val:
            return val
    # Some servers nest results under data/result/results.
    for key in ("data", "result", "results"):
        nested = resp.get(key)
        if isinstance(nested, dict):
            found = _find_audio_ref(nested)
            if found:
                return found
        if isinstance(nested, list) and nested:
            found = _find_audio_ref(nested[0])
            if found:
                return found
    return None


def release_task(params, src_audio_path=None, timeout=60):
    """Submit a generation task. src_audio_path is uploaded as multipart."""
    url = f"{BASE_URL}/release_task"
    if src_audio_path:
        with open(src_audio_path, "rb") as fh:
            files = {"src_audio": (Path(src_audio_path).name, fh, "audio/wav")}
            # Multipart form fields must be strings.
            form = {k: str(v) for k, v in params.items() if v is not None}
            resp = requests.post(url, data=form, files=files, timeout=timeout)
    else:
        resp = requests.post(url, json=params, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    _dump(f"release_{params.get('task_type', 'unknown')}", body)
    return _find_task_id(body)


def query_result(task_id, timeout=30):
    resp = requests.post(f"{BASE_URL}/query_result", json={"task_id": task_id},
                         timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def download_audio(audio_ref, dest):
    """audio_ref is typically '/v1/audio?path=...' relative to the server."""
    url = audio_ref if audio_ref.startswith("http") else f"{BASE_URL}{audio_ref}"
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest


def run(params, dest, src_audio_path=None, poll=5.0, max_wait=1800):
    """Submit, poll to completion, download. Returns (path, metas, seconds)."""
    started = time.time()
    task_id = release_task(params, src_audio_path=src_audio_path)
    print(f"  submitted task {task_id} ({params.get('task_type')})", flush=True)

    while True:
        elapsed = time.time() - started
        if elapsed > max_wait:
            raise AceStepError(f"task {task_id} exceeded {max_wait}s")
        body = query_result(task_id)
        audio_ref = _find_audio_ref(body)
        if audio_ref:
            _dump(f"result_{params.get('task_type', 'unknown')}", body)
            path = download_audio(audio_ref, dest)
            metas = body.get("metas") or {}
            print(f"  -> {path.name} in {elapsed:.1f}s  metas={metas}", flush=True)
            return path, metas, elapsed
        status = body.get("status") or body.get("state") or "pending"
        if str(status).lower() in ("failed", "error"):
            _dump("failure", body)
            raise AceStepError(f"task {task_id} failed: {body!r}")
        print(f"  ... {status} ({elapsed:.0f}s)", flush=True)
        time.sleep(poll)


def new_run_dir(name):
    d = OUT / f"{name}_{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d
