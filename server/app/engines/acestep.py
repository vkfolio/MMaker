"""ACE-Step 1.5 adapter.

Talks to the ACE-Step REST server (default :8001). The response schema is not
fully pinned down in the public docs, so the client stays tolerant and records
every raw payload -- the first real pod run tells us the true shape rather than
us guessing. See research/acestep.py, which proved this path.

Track classes ACE-Step recognises:
  vocals, backing_vocals, drums, bass, guitar, keyboard, percussion,
  strings, synth, fx, brass, woodwinds
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

from ..schemas import Grid
from .base import EngineError, NotSupported

DEFAULT_URL = os.environ.get("ACESTEP_URL", "http://127.0.0.1:8001")
POLL_S = float(os.environ.get("ACESTEP_POLL_S", "3"))
MAX_WAIT_S = float(os.environ.get("ACESTEP_MAX_WAIT_S", "1800"))


AUDIO_RE = re.compile(r"\.(mp3|wav|flac|ogg|m4a)(\?|$)", re.I)


def _deep_audio(obj, _depth=0):
    """Find an audio reference anywhere in the payload.

    ACE-Step wraps results in a `data` envelope and the field naming is not
    documented, so matching on key names is fragile. Matching on the *value*
    -- anything that looks like an audio path or an /v1/audio link -- survives
    schema changes we cannot see in advance.
    """
    if _depth > 6:
        return None
    if isinstance(obj, str):
        text = obj.strip()
        # ACE-Step nests the real payload as a JSON *string* under "result",
        # so parse it and keep looking rather than matching the blob itself.
        if text[:1] in ("[", "{"):
            try:
                return _deep_audio(json.loads(text), _depth + 1)
            except (ValueError, TypeError):
                return None
        if len(text) > 2048:
            return None
        if text.startswith("/v1/audio") or AUDIO_RE.search(text):
            return text
        return None
    if isinstance(obj, dict):
        for value in obj.values():
            found = _deep_audio(value, _depth + 1)
            if found:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = _deep_audio(item, _depth + 1)
            if found:
                return found
    return None


def _envelope(resp):
    """ACE-Step returns {"data": ..., "code": 200}. Unwrap one level."""
    if isinstance(resp, dict) and "code" in resp and "data" in resp:
        return resp["data"]
    return resp


def _find(resp, keys):
    if isinstance(resp, str):
        return resp
    if not isinstance(resp, dict):
        return None
    for key in keys:
        val = resp.get(key)
        if isinstance(val, (str, int)) and val != "":
            return val
    for key in ("data", "result", "results"):
        nested = resp.get(key)
        if isinstance(nested, dict):
            found = _find(nested, keys)
            if found:
                return found
        if isinstance(nested, list) and nested:
            found = _find(nested[0], keys)
            if found:
                return found
    return None


class AceStepEngine:
    name = "acestep"

    def __init__(self, base_url: str = DEFAULT_URL, debug_dir: Path | None = None):
        self.base_url = base_url.rstrip("/")
        self.debug_dir = Path(debug_dir) if debug_dir else None

    # -- plumbing ----------------------------------------------------------

    def _dump(self, tag, payload):
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / f"{int(time.time() * 1000)}_{tag}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _submit(self, params: dict, src_audio: Path | None = None) -> str:
        url = f"{self.base_url}/release_task"
        clean = {k: v for k, v in params.items() if v is not None}
        try:
            if src_audio:
                with open(src_audio, "rb") as fh:
                    files = {"src_audio": (Path(src_audio).name, fh, "audio/wav")}
                    form = {k: str(v) for k, v in clean.items()}
                    resp = requests.post(url, data=form, files=files, timeout=120)
            else:
                resp = requests.post(url, json=clean, timeout=120)
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            raise EngineError(f"ACE-Step unreachable at {self.base_url}: {exc}") from exc

        self._dump(f"release_{params.get('task_type', 'x')}", body)
        task_id = _find(_envelope(body), ("task_id", "taskId", "id", "task", "uuid", "request_id"))
        if not task_id:
            raise EngineError(f"no task id in ACE-Step response: {body!r}")
        return str(task_id)

    def _await(self, task_id: str, dest: Path, report=None) -> Path:
        started = time.time()
        while True:
            elapsed = time.time() - started
            if elapsed > MAX_WAIT_S:
                raise EngineError(f"ACE-Step task {task_id} exceeded {MAX_WAIT_S}s")
            try:
                # The route reads body["task_id_list"] (see
                # acestep/api/http/query_result_route.py). Sending "task_id"
                # parses as an empty list, so results are never returned.
                resp = requests.post(f"{self.base_url}/query_result",
                                     json={"task_id_list": [task_id]}, timeout=60)
                resp.raise_for_status()
                body = resp.json()
            except requests.RequestException as exc:
                raise EngineError(f"ACE-Step query failed: {exc}") from exc

            inner = _envelope(body)
            # /query_result drains: a finished result is returned once and then
            # the queue is empty. Never discard a non-empty payload unexamined.
            if inner not in (None, [], {}):
                self._dump("result", body)

            ref = _deep_audio(inner)
            if ref:
                return self._download(str(ref), dest)

            err = body.get("error") if isinstance(body, dict) else None
            status_src = inner if isinstance(inner, dict) else (body if isinstance(body, dict) else {})
            status = str(status_src.get("status") or status_src.get("state") or "pending").lower()
            if err or status in ("failed", "error"):
                self._dump("failure", body)
                raise EngineError(f"ACE-Step task failed: {body!r}")
            if report:
                # No real progress signal from the server, so report elapsed time
                # against a nominal ceiling rather than inventing a percentage.
                report(min(0.95, elapsed / 180.0), f"rendering ({int(elapsed)}s)")
            time.sleep(POLL_S)

    def _download(self, ref: str, dest: Path) -> Path:
        if ref.startswith("http"):
            url = ref
        elif ref.startswith("/v1/audio") or ref.startswith("?"):
            url = f"{self.base_url}{ref}"
        else:
            # A server-side filesystem path: fetch it through the audio route.
            from urllib.parse import quote
            url = f"{self.base_url}/v1/audio?path={quote(str(ref), safe='')}"
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)

        # ACE-Step returns MP3 regardless of the name we asked for, but the rest
        # of the pipeline -- mixdown, separation, export -- reads WAV. Transcode
        # once here so nothing downstream has to care.
        head = resp.content[:3]
        if dest.suffix.lower() == ".wav" and head in (b"ID3", b"ÿû", b"ÿó"):
            self._to_wav(dest)
        return dest

    @staticmethod
    def _to_wav(path: Path) -> Path:
        """Transcode in place via ffmpeg, which ships in the pod image."""
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            raise EngineError(
                f"{path.name} is MP3 but ffmpeg is missing, so it cannot be "
                "converted to the WAV the pipeline expects")
        tmp = path.with_suffix(".mp3.tmp")
        path.rename(tmp)
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                 "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(path)],
                capture_output=True, text=True, timeout=300)
            if proc.returncode != 0 or not path.exists():
                raise EngineError(f"ffmpeg failed: {proc.stderr.strip()[:300]}")
        finally:
            tmp.unlink(missing_ok=True)
        return path

    def _grid_params(self, grid: Grid) -> dict:
        return {
            "bpm": grid.bpm,
            "key_scale": grid.key_scale,
            "audio_duration": round(grid.duration_s, 3),
        }

    def _run(self, params, dest, src_audio=None, report=None):
        task_id = self._submit(params, src_audio)
        return self._await(task_id, dest, report)

    # -- Engine interface --------------------------------------------------

    def generate(self, dest, prompt, grid, seed, lyrics="", vocal_language="en",
                 src_audio=None, report=None):
        params = {
            **self._grid_params(grid),
            "task_type": "cover" if src_audio else "text2music",
            "prompt": prompt,
            "lyrics": lyrics or None,
            "vocal_language": vocal_language,
            "seed": seed,
        }
        return self._run(params, dest, src_audio, report)

    def layer(self, dest, track_class, src_audio, prompt, grid, seed,
              lyrics="", vocal_language="en", report=None):
        params = {
            **self._grid_params(grid),
            "task_type": "lego",
            "track_name": track_class,
            "prompt": prompt,
            "lyrics": lyrics or None,
            "vocal_language": vocal_language,
            "seed": seed,
        }
        return self._run(params, dest, src_audio, report)

    def repaint(self, dest, src_audio, start_s, end_s, prompt, grid, seed, report=None):
        params = {
            **self._grid_params(grid),
            "task_type": "repaint",
            "prompt": prompt,
            "repainting_start": round(start_s, 3),
            "repainting_end": round(end_s, 3),
            "chunk_mask_mode": "explicit",
            "seed": seed,
        }
        return self._run(params, dest, src_audio, report)

    def extend(self, dest, src_audio, added_s, prompt, grid, seed, report=None):
        params = {
            **self._grid_params(grid),
            "task_type": "complete",
            "prompt": prompt,
            "audio_duration": round(grid.duration_s + added_s, 3),
            "seed": seed,
        }
        return self._run(params, dest, src_audio, report)

    def extract(self, dest, src_audio, track_class, grid, seed=0, report=None):
        """Generative stem separation. Demucs is usually the better choice."""
        params = {
            **self._grid_params(grid),
            "task_type": "extract",
            "track_name": track_class,
            "seed": seed,
        }
        return self._run(params, dest, src_audio, report)

    def separate(self, src_audio, out_dir):
        raise NotSupported("use DemucsEngine for separation, or acestep.extract per track")

    def convert_voice(self, dest, src_audio, reference):
        raise NotSupported("ACE-Step has no voice library; use Seed-VC")

    def sfx(self, dest, prompt, duration_s, seed):
        raise NotSupported("use Stable Audio 3 for sound effects")

    def ping(self) -> bool:
        try:
            requests.get(f"{self.base_url}/openapi.json", timeout=10)
            return True
        except requests.RequestException:
            return False
