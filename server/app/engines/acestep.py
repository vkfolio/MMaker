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

# Quality tiers, from ACE-Step's own model-selection guide.
#
# turbo models are distilled for 8 steps and ignore CFG; base models want
# 32-64 steps with guidance and run two forward passes, so they need roughly
# double the VRAM. The XL pair is 4B rather than 0.6B. "ultra" is what to use
# when the GPU is large and the wait does not matter.
#
# IMPORTANT -- what a pod actually has:
#   The base weights repo (ACE-Step/Ace-Step1.5) ships exactly two things that
#   matter here: acestep-v15-turbo and acestep-5Hz-lm-1.7B. Every other
#   checkpoint below lives in its own Hugging Face repo (ACE-Step/<name>) and
#   must be downloaded deliberately -- the XL DiTs are ~20 GB each.
#
#   Asking for a checkpoint that is not present does NOT fail. ACE-Step falls
#   back to its default and renders anyway, so a tier can look like it worked
#   while producing turbo output. That is exactly what happened here: all three
#   tiers timed identically because all three were the same model.
#
#   `installed` marks what the stock download provides. /api/engine reports
#   which tiers would fall back on this particular pod.
QUALITY = {
    "fast":  {"model": "acestep-v15-turbo",    "lm": "acestep-5Hz-lm-1.7B",
              "inference_steps": 8,  "guidance_scale": None, "vram_gb": 6,
              "installed": True},
    "high":  {"model": "acestep-v15-xl-turbo", "lm": "acestep-5Hz-lm-1.7B",
              "inference_steps": 8,  "guidance_scale": None, "vram_gb": 14,
              "installed": False},
    "ultra": {"model": "acestep-v15-xl-base",  "lm": "acestep-5Hz-lm-4B",
              "inference_steps": 50, "guidance_scale": 7.0,  "vram_gb": 30,
              "installed": False},
}

# What each checkpoint needs, as opposed to what a tier wants.
#
# These are properties of the model, not of the tier, and keeping them apart is
# what makes a fallback safe. A turbo checkpoint is distilled for 8 steps and
# overrides CFG; a base checkpoint wants ~50 steps with guidance and produces
# mush at 8. Sending one model's settings to another is worse than either tier:
# it happened here, where `high` fell back to xl-base and kept asking for 8
# steps, so "better model" rendered worse than turbo.
MODEL_SETTINGS = {
    "acestep-v15-turbo":    {"inference_steps": 8,  "guidance_scale": None},
    "acestep-v15-xl-turbo": {"inference_steps": 8,  "guidance_scale": None},
    "acestep-v15-xl-base":  {"inference_steps": 50, "guidance_scale": 7.0},
    "acestep-v15-base":     {"inference_steps": 50, "guidance_scale": 7.0},
}

# Which task types each kind of checkpoint implements.
#
# Mirrors acestep/constants.py TASK_TYPES_TURBO / TASK_TYPES_BASE. A turbo
# checkpoint is distilled and implements only four of the seven tasks, so
# `lego` (add a layer), `complete` (extend) and `extract` are unavailable on a
# pod that carries turbo alone -- which is the ordinary laptop install.
#
# Getting this wrong is not a soft failure. Asking turbo for `lego` is accepted
# by the HTTP API and fails deep in the render, minutes later, which reads as a
# broken feature rather than an absent one.
TASKS_TURBO = ("text2music", "repaint", "cover", "cover-nofsq")
TASKS_BASE = TASKS_TURBO + ("extract", "lego", "complete")

# Best first. A tier whose checkpoint is absent resolves to the best one that
# is present, and takes that model's settings with it.
MODEL_PREFERENCE = ("acestep-v15-xl-base", "acestep-v15-xl-turbo", "acestep-v15-turbo")

# Extra checkpoints a pod can opt into, with their download sizes, so the
# operator can decide rather than discover.
OPTIONAL_WEIGHTS = {
    "acestep-v15-xl-turbo": 19.9,
    "acestep-v15-xl-base":  19.9,
    "acestep-5Hz-lm-4B":     8.4,
    "acestep-5Hz-lm-0.6B":   1.4,
}

# `high` is the better default only once its weights exist. Until then the
# honest default is the tier this pod can actually render.
DEFAULT_QUALITY = os.environ.get("ACESTEP_QUALITY", "fast").strip().lower()


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


def _seed_params(seed) -> dict:
    """ACE-Step randomises unless told not to.

    use_random_seed defaults to true, so passing a seed alone changes nothing --
    and every version we record claims a seed that would not reproduce it.
    """
    return {"seed": int(seed), "use_random_seed": False}


META_KEYS = ("model", "lm_model_path", "inference_steps", "guidance_scale",
             "generation_info", "infer_step")


def _deep_meta(obj, _depth=0) -> dict:
    """Pull whatever the server reports about how it rendered."""
    if _depth > 6:
        return {}
    if isinstance(obj, str):
        text = obj.strip()
        if text[:1] in ("[", "{"):
            try:
                return _deep_meta(json.loads(text), _depth + 1)
            except (ValueError, TypeError):
                return {}
        return {}
    found = {}
    if isinstance(obj, dict):
        for key in META_KEYS:
            if key in obj and obj[key] not in (None, ""):
                found[key] = obj[key]
        for value in obj.values():
            found = {**_deep_meta(value, _depth + 1), **found}
    if isinstance(obj, list):
        for item in obj:
            found = {**_deep_meta(item, _depth + 1), **found}
    return found


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
        self.last_meta: dict = {}

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
            self._note_engine_meta(inner)
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
                # ACE-Step exposes no progress signal. elapsed/180 looked like
                # information and was not; report indeterminate and show time.
                report(-1, f"rendering ({int(elapsed)}s)")
            time.sleep(POLL_S)

    def _note_engine_meta(self, inner):
        """Remember what the server reported rendering with.

        Asking for a model the pod does not have makes ACE-Step fall back
        silently, which shows up only as a suspiciously fast render. Recording
        the reported model and step count turns that into visible data.
        """
        found = _deep_meta(inner)
        if found:
            # Merge, do not replace. What we asked for was recorded at request
            # time and is the half that identifies the model; the reply only
            # adds timing. Overwriting would throw away the useful half.
            self.last_meta = {**self.last_meta, **found}

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
        is_mp3 = head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF
                                       and head[1] in (0xFB, 0xF3, 0xF2, 0xE3))
        if dest.suffix.lower() == ".wav" and is_mp3:
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
            "time_signature": grid.time_sig.split("/")[0],
            "audio_duration": round(grid.duration_s, 3),
        }

    @staticmethod
    def _resolve_model(wanted: str, present: list[str]) -> str:
        """Which checkpoint a request for `wanted` will actually render on."""
        if not present or wanted in present:
            return wanted
        for candidate in MODEL_PREFERENCE:
            if candidate in present:
                return candidate
        return wanted

    @staticmethod
    def checkpoint_root() -> str:
        root = os.environ.get("ACESTEP_CHECKPOINT_DIR")
        if not root:
            base = os.environ.get("ACESTEP_DIR", "/workspace/ACE-Step-1.5")
            root = os.path.join(base, "checkpoints")
        return root

    @classmethod
    def model_tasks(cls, model: str) -> list[str]:
        """The task types one checkpoint implements.

        Read from the checkpoint's own config.json, the same way ACE-Step
        decides it (`is_turbo`). Guessing from the model name would be wrong
        for any checkpoint not in our table, and a wrong guess here is what
        turns "this pod cannot do that" into a render that fails minutes in.

        An unreadable config is reported as turbo -- the smaller claim. Over-
        claiming offers a button that fails; under-claiming disables one that
        might have worked, and says why.
        """
        path = os.path.join(cls.checkpoint_root(), model, "config.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                is_turbo = bool(json.load(handle).get("is_turbo", False))
        except (OSError, ValueError):
            return list(TASKS_TURBO)
        return list(TASKS_TURBO if is_turbo else TASKS_BASE)

    @classmethod
    def installed_models(cls) -> list[str]:
        """Checkpoints actually on disk.

        ACE-Step runs in this container, so its checkpoint directory is readable
        here -- which is a far better source of truth than /v1/models, whose
        list is only what has been lazily *loaded* and is empty until the first
        render.
        """
        root = cls.checkpoint_root()
        try:
            # Skip dot-directories: .cache sits alongside the checkpoints and
            # is not one, and listing it invites someone to read it as a model.
            return sorted(name for name in os.listdir(root)
                          if not name.startswith(".")
                          and os.path.isdir(os.path.join(root, name)))
        except OSError:
            return []

    def _quality_params(self, quality: str | None = None) -> dict:
        name = (quality or DEFAULT_QUALITY)
        tier = dict(QUALITY.get(name, QUALITY["high"]))

        # If the tier's checkpoint is not installed, ACE-Step silently uses its
        # primary model. Rather than let that happen with the wrong step count,
        # resolve it here and carry the replacement's settings.
        present = self.installed_models()
        fell_back_from = None
        if present and tier["model"] not in present:
            for candidate in MODEL_PREFERENCE:
                if candidate in present:
                    fell_back_from = tier["model"]
                    tier["model"] = candidate
                    tier.update(MODEL_SETTINGS.get(candidate, {}))
                    break
        if present and tier["lm"] not in present:
            for candidate in ("acestep-5Hz-lm-4B", "acestep-5Hz-lm-1.7B",
                              "acestep-5Hz-lm-0.6B"):
                if candidate in present:
                    tier["lm"] = candidate
                    break

        # Record what was asked for, now, rather than hoping the response says.
        # It does not: ACE-Step reports only timing back, so scraping the reply
        # can never answer "which model made this take". Without that written
        # down, a silent fallback is indistinguishable from a subtle one -- the
        # exact confusion that made three identical tiers hard to spot.
        self.last_meta = {
            "tier": name,
            "requested_model": tier["model"],
            "requested_lm": tier["lm"],
            "inference_steps": tier["inference_steps"],
            "guidance_scale": tier["guidance_scale"],
            "fell_back_from": fell_back_from,
        }

        params = {
            "model": tier["model"],
            "lm_model_path": tier["lm"],
            "inference_steps": tier["inference_steps"],
            # Ask for WAV directly. The default is 128 kbps MP3, which we would
            # only have to transcode back for mixing and separation anyway.
            "audio_format": "wav",
            # One clip per call: we make variations from distinct seeds, and the
            # default batch of 2 doubles the work for a second take we discard.
            "batch_size": 1,
        }
        if tier["guidance_scale"] is not None:
            params["guidance_scale"] = tier["guidance_scale"]
        return params

    def _run(self, params, dest, src_audio=None, report=None):
        self._require_task(params)
        task_id = self._submit(params, src_audio)
        return self._await(task_id, dest, report)

    def _require_task(self, params) -> None:
        """Refuse a task this checkpoint does not implement, before submitting.

        ACE-Step's HTTP API accepts any task name and only discovers the model
        cannot do it well into the render, so the failure arrives minutes late
        and looks like a broken feature. Checking here turns that into an
        immediate, readable refusal naming the checkpoint that would be used.
        """
        task = params.get("task_type")
        model = params.get("model")
        if not task or not model:
            return
        supported = self.model_tasks(model)
        if task in supported:
            return
        raise NotSupported(
            f"{model} does not implement '{task}' -- it is a distilled turbo "
            f"checkpoint, which implements {', '.join(supported)}. "
            f"Install acestep-v15-xl-base, or run this on a pod that has it.")

    # -- Engine interface --------------------------------------------------

    def generate(self, dest, prompt, grid, seed, lyrics="", vocal_language="en",
                 src_audio=None, report=None, quality=None):
        params = {
            **self._grid_params(grid),
            **self._quality_params(quality),
            "task_type": "cover" if src_audio else "text2music",
            "prompt": prompt,
            "lyrics": lyrics or None,
            "vocal_language": vocal_language,
            **_seed_params(seed),
        }
        return self._run(params, dest, src_audio, report)

    def layer(self, dest, track_class, src_audio, prompt, grid, seed,
              lyrics="", vocal_language="en", report=None, quality=None):
        params = {
            **self._grid_params(grid),
            **self._quality_params(quality),
            "task_type": "lego",
            "track_name": track_class,
            "prompt": prompt,
            "lyrics": lyrics or None,
            "vocal_language": vocal_language,
            **_seed_params(seed),
        }
        return self._run(params, dest, src_audio, report)

    def repaint(self, dest, src_audio, start_s, end_s, prompt, grid, seed,
                report=None, quality=None):
        params = {
            **self._grid_params(grid),
            **self._quality_params(quality),
            "task_type": "repaint",
            "prompt": prompt,
            "repainting_start": round(start_s, 3),
            "repainting_end": round(end_s, 3),
            "chunk_mask_mode": "explicit",
            **_seed_params(seed),
        }
        return self._run(params, dest, src_audio, report)

    def extend(self, dest, src_audio, added_s, prompt, grid, seed, report=None,
               quality=None):
        params = {
            **self._grid_params(grid),
            **self._quality_params(quality),
            "task_type": "complete",
            "prompt": prompt,
            "audio_duration": round(grid.duration_s + added_s, 3),
            **_seed_params(seed),
        }
        return self._run(params, dest, src_audio, report)

    def extract(self, dest, src_audio, track_class, grid, seed=0, report=None):
        """Generative stem separation. Demucs is usually the better choice."""
        params = {
            **self._grid_params(grid),
            "task_type": "extract",
            "track_name": track_class,
            **_seed_params(seed),
        }
        return self._run(params, dest, src_audio, report)

    def separate(self, src_audio, out_dir):
        raise NotSupported("use DemucsEngine for separation, or acestep.extract per track")

    def convert_voice(self, dest, src_audio, reference):
        raise NotSupported("ACE-Step has no voice library; use Seed-VC")

    def sfx(self, dest, prompt, duration_s, seed):
        raise NotSupported("use Stable Audio 3 for sound effects")

    def available_models(self) -> dict:
        """What the pod can actually render with.

        Asking for a model that is not present makes ACE-Step fall back
        silently, so a quality tier can look like it worked while producing the
        default. This is how that becomes checkable rather than inferred from
        suspiciously equal render times.
        """
        installed = self.installed_models()
        effective = {k: self._resolve_model(v["model"], installed)
                     for k, v in QUALITY.items()}
        tasks_by_tier = {k: self.model_tasks(m) for k, m in effective.items()}
        tasks_union = {t for tasks in tasks_by_tier.values() for t in tasks}
        info: dict = {"requested_tiers": {k: v["model"] for k, v in QUALITY.items()},
                      # What is on disk, which is the answer to "can this pod
                      # render that tier". /v1/models only reports what has been
                      # loaded, and is empty until something renders.
                      "installed": installed,
                      "effective_tiers": effective,
                      # What this pod can actually be asked to do. The union
                      # across tiers, because the client picks a tier per
                      # render: a task is offerable if any installed checkpoint
                      # implements it.
                      "tasks": sorted(tasks_union),
                      "tasks_by_tier": tasks_by_tier,
                      "last_render": dict(self.last_meta)}
        for path in ("/v1/models", "/models"):
            try:
                resp = requests.get(f"{self.base_url}{path}", timeout=15)
                if resp.status_code != 200:
                    continue
                body = resp.json()

                # ACE-Step wraps everything in {"data": ..., "code": 200}, and
                # here `data` is an object holding the list -- not the list.
                # Iterating it yields key names, which is how this managed to
                # report an empty inventory on a server that had models loaded.
                payload = body.get("data")
                if isinstance(payload, dict):
                    entries = payload.get("models") or []
                    info["default_model"] = payload.get("default_model")
                elif isinstance(payload, list):
                    entries = payload
                else:
                    entries = body.get("models") or []

                # The live shape is OpenAI-style and differs from the docs:
                #   {"object":"list","data":[{"id":"acestep/acestep-v15-turbo",
                #                             "name":"ACE-Step acestep-v15-turbo"}]}
                # `id` is namespaced and `name` is prose, so neither compares
                # equal to the bare checkpoint name a tier asks for. Both get
                # normalised rather than trusted.
                def _bare(value: str) -> str:
                    value = value.strip()
                    if "/" in value:
                        value = value.rsplit("/", 1)[-1]
                    for prefix in ("ACE-Step ", "acestep "):
                        if value.startswith(prefix):
                            value = value[len(prefix):]
                    return value.strip()

                names = []
                for entry in entries:
                    if isinstance(entry, dict):
                        raw = entry.get("id") or entry.get("name") or ""
                    elif isinstance(entry, str):
                        raw = entry
                    else:
                        continue
                    if raw:
                        names.append(_bare(raw))
                info["available"] = names

                # This endpoint reports models that are *loaded*, not models
                # that exist. ACE-Step lazy-loads on first request, so a freshly
                # restarted pod returns an empty list and every tier would look
                # like it falls back. Say so, rather than reporting a scary
                # inventory that is merely early.
                info["lists"] = "loaded models, not installed ones"
                if not names:
                    info["hint"] = ("nothing loaded yet -- ACE-Step lazy-loads on "
                                    "first request; render once, then re-check")
                info["source"] = path
                if not info["available"]:
                    # An empty list and an unparsed response look identical from
                    # the outside, and the difference is the whole diagnosis.
                    info["raw"] = str(body)[:400]
                break
            except (requests.RequestException, ValueError):
                continue
        info.setdefault("available", None)
        # Judged against what is on disk, not against what happens to be loaded.
        # Gating this on the loaded list meant a fresh pod reported no problems
        # right up until it rendered something at the wrong quality.
        if installed:
            missing = {tier: model for tier, model in info["requested_tiers"].items()
                       if model not in installed}
            info["tiers_that_would_fall_back"] = missing
            if missing:
                info["note"] = (
                    "These tiers ask for checkpoints this pod does not have. "
                    "ACE-Step falls back to its default silently, so they render "
                    "at the default quality no matter what is requested. The XL "
                    "checkpoints ship as separate Hugging Face repos "
                    "(ACE-Step/<name>), roughly 20 GB each; the base weights repo "
                    "carries only acestep-v15-turbo and acestep-5Hz-lm-1.7B."
                )
        return info

    def ping(self) -> bool:
        try:
            requests.get(f"{self.base_url}/openapi.json", timeout=10)
            return True
        except requests.RequestException:
            return False
