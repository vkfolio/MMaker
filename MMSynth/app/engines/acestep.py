"""ACE-Step 1.5 XL, over HTTP.

MMSynth borrows musicmaker's engine, not its code: nothing is imported across
the two projects, and this speaks the same REST surface the ACE-Step API server
exposes.

**How a melody gets into a model that cannot read notes.** ACE-Step has no MIDI,
melody or note conditioning of any kind. What it has is `cover`, which replaces
the source latents with **FSQ semantic codes** -- and those codes encode melody,
rhythm and harmony explicitly. So a sampler render of the score becomes the
carrier, and the codes hold the tune while the caption changes the timbre.

Measured on an RTX 6000 Ada with `acestep-v15-xl-sft` (see PHASE0.md):

    cover, every noise level tried   pitch error 0.00-0.03 semitones, 7/7 notes
    cover-nofsq, noise 0.15          pitch error 28.6 semitones -- unusable

That difference is the whole design. `cover-nofsq` feeds raw continuous VAE
latents instead of codes, so the melody survives only as long as the waveform
does. **Use `cover`.**

The consequence is counter-intuitive and worth stating plainly, because their own
UI copy implies the opposite: with `cover`, `cover_noise_strength` is a
**timbre-freedom dial, not a melody-safety dial**. Turning it down gives the model
more room to reinterpret while the codes keep the notes where they were.

Two more behaviours that surprise people:

  * `duration` is ignored -- output length is locked to the source audio.
  * `thinking` is a no-op. The LM is skipped for every direct-conditioning task
    (`inference.py:36-38`), so the caption and lyrics have to be written by us.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from ..config import settings
from ..errors import NotReady, RenderError

CONNECT_TIMEOUT = 180
POLL_SECONDS = 3.0
MAX_WAIT = 1800


class AceStepEngine:
    """A thin client. One instance per process; the server holds the model."""

    name = "acestep-xl"

    def __init__(self, base_url: str | None = None):
        self.base = (base_url or settings.acestep_url).rstrip("/")

    # -- plumbing ----------------------------------------------------------

    def health(self) -> dict:
        try:
            r = requests.get(f"{self.base}/health", timeout=10)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise NotReady(
                f"the ACE-Step engine at {self.base} is not answering ({exc}). "
                "Start it, or run with MMSYNTH_ENGINE=stub.") from exc
        return r.json().get("data", {})

    def available(self) -> bool:
        try:
            self.health()
            return True
        except NotReady:
            return False

    def _unwrap(self, payload: dict) -> dict | None:
        """Dig the real result out of two layers of nesting.

        `data` is a list of task envelopes and each envelope's `result` is a JSON
        *string* holding a *list*. The status that matters is the inner one; the
        envelope has none until the task finishes. Getting this wrong makes every
        task look like a crash, which is exactly what it did the first time.
        """
        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                return None
        if isinstance(result, list):
            result = result[0] if result else None
        return result if isinstance(result, dict) else None

    def _run(self, data: dict, files: dict | None, dest: Path,
             report=None) -> Path:
        try:
            r = requests.post(f"{self.base}/release_task", data=data, files=files,
                              timeout=CONNECT_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise NotReady(f"the engine would not accept the job ({exc})") from exc

        body = r.json()
        task_id = (body.get("data") or {}).get("task_id")
        if not task_id:
            raise RenderError(f"no task id in {str(body)[:200]}")

        started = time.time()
        while time.time() - started < MAX_WAIT:
            time.sleep(POLL_SECONDS)
            if report:
                # ACE-Step reports no progress. Saying so beats a bar that moves
                # on a timer and lies about what the GPU is doing.
                report(-1, "rendering")
            try:
                q = requests.post(f"{self.base}/query_result",
                                  json={"task_id_list": [task_id]}, timeout=60)
                q.raise_for_status()
            except requests.RequestException:
                continue

            envelopes = (q.json() or {}).get("data") or []
            if not envelopes:
                continue
            result = self._unwrap(envelopes[0])
            if result is None:
                continue

            if result.get("status") == 1:
                url = result.get("file") or ""
                if not url:
                    raise RenderError("the engine reported success but no file")
                got = requests.get(url if url.startswith("http") else self.base + url,
                                   timeout=600)
                got.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(got.content)
                return dest
            if result.get("status") == 2:
                raise RenderError(f"the engine failed: {str(result)[:300]}")

        raise RenderError(f"the engine produced nothing within {MAX_WAIT}s")

    # -- the two things MMSynth asks for -----------------------------------

    def cover(self, dest, src_audio, caption: str, *, lyrics: str = "",
              noise: float = 0.35, strength: float = 1.0, steps: int = 50,
              guidance: float = 7.0, seed: int = 0, language: str = "en",
              reference: Path | None = None, report=None) -> Path:
        """Re-timbre a sampler render, keeping its melody.

        `noise` is `cover_noise_strength` -- lower gives the model more freedom,
        because the FSQ codes are what hold the tune. `reference` is a 30 s clip
        for the global timbre encoder, which is how a voice is steered.
        """
        src_audio, dest = Path(src_audio), Path(dest)
        data = {
            "task_type": "cover",
            "prompt": caption,
            "lyrics": lyrics or "",
            "vocal_language": language,
            "cover_noise_strength": f"{max(0.0, min(1.0, noise)):.3f}",
            "audio_cover_strength": f"{max(0.0, min(1.0, strength)):.3f}",
            "inference_steps": str(int(steps)),
            "guidance_scale": f"{guidance:.2f}",
            "use_random_seed": "false",
            "seed": str(int(seed)),
            "audio_format": "wav",
            # A no-op for cover, sent explicitly so nobody wonders later.
            "thinking": "false",
        }
        files = {"src_audio": (src_audio.name, src_audio.read_bytes(), "audio/wav")}
        if reference is not None and Path(reference).exists():
            reference = Path(reference)
            files["ref_audio"] = (reference.name, reference.read_bytes(), "audio/wav")

        if report:
            report(-1, "reimagining it")
        return self._run(data, files, dest, report=report)

    def text2music(self, dest, caption: str, *, lyrics: str = "",
                   duration: float = 30.0, steps: int = 50, guidance: float = 7.0,
                   seed: int = 0, language: str = "en", report=None) -> Path:
        """Generate from nothing. Used to build reference timbre clips."""
        data = {
            "task_type": "text2music",
            "prompt": caption,
            "lyrics": lyrics or "",
            "vocal_language": language,
            "audio_duration": f"{duration:.1f}",
            "inference_steps": str(int(steps)),
            "guidance_scale": f"{guidance:.2f}",
            "use_random_seed": "false",
            "seed": str(int(seed)),
            "audio_format": "wav",
            "thinking": "false",
        }
        return self._run(data, None, dest, report=report)
