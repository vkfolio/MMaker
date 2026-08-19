"""Adapters for the supporting models: Demucs, Seed-VC, Stable Audio 3.

Each imports its heavy dependency lazily, so the service starts (and the test
suite runs) on a machine where none of them are installed.
"""

from __future__ import annotations

from pathlib import Path

from .base import EngineError, NotSupported


class DemucsEngine:
    """Deterministic stem separation.

    htdemucs_ft is the strongest open vocal separator (~9.19 dB SDR); mdx_extra_q
    edges it on drums/bass/other. Default to htdemucs_ft because vocals are the
    stem users care most about getting clean.
    """

    name = "demucs"

    def __init__(self, model: str = "htdemucs_ft", device: str | None = None):
        self.model = model
        self.device = device

    def separate(self, src_audio, out_dir) -> dict[str, Path]:
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.audio import save_audio
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise EngineError(
                "demucs is not installed in this image -- enable it in models.yaml "
                "and add `demucs` to requirements") from exc

        from .. import audio as au
        import numpy as np

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        model = get_model(self.model)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        data, sr = au.load(src_audio)
        wav = torch.from_numpy(data.T).float()
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)

        with torch.no_grad():
            sources = apply_model(model, wav[None].to(device), device=device)[0]
        sources = sources * ref.std() + ref.mean()

        result = {}
        for name, source in zip(model.sources, sources):
            path = out_dir / f"{name}.wav"
            save_audio(source.cpu(), str(path), model.samplerate)
            result[name] = path
        return result

    def generate(self, *a, **k):
        raise NotSupported("demucs only separates")

    layer = repaint = extend = generate

    def convert_voice(self, *a, **k):
        raise NotSupported("demucs only separates")

    def sfx(self, *a, **k):
        raise NotSupported("demucs only separates")


class SeedVCEngine:
    """Zero-shot singing voice conversion -- the voice library and cloning.

    This is what fills the one real gap between ACE-Step and the ACE Studio
    experience: ACE-Step has no voice bank at all, so the same prompt gives a
    different singer every render. Converting onto a fixed reference clip gives
    both a curated voice list and consistency across songs.
    """

    name = "seedvc"

    def __init__(self, checkpoint_dir: Path | None = None, device: str | None = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise EngineError("torch is required for Seed-VC") from exc
        raise EngineError(
            "Seed-VC adapter is not wired to a checkpoint yet. Verify the repo id "
            "in models.yaml on the pod, enable it, then implement _load()."
        )

    def convert_voice(self, dest, src_audio, reference) -> Path:
        self._load()
        raise EngineError("unreachable -- _load() always raises until wired")

    def separate(self, *a, **k):
        raise NotSupported("seed-vc only converts voices")

    def generate(self, *a, **k):
        raise NotSupported("seed-vc only converts voices")

    layer = repaint = extend = generate

    def sfx(self, *a, **k):
        raise NotSupported("seed-vc only converts voices")


class StableAudioEngine:
    """Stable Audio 3 -- sound effects and textures.

    Open weights (small and medium), trained entirely on licensed data, which is
    a cleaner rights story than AudioGen's CC-BY-NC.
    """

    name = "stableaudio"

    def __init__(self, model_dir: Path | None = None, device: str | None = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import StableAudioPipeline
        except ImportError as exc:
            raise EngineError(
                "diffusers/torch not installed -- enable stableaudio in models.yaml "
                "and add `diffusers` to requirements") from exc
        if not self.model_dir or not self.model_dir.exists():
            raise EngineError(
                f"Stable Audio weights not found at {self.model_dir}. Verify the "
                "repo id in models.yaml on the pod and enable it.")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = StableAudioPipeline.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        return self._pipe

    def sfx(self, dest, prompt, duration_s, seed) -> Path:
        import torch
        from .. import audio as au

        pipe = self._load()
        generator = torch.Generator(pipe.device).manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            audio_end_in_s=float(duration_s),
            num_inference_steps=100,
            generator=generator,
        ).audios[0]
        data = result.T.float().cpu().numpy()
        return au.write(dest, data, pipe.vae.sampling_rate)

    def generate(self, dest, prompt, grid, seed, lyrics="", vocal_language="en",
                 src_audio=None):
        return self.sfx(dest, prompt, grid.duration_s, seed)

    def separate(self, *a, **k):
        raise NotSupported("stable audio does not separate")

    def layer(self, *a, **k):
        raise NotSupported("stable audio does not layer")

    repaint = extend = layer

    def convert_voice(self, *a, **k):
        raise NotSupported("stable audio does not convert voices")
