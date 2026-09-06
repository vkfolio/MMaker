"""SoulX-Singer: notes plus syllables plus a reference voice, to a sung vocal.

Apache-2.0 on both code and weights.
  paper   arxiv.org/abs/2602.07803
  code    github.com/Soul-AILab/SoulX-Singer
  weights huggingface.co/Soul-AILab/SoulX-Singer

Measured on an RTX 3090, fp16, rather than assumed:

  parameters   704 M
  peak VRAM    3.1 GB          (the research guess was 8-16 GB; it is not)
  render       1.57 s for a 6.9 s phrase, about 4x realtime
  load         ~11 s, once
  output       24 kHz mono -- the model's rate, not a choice

Because loading costs ~11 s and rendering costs ~1.5 s, the model is held
resident and `model.infer` is called directly. Shelling out to their
`cli.inference` would pay the load on every single render, which would make an
interactive tool feel like a batch queue for no reason.

Score control, not melody control. Melody control needs an f0 contour, which
only exists if you already have a recording of the tune -- and if you had that
you would not be drawing notes. Score control takes the notes themselves, which
is the whole reason this model was chosen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import audio, soulx_score
from ..config import settings
from ..errors import NotReady, RenderError


class SoulXSinger:
    name = "soulx"

    def __init__(self, repo_dir: Path | None = None, model_path: Path | None = None):
        self.repo_dir = Path(repo_dir or settings.soulx_dir)
        self.model_path = Path(model_path) if model_path else self._default_model()
        self._model = None
        self._config = None
        self._processor = None
        self._device = "cuda"

    def _default_model(self) -> Path:
        if settings.soulx_model:
            return Path(settings.soulx_model)
        packaged = self.repo_dir / "pretrained_models" / "SoulX-Singer" / "model.pt"
        if packaged.exists():
            return packaged
        return settings.models_dir / "soulx-singer" / "model.pt"

    # -- loading -----------------------------------------------------------

    def _load(self):
        """Build the model once. Everything that can be wrong says which thing."""
        if self._model is not None:
            return self._model

        try:
            import torch
        except ImportError as exc:
            raise NotReady(
                "torch is not installed here. SoulX-Singer runs on the pod; "
                "locally, run with MMSYNTH_ENGINE=stub.") from exc

        if not self.repo_dir.exists():
            raise NotReady(
                f"the SoulX-Singer checkout is not at {self.repo_dir}. It is a "
                "Python package, not just weights, so the code has to be there "
                "too. Set MMSYNTH_SOULX_DIR.")
        if not self.model_path.exists():
            raise NotReady(
                f"SoulX-Singer weights are not at {self.model_path}. Fetch them "
                "with: hf download Soul-AILab/SoulX-Singer --local-dir "
                f"{self.repo_dir}/pretrained_models/SoulX-Singer")

        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        try:
            from soulxsinger.models.soulxsinger import SoulXSinger as Model
            from soulxsinger.utils.data_processor import DataProcessor
            from soulxsinger.utils.file_utils import load_config
        except ImportError as exc:
            raise NotReady(
                f"the SoulX-Singer package at {self.repo_dir} could not be "
                f"imported ({exc}). Its inference dependencies are numpy<2, "
                "transformers==4.41.2, librosa, omegaconf, soundfile, scipy, "
                "accelerate.") from exc

        config_path = self.repo_dir / "soulxsinger" / "config" / "soulxsinger.yaml"
        phoneset = self.repo_dir / "soulxsinger" / "utils" / "phoneme" / "phone_set.json"
        if not config_path.exists():
            raise NotReady(f"no model config at {config_path}")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        config = load_config(str(config_path))

        model = Model(config).to(self._device)
        checkpoint = torch.load(str(self.model_path), weights_only=False,
                                map_location="cpu")
        if "state_dict" not in checkpoint:
            raise RenderError(
                f"{self.model_path.name} has no 'state_dict'; it is not a "
                "SoulX-Singer checkpoint")
        model.load_state_dict(checkpoint["state_dict"], strict=True)

        if self._device.startswith("cuda"):
            # fp16 everywhere except the mel transform, which their own loader
            # keeps in fp32 -- halving it produces silence, not a worse render.
            model.half()
            model.mel.float()
        model.eval()

        self._config = config
        self._processor = DataProcessor(
            hop_size=config.audio.hop_size,
            sample_rate=config.audio.sample_rate,
            phoneset_path=str(phoneset),
            device=self._device,
        )
        self._model = model
        return model

    def warm(self) -> dict:
        """Load now rather than on the first user's render.

        The 11 s load is the difference between a first render that feels broken
        and one that feels like the rest of them.
        """
        import torch
        self._load()
        used = (torch.cuda.max_memory_allocated() / 1e9
                if self._device.startswith("cuda") else 0.0)
        return {"loaded": True, "device": self._device,
                "vram_gb": round(used, 2),
                "params_m": round(sum(p.numel() for p in self._model.parameters()) / 1e6, 1)}

    # -- rendering ---------------------------------------------------------

    def sing(self, dest, notes, bpm: float, reference=None, language: str = "en",
             seed: int = 0, report=None) -> Path:
        import torch

        if reference is None:
            raise NotReady(
                "SoulX-Singer is zero-shot: it needs a reference voice to sing "
                "as. Pick one with voice_id.")

        model = self._load()
        sr = self._config.audio.sample_rate

        segments = soulx_score.build_target(notes)
        if not segments:
            raise NotReady("there is nothing to sing")

        if report:
            report(0.2, "reading the voice")
        prompt_meta = json.loads(Path(reference.meta).read_text(encoding="utf-8"))
        if isinstance(prompt_meta, list):
            if not prompt_meta:
                raise NotReady(f"{reference.label or 'that voice'} has empty metadata")
            prompt_meta = prompt_meta[0]
        prompt = self._processor.process(prompt_meta, str(reference.wav))

        total = int(segments[-1]["time"][1] / 1000 * sr)
        merged = np.zeros(max(total, 1), dtype=np.float32)

        for index, segment in enumerate(segments):
            if report:
                report(0.25 + 0.65 * index / len(segments),
                       f"singing {index + 1} of {len(segments)}")
            start = int(segment["time"][0] / 1000 * sr)
            target = self._processor.process(dict(segment), None)
            with torch.no_grad():
                got = model.infer(
                    {"prompt": prompt, "target": target},
                    auto_shift=True,
                    pitch_shift=0,
                    n_steps=self._config.infer.n_steps,
                    cfg=self._config.infer.cfg,
                    control="score",
                    use_fp16=self._device.startswith("cuda"),
                )
            got = got.squeeze().float().cpu().numpy()
            room = min(got.shape[0], merged.shape[0] - start)
            if room > 0:
                merged[start:start + room] = got[:room]

        if report:
            report(0.92, "writing it out")
        # The model is 24 kHz mono. Resampling to the service rate here keeps
        # every file MMSynth produces the same shape, so a client never has to
        # ask which engine made a wav before it can play it.
        return audio.write_resampled(dest, merged, sr, settings.sample_rate)
