"""Environment-driven settings.

Defaults are chosen so `python -m uvicorn app.main:app` works with no
configuration, and so a RunPod pod with a network volume at /workspace keeps
its weights across restarts.

MMSynth listens on 8100, not 8000, so it can share a pod with musicmaker
without either one having to move.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    models_dir = Path(os.environ.get("MMSYNTH_MODELS_DIR", "/workspace/models"))
    data_dir = Path(os.environ.get("MMSYNTH_DATA_DIR", "/workspace/mmsynth-data"))

    host = os.environ.get("MMSYNTH_HOST", "0.0.0.0")
    port = int(os.environ.get("MMSYNTH_PORT", "8100"))
    cors_origins = os.environ.get("MMSYNTH_CORS_ORIGINS", "*").split(",")

    # "stub" forces the synthetic engines: no GPU, no weights, no soundfont.
    engine = os.environ.get("MMSYNTH_ENGINE", "").strip().lower()
    stub_models = _bool("MMSYNTH_STUB_MODELS", False) or engine == "stub"

    # RunPod proxy URLs are public. Set this on any pod that has one.
    api_token = os.environ.get("MMSYNTH_API_TOKEN", "").strip()

    registry_path = Path(os.environ.get("MMSYNTH_REGISTRY", HERE / "models.yaml"))
    instruments_path = Path(
        os.environ.get("MMSYNTH_INSTRUMENTS", HERE / "instruments.yaml"))

    # The sampler. A soundfont is the floor; an SFZ library is the quality path.
    soundfont = os.environ.get("MMSYNTH_SOUNDFONT", "").strip()
    sfz_dir = Path(os.environ.get("MMSYNTH_SFZ_DIR", HERE / "sfz"))

    # ACE-Step 1.5 XL. Port 8011, not 8001: RunPod's own nginx squats on 8001
    # (and 8081, 7861, 9091, 3001), which musicmaker's start.sh hits too.
    acestep_url = os.environ.get("ACESTEP_URL", "http://127.0.0.1:8011")
    acestep_steps = int(os.environ.get("MMSYNTH_ACESTEP_STEPS", "50"))
    acestep_guidance = float(os.environ.get("MMSYNTH_ACESTEP_GUIDANCE", "7.0"))

    # Whether the instrument path runs the sampler render through ACE-Step.
    # Off gives you the soundfont, which is honest but not the product.
    refine_enabled = _bool("MMSYNTH_REFINE", True)

    # Running a sung vocal back through ACE-Step `cover`. Measured to keep both
    # the singing and the melody (PHASE0.md gate 3), so it is a "another take,
    # richer production" pass -- not a way to change who is singing.
    revoice_noise = float(os.environ.get("MMSYNTH_REVOICE_NOISE", "0.35"))

    sample_rate = int(os.environ.get("MMSYNTH_SAMPLE_RATE", "48000"))

    # SoulX-Singer. The checkout is needed as well as the weights: the model is
    # a Python package, not a serialised graph, so `soulxsinger` has to be
    # importable. It outputs 24 kHz, which is the model's rate and not a choice.
    soulx_dir = Path(os.environ.get("MMSYNTH_SOULX_DIR", "/opt/SoulX-Singer"))
    # Weights live on the volume; code lives in the image. The volume is
    # mounted at /workspace, which would hide anything the image put there.
    soulx_weights = Path(os.environ.get("MMSYNTH_SOULX_WEIGHTS",
                                        "/workspace/soulx-weights"))
    acestep_checkpoints = Path(os.environ.get("MMSYNTH_ACESTEP_CHECKPOINTS",
                                              "/workspace/checkpoints"))
    soulx_model = os.environ.get("MMSYNTH_SOULX_MODEL", "").strip()
    soulx_sample_rate = 24000


    # Development escape hatch, off by default.
    #
    # A voice with no recorded consent is refused at render time. That is the
    # right default and it stays the default; this exists so the pipeline can be
    # proved against SoulX-Singer's own example prompt, which is a demonstration
    # clip and not a cleared product voice. Never set it on anything a user
    # touches.
    allow_unconsented_voices = _bool("MMSYNTH_ALLOW_UNCONSENTED_VOICES", False)

    @classmethod
    def ensure_dirs(cls):
        cls.models_dir.mkdir(parents=True, exist_ok=True)
        cls.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
