"""Environment-driven settings.

Defaults are chosen so that `docker run -p 8000:8000 musicmaker` works with no
configuration at all, and so that a RunPod pod with a network volume mounted at
/workspace persists model weights across restarts.
"""

import os
from pathlib import Path


def _bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # Where weights live. On RunPod this should be the network volume so the
    # 20-40 GB download happens once, not on every pod start.
    models_dir = Path(os.environ.get("MUSICMAKER_MODELS_DIR", "/workspace/models"))

    # Where generated audio and project state live.
    data_dir = Path(os.environ.get("MUSICMAKER_DATA_DIR", "/workspace/data"))

    host = os.environ.get("MUSICMAKER_HOST", "0.0.0.0")
    port = int(os.environ.get("MUSICMAKER_PORT", "8000"))

    # Browsers hit the pod directly, and the pod's hostname is not known ahead
    # of time, so the default is permissive. Lock this down if you ever expose
    # a pod publicly.
    cors_origins = os.environ.get("MUSICMAKER_CORS_ORIGINS", "*").split(",")

    # Skip real downloads and fake the progress. Used for local development and
    # for the test suite -- never set this on a real pod.
    stub_models = _bool("MUSICMAKER_STUB_MODELS", False)

    # RunPod proxy URLs are publicly reachable. When this is set, every request
    # must carry it (X-API-Token header, or ?token= for links the browser opens
    # directly). Unset means open, which is fine only on a private network.
    api_token = os.environ.get("MUSICMAKER_API_TOKEN", "").strip()

    registry_path = Path(os.environ.get(
        "MUSICMAKER_REGISTRY",
        Path(__file__).resolve().parent.parent / "models.yaml",
    ))

    @classmethod
    def ensure_dirs(cls):
        cls.models_dir.mkdir(parents=True, exist_ok=True)
        cls.data_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
