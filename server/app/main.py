"""musicmaker engine -- FastAPI app.

Phase 1 scope: boot, download weights to the volume, report readiness, and serve
the UI. Generation endpoints arrive in Phase 2.
"""

import contextlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import bootstrap, engines, jobs
from .config import settings
from .routers import misc, projects, stems

# Independent of the registry path -- overriding MUSICMAKER_REGISTRY for a
# test must not silently unmount the UI.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    settings.ensure_dirs()
    jobs.queue.bind(asyncio.get_running_loop())
    bootstrap.start_background()
    yield


app = FastAPI(title="musicmaker engine", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(projects.router)
app.include_router(stems.router)
app.include_router(misc.jobs_router)
app.include_router(misc.voices_router)


@app.middleware("http")
async def require_token(request, call_next):
    """Optional shared-secret gate, for pods exposed on a public proxy URL."""
    if settings.api_token:
        supplied = (request.headers.get("X-API-Token")
                    or request.query_params.get("token", ""))
        # The page itself is not sensitive and must load so it can ask for a
        # token; /health stays open for the pod's healthcheck. Everything that
        # spends GPU or returns audio is gated.
        path = request.url.path
        public = path == "/health" or path == "/" or path.startswith("/static/")
        if not public and supplied != settings.api_token:
            return JSONResponse({"detail": "bad or missing API token"}, status_code=401)
    return await call_next(request)


@app.get("/health")
def health():
    """Boot and download progress. The UI polls this until status == ready."""
    body = bootstrap.state.public()
    body["gpu"] = _gpu_info()
    body["paths"] = {
        "models_dir": str(settings.models_dir),
        "data_dir": str(settings.data_dir),
    }
    body["stub_models"] = settings.stub_models
    body["auth_required"] = bool(settings.api_token)
    return body


def _gpu_info():
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "count": torch.cuda.device_count(),
        "total_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    }


@app.get("/api/info")
def info():
    """What this engine can do right now. The UI uses it to hide unbuilt features."""
    ready = {m.name for m in bootstrap.state.models if m.status == "ready"}
    stub = settings.stub_models
    return {
        "version": app.version,
        "phase": 7,
        "engines": engines.active(),
        "capabilities": {
            # Stub mode can serve every capability, which is what makes the
            # whole API testable without a GPU.
            "generate": stub or "acestep" in ready,
            "separate": stub or "demucs" in ready,
            "voice_convert": stub or "seedvc" in ready,
            "sfx": stub or "stableaudio" in ready,
        },
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))
