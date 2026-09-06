"""MMSynth: a bench of single-purpose audio tools, served by the pod that runs them.

Turn the pod on, open its URL, work. The app is static files served from this
same process, so there is no second thing to host and no CORS problem to have.

Two engines behind it:

  ACE-Step 1.5 XL   audio in, audio out. Holds melody and timing through FSQ
                    semantic codes while the caption changes the character.
  SoulX-Singer      notes and phonemes in, a sung voice out.

Neither does the other's job, which is why there are two.
"""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import bootstrap, jobs
from .config import settings
from .errors import MMSynthError, NotReady, NotSupported
from .routers import misc, render, sing

# Deliberately independent of the config paths: overriding a directory for a
# test must not silently unmount the app.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    # The queue schedules work from route handlers, which run in the threadpool
    # with no running loop of their own. It has to be handed this one.
    jobs.queue.bind(asyncio.get_running_loop())
    # Weights download on a daemon thread so HTTP answers immediately and the
    # boot screen can show progress rather than a hung page.
    bootstrap.start_background()
    yield


app = FastAPI(title="MMSynth", version="0.2.0", lifespan=lifespan)

# Same-origin by default. musicmaker ships `*` together with
# allow_credentials=True, which makes Starlette echo whatever origin asked --
# effectively granting any website credentialed access to the API. On a public
# pod proxy URL that is a real hole, so credentials are off unless origins are
# named explicitly.
_origins = [o.strip() for o in settings.cors_origins if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=("*" not in _origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_public(path: str) -> bool:
    """Paths that must work without a token.

    The page itself has to load unauthenticated so it can *ask* for a token,
    and /health has to answer so the pod's healthcheck and the boot screen
    work. Everything that spends GPU or returns audio is gated.
    """
    return (path in ("/", "/health", "/favicon.ico")
            or path.startswith(("/static/", "/docs", "/openapi.json", "/redoc")))


@app.middleware("http")
async def require_token(request: Request, call_next):
    """A shared secret, when one is set.

    The token rides in a query parameter as well as a header because
    EventSource and <audio> cannot set headers, and both need to reach gated
    routes. Compared with compare_digest so the check does not leak its answer
    through timing.
    """
    if settings.api_token and not _is_public(request.url.path):
        supplied = (request.headers.get("X-API-Token")
                    or request.query_params.get("token", ""))
        if not hmac.compare_digest(supplied, settings.api_token):
            return JSONResponse({"detail": "bad or missing access token"},
                                status_code=401)
    return await call_next(request)


@app.exception_handler(NotReady)
async def not_ready(request: Request, exc: NotReady):
    # 503, not 500: the request was fine and will work once something is
    # installed, downloaded or wired. The message says which.
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(NotSupported)
async def not_supported(request: Request, exc: NotSupported):
    return JSONResponse({"detail": str(exc)}, status_code=501)


@app.exception_handler(MMSynthError)
async def failed(request: Request, exc: MMSynthError):
    return JSONResponse({"detail": str(exc)}, status_code=500)


app.include_router(misc.router)
app.include_router(render.router)
app.include_router(sing.router)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        """The app.

        Never cached. The whole app is one file that changes every time
        anything ships, and a cached copy is worse than an old one: the page
        keeps working, so nothing looks broken, while the JavaScript running in
        the browser is a version nobody is looking at. That cost an afternoon
        of debugging a bug that had already been fixed.
        """
        return FileResponse(
            str(STATIC_DIR / "index.html"),
            headers={"Cache-Control": "no-store, must-revalidate"},
        )
else:                                                          # pragma: no cover
    @app.get("/", include_in_schema=False)
    def index():
        return {"name": "MMSynth", "note": f"no app built at {STATIC_DIR}",
                "health": "/health"}
