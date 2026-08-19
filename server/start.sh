#!/usr/bin/env bash
# Pod entrypoint: bring up ACE-Step, then the musicmaker engine.
#
# Two processes, one container. ACE-Step serves the model on :8001 and only ever
# listens on localhost; musicmaker serves the API and UI on :8000 and is the only
# thing RunPod exposes. Set MUSICMAKER_ENGINE=stub to skip ACE-Step entirely --
# useful for proving the plumbing before waiting on a model download.
set -euo pipefail

ACESTEP_DIR="${ACESTEP_DIR:-/opt/ACE-Step-1.5}"
ACESTEP_PORT="${ACESTEP_PORT:-8001}"
export ACESTEP_URL="${ACESTEP_URL:-http://127.0.0.1:${ACESTEP_PORT}}"
PORT="${MUSICMAKER_PORT:-8000}"

log() { echo "[start] $*"; }

if [ "${MUSICMAKER_ENGINE:-}" = "stub" ] || [ "${MUSICMAKER_STUB_MODELS:-0}" = "1" ]; then
  log "stub mode -- not starting ACE-Step"
else
  if [ ! -d "$ACESTEP_DIR" ]; then
    log "ERROR: ACE-Step not found at $ACESTEP_DIR"
    log "Either build the image with the ACE-Step layer, or run with MUSICMAKER_ENGINE=stub."
    exit 1
  fi

  log "starting ACE-Step on :${ACESTEP_PORT} (first run downloads weights)"
  (
    cd "$ACESTEP_DIR"
    # Bind to localhost: the model server is an internal dependency, not a
    # public surface. Only musicmaker's port is published.
    exec uv run acestep-api --host 127.0.0.1 --port "$ACESTEP_PORT"
  ) &
  ACESTEP_PID=$!
  trap 'kill $ACESTEP_PID 2>/dev/null || true' EXIT

  log "waiting for ACE-Step to answer…"
  for i in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${ACESTEP_PORT}/openapi.json" >/dev/null 2>&1; then
      log "ACE-Step is up after ${i}s"
      break
    fi
    if ! kill -0 $ACESTEP_PID 2>/dev/null; then
      log "ERROR: ACE-Step exited during startup. Check its logs above."
      exit 1
    fi
    [ "$i" = 600 ] && log "WARNING: ACE-Step still not answering after 10 min; starting anyway"
    sleep 1
  done
fi

log "starting musicmaker on :${PORT}"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
