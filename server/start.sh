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

port_taken() {
  (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -q ":$1[[:space:]]"
}

# RunPod images run their own nginx on 8001. Rather than fail with an opaque
# "address already in use", step to the next free port -- ACESTEP_URL follows.
PORT_MOVED=0
if port_taken "$ACESTEP_PORT"; then
  for candidate in 8801 8802 8803 8804 8805; do
    if ! port_taken "$candidate"; then
      echo "[start] port $ACESTEP_PORT is taken; using $candidate for ACE-Step"
      ACESTEP_PORT="$candidate"
      PORT_MOVED=1
      break
    fi
  done
fi

# If we had to move, the port we just picked wins over any inherited
# ACESTEP_URL -- otherwise the engine would keep calling the port we avoided.
if [ "$PORT_MOVED" = "1" ]; then
  export ACESTEP_URL="http://127.0.0.1:${ACESTEP_PORT}"
else
  export ACESTEP_URL="${ACESTEP_URL:-http://127.0.0.1:${ACESTEP_PORT}}"
fi
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
    #
    # Call the venv binary directly. `uv run` re-syncs against uv.lock first,
    # which re-downloads multi-GB CUDA wheels whenever the venv was populated
    # any other way -- turning a start into another hour of downloading.
    if [ -x .venv/bin/acestep-api ]; then
      exec .venv/bin/acestep-api --host 127.0.0.1 --port "$ACESTEP_PORT"
    fi
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
