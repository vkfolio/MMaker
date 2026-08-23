#!/usr/bin/env bash
# Pod entrypoint: bring up ACE-Step, then the musicmaker engine.
#
# Two processes, one container. ACE-Step serves the model on :8001 and only ever
# listens on localhost; musicmaker serves the API and UI on :8000 and is the only
# thing RunPod exposes. Set MUSICMAKER_ENGINE=stub to skip ACE-Step entirely --
# useful for proving the plumbing before waiting on a model download.
set -euo pipefail

# Where ACE-Step lives depends on how the pod was built: the Dockerfile puts it
# in /opt, install.sh puts it on the volume, and a crane-appended image keeps
# whatever path it was tarred from. Look rather than assume.
if [ -z "${ACESTEP_DIR:-}" ]; then
  CANDIDATES="/opt/ACE-Step-1.5 /workspace/ACE-Step-1.5 ${MUSICMAKER_ROOT:-/workspace}/ACE-Step-1.5 /data/ACE-Step-1.5 $HOME/ACE-Step-1.5"
  # Two passes: a directory that actually holds checkpoints wins over one that
  # merely exists. This pod has an empty /opt/ACE-Step-1.5 beside a populated
  # /workspace one, and taking the first that existed pointed everything --
  # start.sh and the weight fetcher alike -- at the empty one.
  for candidate in $CANDIDATES; do
    if [ -n "$(ls -A "$candidate/checkpoints" 2>/dev/null)" ]; then
      ACESTEP_DIR="$candidate"
      break
    fi
  done
  if [ -z "${ACESTEP_DIR:-}" ]; then
    for candidate in $CANDIDATES; do
      if [ -d "$candidate" ]; then
        ACESTEP_DIR="$candidate"
        break
      fi
    done
  fi
fi
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
    log "ERROR: ACE-Step not found. Looked in /opt, /workspace, /data and \$HOME."
    log "Set ACESTEP_DIR explicitly, or run with MUSICMAKER_ENGINE=stub."
    exit 1
  fi
  log "ACE-Step at $ACESTEP_DIR"

  # Register the quality tiers' checkpoints as model slots.
  #
  # ACE-Step's model registry is fixed at startup and is NOT a scan of the
  # checkpoints directory. Downloading acestep-v15-xl-turbo is not enough: a
  # request naming it gets
  #     Model 'acestep-v15-xl-turbo' not found in ['acestep-v15-turbo'],
  #     using primary: acestep-v15-turbo
  # and renders at turbo quality without failing. Slots exist for exactly this
  # and are enabled only by these env vars, which take model names.
  #
  # Best-first, and deliberately so. Slot 1 is also the fallback for any model
  # ACE-Step does not recognise, so ordering it best-first means an unmatched
  # request degrades toward quality rather than away from it -- the failure
  # that actually happened here was silent degradation to the weakest model.
  # There are only three slots, which is exactly the number of tiers.
  CKPT_DIR="${ACESTEP_CHECKPOINT_DIR:-$ACESTEP_DIR/checkpoints}"
  PREFERENCE="${ACESTEP_MODEL_PREFERENCE:-acestep-v15-xl-base acestep-v15-xl-turbo acestep-v15-base acestep-v15-turbo}"

  slot=1
  registered=""
  for model in $PREFERENCE; do
    [ -d "$CKPT_DIR/$model" ] || continue
    [ "$slot" -gt 3 ] && { log "no slot left for $model (ACE-Step allows 3)"; continue; }
    if [ "$slot" = "1" ]; then
      export ACESTEP_CONFIG_PATH="$model"
    else
      export "ACESTEP_CONFIG_PATH${slot}=$model"
    fi
    log "model slot ${slot}: $model"
    registered="$registered $model"
    slot=$((slot + 1))
  done

  case "$registered" in
    *xl-base*)  : ;;
    *xl-turbo*) log "no acestep-v15-xl-base -- 'ultra' will render as 'high'" ;;
    # acestep-v15-base is not a downgrade in capability, only in size: it is
    # the smallest checkpoint that implements every task type, so lego,
    # complete and extract work here where turbo alone would refuse them.
    *v15-base*) log "no XL weights -- 'ultra' renders as acestep-v15-base" ;;
    *)          log "only acestep-v15-turbo present -- every tier renders as 'fast',"
                log "  and Add a Layer, Extend and Extract are unavailable"
                log "  fetch better weights: bash tools/fetch-quality-weights.sh all" ;;
  esac

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
      exec .venv/bin/acestep-api --host 127.0.0.1 --port "$ACESTEP_PORT"         >> "${ACESTEP_LOG:-/workspace/acestep.log}" 2>&1
    fi
    exec uv run acestep-api --host 127.0.0.1 --port "$ACESTEP_PORT"         >> "${ACESTEP_LOG:-/workspace/acestep.log}" 2>&1
  ) &
  ACESTEP_PID=$!
  # No EXIT trap here. This script ends by exec'ing uvicorn, and an EXIT trap
  # fires on the way out -- killing the ACE-Step we just waited for and leaving
  # a zombie. Keep it a plain background child so $! stays valid for the
  # liveness check below; setsid would detach it and break that check.

  # Its output would otherwise interleave with musicmaker's and be unreadable.
  log "ACE-Step logging to ${ACESTEP_LOG:-/workspace/acestep.log}"
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

# Supervise rather than exec.
#
# `exec` made uvicorn the container's own process, so stopping it to pick up
# new code stopped the pod -- deploying a one-line server change meant a
# console restart and a cold model load. Under this loop, `pkill -f uvicorn`
# is the whole deploy: the server comes back on the new code, ACE-Step keeps
# its weights resident, and the container never notices.
#
# MUSICMAKER_NO_SUPERVISE=1 restores the old behaviour for anything that
# expects the process to be the container.
if [ "${MUSICMAKER_NO_SUPERVISE:-0}" = "1" ]; then
  log "starting musicmaker on :${PORT} (unsupervised)"
  exec python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
fi

log "starting musicmaker on :${PORT} (supervised; pkill -f uvicorn to redeploy)"
STOP_FILE="${MUSICMAKER_STOP_FILE:-/workspace/.musicmaker-stop}"
rm -f "$STOP_FILE"
fails=0
while true; do
  start_ts=$(date +%s)
  python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" || true
  code=$?
  [ -f "$STOP_FILE" ] && { log "stop file present -- not restarting"; break; }

  # A run that lasted a while was a deploy or a kill; one that died instantly
  # is a crash loop, and restarting it as fast as possible helps nobody.
  if [ $(( $(date +%s) - start_ts )) -lt 10 ]; then
    fails=$((fails + 1))
    if [ "$fails" -ge 5 ]; then
      log "musicmaker exited immediately $fails times (last code $code) -- giving up"
      break
    fi
    log "musicmaker exited after less than 10s (code $code); retry $fails in 5s"
    sleep 5
  else
    fails=0
    log "musicmaker exited (code $code) -- restarting on current code"
    # Pick up whatever was pulled while it was down, so a redeploy is just a
    # git pull followed by a pkill.
    sleep 1
  fi
done
