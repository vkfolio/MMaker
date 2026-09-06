#!/usr/bin/env bash
# Bring up both engines and the app, and keep them up.
#
# Two supervised loops rather than one process: ACE-Step and MMSynth fail
# independently, and losing the API because a model server crashed would take
# away the very page that could tell you it had.
#
# Redeploying code is `pkill -f uvicorn` (or `pkill -f acestep-api`), which the
# loop below notices and restarts. Redeploying *environment* needs THIS script
# restarted, because the exports happen once, here -- getting that wrong once
# silently left the access token unset on a public pod.
set -uo pipefail

export PATH="/root/.local/bin:$PATH"
export UV_LINK_MODE=copy
export UV_CACHE_DIR=/root/.uv-cache

ACESTEP_DIR="${ACESTEP_DIR:-/opt/ACE-Step-1.5}"
MMSYNTH_DIR="${MMSYNTH_DIR:-/opt/MMSynth}"
CHECKPOINTS="${MMSYNTH_ACESTEP_CHECKPOINTS:-/workspace/checkpoints}"
SOULX_WEIGHTS="${MMSYNTH_SOULX_WEIGHTS:-/workspace/soulx-weights}"
PORT="${MMSYNTH_PORT:-8100}"

log() { printf '[start] %s\n' "$*"; }

mkdir -p "$CHECKPOINTS" "$SOULX_WEIGHTS" "${MMSYNTH_DATA_DIR:-/workspace/mmsynth-data}"

# RunPod's own nginx listens on 8001, 8081, 7861, 9091 and 3001. Binding
# ACE-Step to 8001 looks like it worked and then serves you nginx's 502 page,
# so pick something it does not want.
ACESTEP_PORT="${ACESTEP_PORT:-8011}"
for candidate in 8011 8012 8013 8014; do
  if ! (ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | grep -q ":${ACESTEP_PORT} "; then
    break
  fi
  log "port $ACESTEP_PORT is taken; trying $candidate"
  ACESTEP_PORT="$candidate"
done
export ACESTEP_URL="http://127.0.0.1:${ACESTEP_PORT}"
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-$CHECKPOINTS/acestep-v15-xl-sft}"
export ACESTEP_CHECKPOINT_PATH="$CHECKPOINTS"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
export ACESTEP_API_HOST=127.0.0.1
export ACESTEP_API_PORT="$ACESTEP_PORT"
export HF_HOME="${HF_HOME:-/workspace/.hf}"

# An access token is generated on first boot if none was supplied. A pod on a
# public proxy URL with no token is an open GPU.
TOKEN_FILE="/workspace/.mmsynth-token"
if [ -z "${MMSYNTH_API_TOKEN:-}" ]; then
  if [ -f "$TOKEN_FILE" ]; then
    MMSYNTH_API_TOKEN="$(cat "$TOKEN_FILE")"
  else
    MMSYNTH_API_TOKEN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
    printf '%s' "$MMSYNTH_API_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
  fi
fi
export MMSYNTH_API_TOKEN
log "access token: $MMSYNTH_API_TOKEN"
log "open:  https://<POD_ID>-${PORT}.proxy.runpod.net/?token=${MMSYNTH_API_TOKEN}"

# --- ACE-Step, supervised --------------------------------------------------
(
  cd "$ACESTEP_DIR" || exit 0
  while [ ! -f /workspace/.acestep-stop ]; do
    # The weights may still be downloading on first boot. Waiting here rather
    # than failing keeps the restart loop quiet instead of thrashing.
    if [ ! -f "$CHECKPOINTS/acestep-v15-xl-sft/config.json" ]; then
      sleep 20
      continue
    fi
    log "starting ACE-Step on :${ACESTEP_PORT}"
    uv run acestep-api --host 127.0.0.1 --port "$ACESTEP_PORT" || true
    sleep 3
  done
) >> /workspace/acestep.log 2>&1 &

# --- MMSynth, supervised, in the foreground so the container lives ---------
cd "$MMSYNTH_DIR" || exit 1
export PYTHONPATH="$MMSYNTH_DIR"
while [ ! -f /workspace/.mmsynth-stop ]; do
  log "starting MMSynth on :${PORT}"
  python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" || true
  sleep 2
done
