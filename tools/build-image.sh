#!/usr/bin/env bash
# Build and push the musicmaker pod image FROM a running pod, with no Docker
# daemon and no privileges.
#
# RunPod pods are containers: there is no dockerd, no docker.sock, and no
# CAP_SYS_ADMIN, so `docker build` is impossible. crane (go-containerregistry)
# builds by appending a layer tarball to a base image over the registry API,
# which needs none of that.
#
#   bash tools/build-image.sh
#
# Env (or /workspace/.env):
#   DOCKER_HUB_USERNAME, DOCKER_HUB_TOKEN   required
#   IMAGE_TAG          default: latest
#   BASE_IMAGE         default: the image this pod is running
#
# WHAT GOES IN, AND WHY:
#   - the ACE-Step checkout and its .venv, at the SAME absolute path. uv bakes
#     absolute paths into venv shebangs and pyvenv.cfg, so relocating it breaks
#     it. Keeping /workspace/... means a byte-identical copy that just works.
#   - our pip packages from the system site-packages
#   - the app itself, INCLUDING its .git. Without it a pod cannot `git pull`
#     to pick up fixes made after the image was built, and a start command that
#     tries will fail and restart-loop the container.
#
# WHAT STAYS OUT:
#   - model weights. They pull from HF at ~1 GB/s on RunPod; a Docker Hub pull
#     is far slower, so baking them in would make cold start WORSE.
#
# CONSEQUENCE: a pod using this image must mount its volume somewhere OTHER
# than /workspace -- /data is the convention here -- or the mount hides the
# image's own /workspace contents.
set -euo pipefail

ROOT="${MUSICMAKER_ROOT:-/workspace}"
TAG="${IMAGE_TAG:-latest}"
WORK="${BUILD_WORKDIR:-/tmp/mmimg}"

# shellcheck disable=SC1091
[ -f "$ROOT/.env" ] && . "$ROOT/.env"

USER_NAME="${DOCKER_HUB_USERNAME:-}"
TOKEN="${DOCKER_HUB_TOKEN:-}"
if [ -z "$USER_NAME" ] || [ -z "$TOKEN" ]; then
  echo "ERROR: set DOCKER_HUB_USERNAME and DOCKER_HUB_TOKEN (or put them in $ROOT/.env)" >&2
  exit 1
fi

IMAGE="docker.io/${USER_NAME}/mmaker:${TAG}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }

step "sanity"
for p in "$ROOT/ACE-Step-1.5/.venv" "$ROOT/MMaker/server"; do
  [ -d "$p" ] || { echo "ERROR: missing $p -- run install.sh first" >&2; exit 1; }
done
info "ACE-Step venv and app present"

# The base must match what this pod runs, or the copied venv meets a different
# libc/python and breaks in ways that surface only at runtime.
BASE="${BASE_IMAGE:-}"
if [ -z "$BASE" ]; then
  BASE="$(cat /etc/runpod-image 2>/dev/null || true)"
fi
if [ -z "$BASE" ]; then
  echo "ERROR: set BASE_IMAGE to this pod's container image (RunPod console shows it)." >&2
  echo "       e.g. BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" >&2
  exit 1
fi
info "base image: $BASE"

step "crane"
mkdir -p "$WORK"
if [ ! -x "$WORK/crane" ]; then
  CRANE_VER="${CRANE_VERSION:-0.20.2}"
  curl -sL "https://github.com/google/go-containerregistry/releases/download/v${CRANE_VER}/go-containerregistry_Linux_x86_64.tar.gz" \
    | tar -xz -C "$WORK" crane
  chmod +x "$WORK/crane"
fi
info "$("$WORK/crane" version 2>/dev/null || echo present)"

step "layer"
SITE="$(python -c 'import site;print(site.getsitepackages()[0])')"
info "site-packages: $SITE"

LAYER="$WORK/layer.tar"
rm -f "$LAYER"

# Exclude caches and the things that must stay per-pod: model weights (huge and
# faster from HF), project data, logs, pid files, and the token files -- baking
# a token into an image would ship a credential to anyone who can pull it.
tar --create --file "$LAYER" \
    --exclude='*.pyc' --exclude='__pycache__' \
    --exclude="$ROOT/models" --exclude="$ROOT/data" \
    --exclude="$ROOT/.env" --exclude="$ROOT/.hf_token" \
    --exclude="$ROOT/.musicmaker_token" \
    --exclude="$ROOT/*.log" --exclude="$ROOT/*.pid" \
    "$ROOT/ACE-Step-1.5" \
    "$ROOT/MMaker" \
    "$SITE" 2>/dev/null || true

info "layer: $(du -sh "$LAYER" | cut -f1)"

step "push  $IMAGE"
export CRANE_USERNAME="$USER_NAME" CRANE_PASSWORD="$TOKEN"
echo "$TOKEN" | "$WORK/crane" auth login docker.io -u "$USER_NAME" --password-stdin >/dev/null
"$WORK/crane" append -b "$BASE" -f "$LAYER" -t "$IMAGE"
info "pushed"

step "done"
cat <<EOF

  Image:  ${USER_NAME}/mmaker:${TAG}   (private)

  New pod:
    Image             ${USER_NAME}/mmaker:${TAG}
    Registry auth     your Docker Hub credentials (RunPod > Settings > Registry)
    Volume mount      /data        <-- NOT /workspace; the image owns /workspace
    Expose HTTP port  8000

  Then, in the pod:
    export MUSICMAKER_MODELS_DIR=/data/models
    export MUSICMAKER_DATA_DIR=/data/data
    export HF_TOKEN=<your token>
    /workspace/MMaker/server/start.sh

  Weights download from HF on first run (fast); later runs reuse /data.
EOF
