#!/usr/bin/env bash
# Build and push the MMSynth pod image FROM a running pod, with no Docker
# daemon and no privileges.
#
#   bash tools/build-image.sh
#
# WHY NOT `docker build`: a RunPod pod is itself a container. There is no
# dockerd, no docker.sock and no CAP_SYS_ADMIN. crane (go-containerregistry)
# builds by appending a layer tarball to a base image over the registry API,
# which needs none of that -- and it pushes from the datacentre, so ~20 GB never
# crosses a home connection.
#
# WHAT GOES IN:
#   /opt/ACE-Step-1.5   the checkout and its uv venv, at the SAME absolute path.
#                       uv bakes absolute paths into shebangs and pyvenv.cfg, so
#                       relocating it breaks it in ways that only show at runtime.
#   /opt/SoulX-Singer   the checkout
#   /opt/MMSynth        the app, including static/
#   system site-packages  SoulX's and MMSynth's pip installs
#   /root/nltk_data     g2p_en's tagger and cmudict, or phonemes fail on a cold pod
#
# WHAT STAYS OUT:
#   model weights   ~35 GB, and the volume mounts at /workspace where it would
#                   hide them anyway. app/bootstrap.py fetches them on first boot.
#   every token     baking a credential into an image ships it to anyone who
#                   can pull it. .env, .mmsynth-token and the HF cache are excluded.
set -euo pipefail

TAG="${IMAGE_TAG:-latest}"
WORK="${BUILD_WORKDIR:-/root/mmimg}"

# Credentials come from the environment or /workspace/.env, never from the tree
# being tarred.
# shellcheck disable=SC1091
[ -f /workspace/.env ] && . /workspace/.env

USER_NAME="${DOCKER_HUB_USERNAME:-}"
TOKEN="${DOCKER_HUB_TOKEN:-}"
if [ -z "$USER_NAME" ] || [ -z "$TOKEN" ]; then
  echo "ERROR: set DOCKER_HUB_USERNAME and DOCKER_HUB_TOKEN (or put them in /workspace/.env)" >&2
  exit 1
fi
IMAGE="docker.io/${USER_NAME}/mmsynth:${TAG}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }

step "sanity"
for p in /opt/ACE-Step-1.5/.venv /opt/SoulX-Singer /opt/MMSynth/app /opt/MMSynth/static; do
  [ -e "$p" ] || { echo "ERROR: missing $p" >&2; exit 1; }
done
info "all three trees present"

# The base must be exactly what this pod runs, or the copied venvs meet a
# different libc and python and break at runtime rather than at build time.
BASE="${BASE_IMAGE:-runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04}"
info "base: $BASE"

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
SITE="$(python3 -c 'import site;print(site.getsitepackages()[0])')"
info "site-packages: $SITE"

LAYER="$WORK/layer.tar"
# REUSE_LAYER=1 skips the tar when one is already built. Tarring 14 GB twice to
# retry a failed push is ten minutes nobody needs to spend.
if [ -s "$LAYER" ] && [ "${REUSE_LAYER:-0}" = "1" ]; then
  info "reusing the existing layer"
else
  rm -f "$LAYER"
fi

if [ ! -s "$LAYER" ]; then

# The symlinks into /workspace are kept: they are how the image finds weights on
# the volume. tar stores them as links, so they cost nothing and resolve once
# the volume is mounted.
tar --create --file "$LAYER" \
    --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
    --exclude='*.log' --exclude='.work' \
    --exclude='/opt/MMSynth/voices' \
    /opt/ACE-Step-1.5 /opt/SoulX-Singer /opt/MMSynth \
    /root/nltk_data /root/.local/bin \
    "$SITE" 2>/dev/null || true
fi

info "layer: $(du -sh "$LAYER" | cut -f1)"

step "push  $IMAGE"
# Strip CR: a .env copied from Windows carries \r, and a token with a trailing
# carriage return produces "malformed HTTP Authorization header" from the
# registry -- which reads like a credentials problem and is not one.
USER_NAME="${USER_NAME%$'\r'}"; TOKEN="${TOKEN%$'\r'}"
echo "$TOKEN" | "$WORK/crane" auth login docker.io -u "$USER_NAME" --password-stdin >/dev/null

"$WORK/crane" append -b "$BASE" -f "$LAYER" -t "$IMAGE"

# A separate step on purpose: `crane append` only stacks a layer and has no
# --set-entrypoint. `crane mutate` rewrites the config of an image that is
# already in the registry, so the append has to land first.
"$WORK/crane" mutate "$IMAGE" -t "$IMAGE" \
  --entrypoint /opt/MMSynth/start.sh \
  --env MMSYNTH_SOULX_DIR=/opt/SoulX-Singer \
  --env MMSYNTH_SOULX_WEIGHTS=/workspace/soulx-weights \
  --env MMSYNTH_ACESTEP_CHECKPOINTS=/workspace/checkpoints \
  --env MMSYNTH_DATA_DIR=/workspace/mmsynth-data \
  --env MMSYNTH_PORT=8100
info "pushed"

cat <<EOF

  Image:  ${USER_NAME}/mmsynth:${TAG}

  New pod:
    Image            ${USER_NAME}/mmsynth:${TAG}
    Registry auth    your Docker Hub credentials (RunPod > Settings > Registry)
    Volume           mount at /workspace  -- weights and your voices live there
    Expose HTTP      8100
    Disk             60 GB container

  First boot on a fresh volume downloads ~35 GB of weights and reports progress
  on /health; the page shows it. Later boots are instant.

  The access token is printed in the pod log and stored at
  /workspace/.mmsynth-token.
EOF
