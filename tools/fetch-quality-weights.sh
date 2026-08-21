#!/usr/bin/env bash
# Download the checkpoints the `high` and `ultra` tiers need.
#
# Why this exists: the base weights repo (ACE-Step/Ace-Step1.5) ships only
# acestep-v15-turbo and acestep-5Hz-lm-1.7B. Every other checkpoint is its own
# Hugging Face repo. Asking ACE-Step for one it does not have is not an error --
# it falls back to the default and renders anyway, so `ultra` silently produces
# turbo output. Nothing about the result says so; the only visible symptom is
# that all three tiers take the same time.
#
#   bash tools/fetch-quality-weights.sh high     # ~20 GB, xl-turbo
#   bash tools/fetch-quality-weights.sh ultra    # ~28 GB, xl-base + 4B LM
#   bash tools/fetch-quality-weights.sh all      # ~48 GB
#
# Weights land in ACE-Step's own checkpoint directory so it finds them without
# reconfiguration. Re-running is safe: snapshot_download skips what is present.
set -euo pipefail

TIER="${1:-high}"
ACESTEP_DIR="${ACESTEP_DIR:-/opt/ACE-Step-1.5}"
# ACE-Step loads from its own checkpoints/ directory, and start.sh reads the
# same place when deciding which model slots to register. A checkpoint
# anywhere else is invisible to both.
CKPT_DIR="${ACESTEP_CHECKPOINT_DIR:-${ACESTEP_DIR:-/workspace/ACE-Step-1.5}/checkpoints}"

case "$TIER" in
    high)  REPOS=(acestep-v15-xl-turbo) ;;
    ultra) REPOS=(acestep-v15-xl-base acestep-5Hz-lm-4B) ;;
    all)   REPOS=(acestep-v15-xl-turbo acestep-v15-xl-base acestep-5Hz-lm-4B) ;;
    *)     echo "usage: $0 [high|ultra|all]" >&2; exit 2 ;;
esac

echo "target      : $CKPT_DIR"
echo "checkpoints : ${REPOS[*]}"

# Free space is worth checking first: a half-downloaded 20 GB checkpoint is
# indistinguishable from a present one to anything that only looks for a folder.
AVAIL_GB=$(df -BG --output=avail "$(dirname "$CKPT_DIR")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "free space  : ${AVAIL_GB} GB"
NEED=0
for r in "${REPOS[@]}"; do
    case "$r" in
        acestep-v15-xl-*)   NEED=$((NEED + 20)) ;;
        acestep-5Hz-lm-4B)  NEED=$((NEED + 9))  ;;
    esac
done
echo "needs about : ${NEED} GB"
if [ "$AVAIL_GB" -gt 0 ] && [ "$AVAIL_GB" -lt "$NEED" ]; then
    echo "NOT ENOUGH SPACE -- ${NEED} GB needed, ${AVAIL_GB} GB free." >&2
    exit 1
fi

export HF_HUB_ENABLE_HF_TRANSFER=1     # ~18 MB/s vs ~0.4 MB/s without it
mkdir -p "$CKPT_DIR"

for repo in "${REPOS[@]}"; do
    echo
    echo "=== $repo ==="
    python - "$repo" "$CKPT_DIR" <<'PY'
import sys, pathlib
from huggingface_hub import snapshot_download

repo, root = sys.argv[1], pathlib.Path(sys.argv[2])
dest = root / repo
# Weight formats only. These repos also carry .bin duplicates of the
# safetensors, and pulling both doubles a 20 GB download for nothing.
path = snapshot_download(
    repo_id=f"ACE-Step/{repo}",
    local_dir=str(dest),
    # The XL repos carry no .bin duplicates, so the download is the four
    # shards plus small files -- and those small files matter: ACE-Step loads
    # the checkpoint through its own modeling_*.py / configuration_*.py, and
    # silence_latent.pt is needed at inference. Filtering to safetensors and
    # json fetches 20 GB of weights that then will not load.
    allow_patterns=["*.safetensors", "*.json", "*.py", "*.pt",
                    "*.txt", "*.model", "*.yaml"],
)
size = sum(f.stat().st_size for f in pathlib.Path(path).rglob("*") if f.is_file())
print(f"{repo}: {size/1e9:.2f} GB at {path}")
PY
done

echo
echo "Downloaded. ACE-Step does NOT rescan -- its model registry is fixed at"
echo "startup. Restart the server so start.sh registers the new model slot:"
echo "  pkill -f acestep-api && pkill -f 'uvicorn app.main'   # supervisor restarts both"
echo
echo "  curl -s -H \"X-API-Token: \$TOKEN\" http://localhost:8000/api/engine"
echo
echo "'available' should now list them, and 'tiers_that_would_fall_back' should"
echo "be empty for the tier you fetched. If a tier is still listed there, the"
echo "checkpoint is not where ACE-Step is looking -- check ACESTEP_CHECKPOINT_DIR."
