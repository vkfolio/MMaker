#!/usr/bin/env bash
# Mirror the ACE-Step weights into your own Hugging Face account.
#
# Why: upstream repos get renamed, gated, or deleted, and a model you cannot
# re-download is a model you cannot redeploy. A mirror pins a known-good
# version under your control.
#
# Legally fine: ACE-Step/Ace-Step1.5 is MIT and ungated. MIT permits
# redistribution provided the copyright notice travels with it -- the README
# this script writes carries the attribution, so keep it.
#
#   bash tools/mirror-model.sh                  # private mirror (default)
#   MIRROR_PRIVATE=0 bash tools/mirror-model.sh # public mirror
#
# Env (or /workspace/.env):
#   HF_TOKEN        required, needs write scope
#   SOURCE_REPO     default ACE-Step/Ace-Step1.5
#   MIRROR_REPO     default <your-username>/ace-step-1.5
set -euo pipefail

ROOT="${MUSICMAKER_ROOT:-/workspace}"
# shellcheck disable=SC1091
[ -f "$ROOT/.env" ] && . "$ROOT/.env"
[ -f "$ROOT/.hf_token" ] && HF_TOKEN="${HF_TOKEN:-$(cat "$ROOT/.hf_token")}"

: "${HF_TOKEN:?set HF_TOKEN (write scope) or put it in $ROOT/.env}"
SOURCE_REPO="${SOURCE_REPO:-ACE-Step/Ace-Step1.5}"
PRIVATE="${MIRROR_PRIVATE:-1}"
LOCAL_DIR="${LOCAL_DIR:-$ROOT/models/acestep}"

export HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1

python - <<'PY'
import os, sys, pathlib
from huggingface_hub import HfApi, snapshot_download

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
me = api.whoami()["name"]

source = os.environ.get("SOURCE_REPO", "ACE-Step/Ace-Step1.5")
mirror = os.environ.get("MIRROR_REPO") or f"{me}/{source.split('/')[-1].lower()}"
private = os.environ.get("MIRROR_PRIVATE", "1") != "0"
local = pathlib.Path(os.environ.get("LOCAL_DIR", "/workspace/models/acestep"))

print(f"source : {source}")
print(f"mirror : {mirror}  (private={private})")

# Reuse the weights the pod already pulled rather than downloading 10 GB twice.
if local.exists() and any(local.rglob("*.safetensors")):
    print(f"using local copy at {local}")
    folder = local
else:
    print("no local copy -- downloading first")
    folder = pathlib.Path(snapshot_download(repo_id=source, local_dir=str(local)))

total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
print(f"payload: {total/1e9:.2f} GB")

api.create_repo(mirror, repo_type="model", private=private, exist_ok=True)
print("repo ready")

# MIT requires the notice to travel with the copy.
readme = folder / "README.md"
attribution = f"""---
license: mit
tags: [music, text2music, acestep, mirror]
---

# {mirror}

A pinned mirror of [`{source}`](https://huggingface.co/{source}), kept so that
deployments do not depend on an upstream repo that may be renamed, gated, or
removed.

**All credit to the ACE-Step authors.** Original: https://github.com/ace-step/ACE-Step-1.5
Licensed MIT; this mirror redistributes under the same terms.
"""
existing = readme.read_text(encoding="utf-8") if readme.exists() else ""
if "mirror of" not in existing:
    readme.write_text(attribution + ("\n\n---\n\n" + existing if existing else ""),
                      encoding="utf-8")

print("uploading… (large files go through the LFS path; this takes a while)")
api.upload_folder(
    folder_path=str(folder),
    repo_id=mirror,
    repo_type="model",
    commit_message=f"Mirror of {source}",
    ignore_patterns=["*.lock", ".cache*", "**/.cache*"],
)
print(f"\nDONE  https://huggingface.co/{mirror}")
print(f"\nPoint the engine at it:  export MUSICMAKER_ACESTEP_REPO={mirror}")
print("or edit server/models.yaml -> models[0].repo_id")
PY
