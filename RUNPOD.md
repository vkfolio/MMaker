# Running on RunPod

Two paths. **Start with Path A** — it proves the pod, the ports, and the UI in about
five minutes without waiting on a 12 GB model download. Then move to Path B once you
know the plumbing works.

> **Not yet verified on real hardware.** Everything below is built and tested locally
> against the stub engine. The ACE-Step layer in the Dockerfile and the launch command
> in `server/start.sh` follow ACE-Step's documented install (`uv sync`, `uv run
> acestep-api`) but have not been run on a GPU. Expect to fix something on first boot —
> that is what Path A is for.

---

## The short version

On a stock RunPod pod, one line:

```bash
bash <(curl -sL https://raw.githubusercontent.com/vkfolio/MMaker/main/install.sh)
```

It installs system packages, the app, and ACE-Step; generates an access token; writes
`/workspace/run-musicmaker.sh`; and starts the server, printing the URL to open.
Re-running it updates the code and skips whatever is already installed.

Add `--stub` to skip ACE-Step entirely and prove the pod in about two minutes. Do that
first — see Path A.

---

## Path A — prove the plumbing (stub engine, no GPU needed)

Use any cheap pod. A GPU is not required for this step.

**1. Create the pod**

- Template: `runpod/pytorch:2.5.1-py3.11-cuda12.1.1-devel-ubuntu22.04` (or any Python image)
- Network volume mounted at `/workspace` — 100 GB
- Expose HTTP port **8000**

**2. In the pod's terminal**

```bash
cd /workspace
git clone https://github.com/vkfolio/MMaker.git
cd MMaker/server
pip install -r requirements.txt

MUSICMAKER_ENGINE=stub \
MUSICMAKER_STUB_MODELS=1 \
MUSICMAKER_MODELS_DIR=/workspace/models \
MUSICMAKER_DATA_DIR=/workspace/data \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**3. Open it**

```
https://<POD_ID>-8000.proxy.runpod.net
```

You should get the boot screen, then the app. Make a take, split it, drag a bar range,
regenerate. It will sound like tones — that is the stub. If this works, the pod, the
ports, the proxy and the UI are all good, and anything that breaks later is the model.

---

## Bringing a terminated pod back

Terminating deletes the pod *and* its volume, so projects and generated audio
are gone. The environment is not: it is in the image below, weights included.

1. **RunPod → Settings → SSH Public Keys** — add your key here, not to a running
   container. Keys added here are injected at pod start and survive restarts.
2. Create a pod from `vignesh07021990/mmaker:latest` using the table below.
3. Open the URL the start command prints, or `curl <url>/health` until ready.

First boot is a 34.77 GB image pull. After that there is no install and no
weight download.

---

## Path B — start from the prebuilt image (fastest)

An image built from a working pod already exists:

    vignesh07021990/mmaker:latest      private, 34.77 GB, linux/amd64

It contains ACE-Step, its venv, this app, the Python deps **and the model
weights**, so a pod needs no install and no download.

| Setting | Value |
|---|---|
| Image | `vignesh07021990/mmaker:latest` |
| Registry auth | your Docker Hub credentials (RunPod → Settings → Registry) |
| GPU | 24 GB+ (32 GB comfortable) |
| Container disk | 60 GB+ (the image is 34.77 GB) |
| Volume mount | **`/data`** — NOT `/workspace` |
| Expose HTTP port | 8000 (do **not** also list 8000 as a TCP port) |

**The volume path matters.** The image owns `/workspace`; mounting a volume
there hides ACE-Step and the app, and the pod boots into an empty shell.

**A start command is required.** `crane append` keeps the base image's config,
so the image inherits RunPod's default command rather than `start.sh`. Set the
pod's start command to:

```
bash -c 'cd /workspace/MMaker/server && MUSICMAKER_MODELS_DIR=/data/models MUSICMAKER_DATA_DIR=/data/data MUSICMAKER_API_TOKEN=<your-token> ./start.sh'
```

Or leave the default command and run that line yourself in the pod terminal.

Rebuild the image after changing anything, from a pod that works:

```bash
BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 bash tools/build-image.sh
```

---

## Path C — build the engine from source

### B1. Get the image

**Option 1 — let CI build it (recommended).** `.github/workflows/docker.yml` builds and
pushes on every tag. Add two repo secrets under Settings → Secrets → Actions:

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | your Docker Hub username |
| `DOCKERHUB_TOKEN` | a Docker Hub access token (Account Settings → Security) |

Then:

```bash
git tag v0.1.0 && git push --tags
```

The image lands at `docker.io/<username>/mmaker:latest`. No local Docker needed, and
the runner has more disk than most laptops want to spare — the ACE-Step layer is large.

**Option 2 — build locally:**

```bash
cd server
docker build -t <dockerhub-user>/mmaker:latest .
docker push <dockerhub-user>/mmaker:latest
```

Weights are **not** in the image either way; they land on the volume at first run.

### B2. Create the pod

Use `runpod-template.json` as the reference, or set it up by hand:

| Setting | Value |
|---|---|
| Image | `<dockerhub-user>/mmaker:latest` |
| GPU | 24 GB is comfortable; ACE-Step runs from ~4 GB but headroom helps |
| Container disk | 20 GB |
| Network volume | 100 GB at `/workspace` |
| Exposed HTTP port | 8000 |
| Env | `MUSICMAKER_API_TOKEN=<a long random string>` |

`start.sh` launches ACE-Step on `127.0.0.1:8001`, waits for it to answer, then starts
musicmaker on `0.0.0.0:8000`. Only 8000 is published — the model server is never
exposed.

### B3. First boot

Watch the pod logs. First boot downloads ACE-Step's weights and can take 10–30 minutes.
`/health` reports progress the whole time, and the UI shows a progress bar rather than
hanging.

```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
```

Wait for `"status": "ready"`, then open the URL in a browser.

Later boots skip the download and are ready in under a minute.

---

## Protect the pod

**RunPod proxy URLs are publicly reachable.** Anyone with the URL can spend your GPU.

Set `MUSICMAKER_API_TOKEN` to a long random string. Then:

- `/` and `/health` stay open, so the page can load and the healthcheck works
- everything that generates audio, returns audio, or lists projects requires the token
- open `https://<POD_ID>-8000.proxy.runpod.net/?token=<your-token>` and the browser
  remembers it for the session; without it the page prompts you

```bash
curl -H "X-API-Token: <your-token>" https://<POD_ID>-8000.proxy.runpod.net/api/styles
```

Leaving it unset is fine only if the pod is not publicly reachable.

---

## Run Phase 0 on the same pod

The gate scripts talk to ACE-Step directly, so run them from the pod's terminal once
the engine is up:

```bash
cd /workspace/MMaker
pip install -r research/requirements.txt
export ACESTEP_URL=http://127.0.0.1:8001

python research/probe.py                        # sample rate, cold start, real API schema
python research/lego_chain.py                   # drums -> bass -> keys -> vocals
python research/sync_check.py research/out/lego_chain_<id>
python research/repaint_region.py research/out/lego_chain_<id> 02_bass.wav 9 13
```

**Listen to `99_final_mix.wav` before building on any of this.** ACE Studio 2.0 ships the
same "add a layer" feature and the trade press called it unusable for professional work.
Phase 0 is where you find out whether your output clears that bar.

---

## Enabling the other models

`server/models.yaml` ships with only ACE-Step enabled, so first boot pulls ~12 GB rather
than 40. As you need them:

| Model | Phase | Before enabling |
|---|---|---|
| `demucs` | 3 — stem separation | Should work as written |
| `seedvc` | 5 — voice library | **Verify the repo id**, and wire `SeedVCEngine._load()` — it currently raises on purpose |
| `stableaudio` | 6 — SFX | **Verify the repo id** |

The Seed-VC and Stable Audio repo ids in `models.yaml` are placeholders I could not
confirm. Check them on Hugging Face from the pod before flipping `enabled: true`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Boot screen never leaves "Downloading" | Weights still coming down | Check pod logs; 12 GB takes a while |
| `ACE-Step not found at /opt/ACE-Step-1.5` | Running the image without the ACE-Step layer | Rebuild, or run with `MUSICMAKER_ENGINE=stub` |
| `ACE-Step exited during startup` | Its install or GPU driver | Read the ACE-Step lines in the pod log; try `uv run acestep-api` by hand |
| Everything says `sync unmeasured` | librosa missing | It is in `requirements-pod.txt`; on a hand-rolled pod, `pip install librosa` |
| Layers sound right but drift | Real drift, not a bug | `sync_check.py` quantifies it; see the sync section in `README.md` |
| 401 on every call | Token set but not supplied | Open with `?token=...` |
| MIDI upload fails | No SoundFont | The image installs `fluid-soundfont-gm`; on a hand-rolled pod, `apt-get install fluid-soundfont-gm` |
