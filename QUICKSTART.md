# Quickstart

## On RunPod

1. Create a pod from `runpod-template.json` — any CUDA GPU, **network volume mounted
   at `/workspace`** (100 GB is comfortable), HTTP port 8000 exposed.
2. Wait for first boot. Weights download once to the volume; `/health` reports progress.
3. Open the pod's HTTP URL. The UI is served by the pod itself — nothing to install locally.

Later boots skip the download and are ready in seconds.

## Locally (no GPU)

Stub mode fakes the model downloads so the whole boot flow can be exercised on a laptop:

```bash
cd server
pip install -r requirements.txt
MUSICMAKER_STUB_MODELS=1 \
MUSICMAKER_MODELS_DIR=./_models \
MUSICMAKER_DATA_DIR=./_data \
python -m uvicorn app.main:app --reload --port 8000
```

Then open http://127.0.0.1:8000

## Tests

```bash
cd server
python -m pytest test_boot.py test_pipeline.py -q     # 29 tests, no GPU needed
```

`test_boot.py` covers startup and model download; `test_pipeline.py` drives the whole
product — idea, takes, split, repaint, extend, vocals, voices, SFX, export — against the
stub engine.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MUSICMAKER_MODELS_DIR` | `/workspace/models` | Weights — put this on the network volume |
| `MUSICMAKER_DATA_DIR` | `/workspace/data` | Generated audio and project state |
| `MUSICMAKER_PORT` | `8000` | |
| `MUSICMAKER_CORS_ORIGINS` | `*` | Restrict if a pod is publicly reachable |
| `MUSICMAKER_STUB_MODELS` | `0` | Fake downloads for local dev. **Never set on a pod.** |
| `MUSICMAKER_ENGINE` | *(auto)* | Set to `stub` to force the synthetic engine |
| `ACESTEP_URL` | `http://127.0.0.1:8001` | Where the ACE-Step server listens |
| `MUSICMAKER_REGISTRY` | `server/models.yaml` | Model registry path |

## Enabling later-phase models

`server/models.yaml` ships with only ACE-Step enabled. Demucs, Seed-VC, and Stable
Audio 3 are listed but disabled, and **their repo ids are unverified placeholders**.
Confirm each id on Hugging Face from the pod before flipping `enabled: true` —
do not trust the ids as written.

## Using it

1. **Idea** — type a description, or drop a MIDI file or a recording. From a file the
   app detects tempo and key and **stops to let you confirm them**: everything you layer
   locks to that grid, so a wrong key would corrupt every stem downstream.
2. **Takes** — same grid, different seeds. Play them, pick one.
3. **Split** — Demucs separates it into stems.
4. **Edit** — drag across any lane to select bars, then regenerate just those. Or take
   another pass at a whole stem, extend it, or change the voice on a vocal.
5. **Export** — stems plus a manifest with bpm and key, ready for FL Studio.

Every edit appends a version. Nothing is overwritten, so switching back is free.
