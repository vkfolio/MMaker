# MMSynth pod image.
#
# Three Python environments, deliberately, because the models disagree:
#
#   system python 3.10 + torch 2.2   SoulX-Singer and the MMSynth service
#   /opt/ACE-Step-1.5/.venv 3.12     ACE-Step 1.5 XL, via uv
#
# What goes in: code, virtualenvs, the soundfont, the app. All of it under /opt.
# What stays out: **model weights**. They are ~35 GB, and the network volume is
# mounted at /workspace where it would hide anything the image put there. So
# app/bootstrap.py fetches them to the volume on first boot and /health reports
# progress. Every later boot on the same volume is instant.
#
# Also out: every credential. Baking a token into an image ships it to anyone
# who can pull it.
#
# NOTE: `docker build` needs a daemon, which a RunPod pod does not have. This
# file is the reproducible definition; tools/build-image.sh is how the image is
# actually made -- crane appends a layer to the base over the registry API,
# from inside a running pod, so 20 GB never crosses a home connection.

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.uv-cache \
    MMSYNTH_DATA_DIR=/workspace/mmsynth-data \
    MMSYNTH_SOULX_DIR=/opt/SoulX-Singer \
    MMSYNTH_SOULX_WEIGHTS=/workspace/soulx-weights \
    MMSYNTH_ACESTEP_CHECKPOINTS=/workspace/checkpoints \
    ACESTEP_URL=http://127.0.0.1:8011 \
    MMSYNTH_PORT=8100

# fluidsynth is not optional: it is the melody carrier, and without it a MIDI
# file cannot reach a model that has no note input.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fluidsynth fluid-soundfont-gm libsndfile1 curl git \
    && rm -rf /var/lib/apt/lists/*

# --- ACE-Step 1.5, in its own uv venv -------------------------------------
# uv bakes absolute paths into the venv, so this path must never move.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ARG ACESTEP_REF=main
RUN git clone --depth 1 --branch ${ACESTEP_REF} \
        https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5 \
    && cd /opt/ACE-Step-1.5 && uv sync \
    && rm -rf /opt/ACE-Step-1.5/checkpoints \
    && ln -s /workspace/checkpoints /opt/ACE-Step-1.5/checkpoints

# --- SoulX-Singer, in the system python -----------------------------------
# Its inference needs far less than its requirements.txt suggests: NeMo, funasr
# and Gradio are all preprocessing or demo dependencies. numpy<2 and
# transformers==4.41.2 are load-bearing.
RUN pip install --no-cache-dir \
        "numpy<2.0.0" "transformers==4.41.2" "librosa==0.11.0" \
        omegaconf soundfile scipy accelerate tqdm g2p_en "huggingface_hub>=0.34" \
    && python -m nltk.downloader -q averaged_perceptron_tagger \
        averaged_perceptron_tagger_eng cmudict \
    && git clone --depth 1 https://github.com/Soul-AILab/SoulX-Singer.git /opt/SoulX-Singer \
    && ln -s /workspace/soulx-weights /opt/SoulX-Singer/pretrained_models

# --- MMSynth --------------------------------------------------------------
RUN pip install --no-cache-dir \
        fastapi "uvicorn[standard]" pydantic pyyaml mido python-multipart requests

WORKDIR /opt/MMSynth
COPY app ./app
COPY static ./static
COPY data ./data
COPY gate ./gate
COPY instruments.yaml models.yaml requirements.txt start.sh ./
RUN chmod +x start.sh

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8100/health || exit 1

CMD ["/opt/MMSynth/start.sh"]
