# musicmaker engine.
#
# Runs two processes: ACE-Step 1.5 serving the model on :8001 (localhost only)
# and the musicmaker API + UI on :8000 (the only published port).
#
# Model WEIGHTS are not baked in -- they download once to the network volume at
# /workspace on first boot, so the image stays pullable and pods stay
# interchangeable. ACE-Step's *code* is installed here so startup is predictable.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MUSICMAKER_MODELS_DIR=/workspace/models \
    MUSICMAKER_DATA_DIR=/workspace/data \
    ACESTEP_URL=http://127.0.0.1:8001 \
    ACESTEP_DIR=/opt/ACE-Step-1.5

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fluidsynth fluid-soundfont-gm libsndfile1 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-pod.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-pod.txt hf_transfer

# ACE-Step 1.5, installed with uv as its docs prescribe.
ARG ACESTEP_REF=main
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN git clone --depth 1 --branch ${ACESTEP_REF} \
        https://github.com/ace-step/ACE-Step-1.5.git ${ACESTEP_DIR} \
    && cd ${ACESTEP_DIR} \
    && uv sync

COPY app ./app
COPY static ./static
COPY models.yaml start.sh ./
RUN chmod +x start.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["./start.sh"]
