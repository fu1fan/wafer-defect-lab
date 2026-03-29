FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/src \
    WAFERLAB_DATA_ROOT=/workspace/data \
    WAFERLAB_OUTPUT_ROOT=/workspace/outputs

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./
RUN python -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --upgrade pip \
    && "${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY README.md ./

CMD ["python", "scripts/train_classifier.py"]
