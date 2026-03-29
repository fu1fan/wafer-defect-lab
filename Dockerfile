FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src \
    WAFERLAB_DATA_ROOT=/workspace/data \
    WAFERLAB_OUTPUT_ROOT=/workspace/outputs

WORKDIR /workspace

COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY README.md ./

CMD ["python", "scripts/train_classifier.py"]
