# syntax=docker/dockerfile:1.4
ARG BASE_IMAGE=ghcr.io/fu1fan/wafer-defect-lab-base:cu128
FROM ${BASE_IMAGE}

COPY --link src ./src
COPY --link scripts ./scripts
COPY --link configs ./configs
COPY --link README.md ./

CMD ["python", "scripts/train_classifier.py"]
