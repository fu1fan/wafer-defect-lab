ARG BASE_IMAGE=ghcr.io/fu1fan/wafer-defect-lab-base:cu130
FROM ${BASE_IMAGE}

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY README.md ./

CMD ["python", "scripts/train_classifier.py"]
