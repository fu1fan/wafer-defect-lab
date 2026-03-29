#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-wafer-defect-lab:local}"
DATA_ROOT="${WAFERLAB_DATA_ROOT:-${ROOT_DIR}/data}"
OUTPUT_ROOT="${WAFERLAB_OUTPUT_ROOT:-${ROOT_DIR}/outputs}"

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

docker run --rm \
  "${GPU_ARGS[@]}" \
  -e WAFERLAB_DATA_ROOT=/workspace/data \
  -e WAFERLAB_OUTPUT_ROOT=/workspace/outputs \
  -v "${DATA_ROOT}:/workspace/data" \
  -v "${OUTPUT_ROOT}:/workspace/outputs" \
  "${IMAGE}" python scripts/train_classifier.py "$@"
