#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <ssh-host> <image> [train args...]" >&2
  exit 1
fi

SSH_HOST="$1"
IMAGE="$2"
shift 2

REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/srv/waferlab/data}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/srv/waferlab/outputs}"
REMOTE_CONTAINER_NAME="${REMOTE_CONTAINER_NAME:-waferlab-train}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/workspace}"

TRAIN_ARGS="$*"

ssh "${SSH_HOST}" \
  "mkdir -p '${REMOTE_DATA_ROOT}' '${REMOTE_OUTPUT_ROOT}' && \
   nvidia-smi >/dev/null && \
   docker pull '${IMAGE}' && \
   docker rm -f '${REMOTE_CONTAINER_NAME}' >/dev/null 2>&1 || true && \
   docker run --name '${REMOTE_CONTAINER_NAME}' --gpus all --rm \
     -e WAFERLAB_DATA_ROOT='${REMOTE_WORKDIR}/data' \
     -e WAFERLAB_OUTPUT_ROOT='${REMOTE_WORKDIR}/outputs' \
     -v '${REMOTE_DATA_ROOT}:${REMOTE_WORKDIR}/data' \
     -v '${REMOTE_OUTPUT_ROOT}:${REMOTE_WORKDIR}/outputs' \
     '${IMAGE}' python scripts/train_classifier.py ${TRAIN_ARGS}"
