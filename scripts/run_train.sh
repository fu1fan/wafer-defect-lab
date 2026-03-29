#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export WAFERLAB_DATA_ROOT="${WAFERLAB_DATA_ROOT:-${ROOT_DIR}/data}"
export WAFERLAB_OUTPUT_ROOT="${WAFERLAB_OUTPUT_ROOT:-${ROOT_DIR}/outputs}"

python "${ROOT_DIR}/scripts/train_classifier.py" "$@"
