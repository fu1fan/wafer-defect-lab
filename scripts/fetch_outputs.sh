#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <ssh-host> [remote-subdir]" >&2
  exit 1
fi

SSH_HOST="$1"
REMOTE_SUBDIR="${2:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-${ROOT_DIR}/outputs}"
REMOTE_OUTPUT_ROOT="${REMOTE_OUTPUT_ROOT:-/srv/waferlab/outputs}"

mkdir -p "${LOCAL_OUTPUT_ROOT}"

REMOTE_PATH="${REMOTE_OUTPUT_ROOT}"
if [[ -n "${REMOTE_SUBDIR}" ]]; then
  REMOTE_PATH="${REMOTE_OUTPUT_ROOT%/}/${REMOTE_SUBDIR}"
fi

rsync -avz "${SSH_HOST}:${REMOTE_PATH}/" "${LOCAL_OUTPUT_ROOT}/"
