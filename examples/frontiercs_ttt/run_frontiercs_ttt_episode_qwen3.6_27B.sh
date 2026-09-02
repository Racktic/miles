#!/usr/bin/env bash
# Qwen3.6-27B full-parameter Frontier-CS training entrypoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

export FRONTIERCS_MODEL_CONFIG_SCRIPT="${FRONTIERCS_MODEL_CONFIG_SCRIPT:-${MILES_DIR}/scripts/models/qwen3.6-27B.sh}"
export FRONTIERCS_MODEL_LABEL="${FRONTIERCS_MODEL_LABEL:-qwen3.6-27B}"
export FRONTIERCS_TP="${FRONTIERCS_TP:-4}"

if [ ! -f "${FRONTIERCS_MODEL_CONFIG_SCRIPT}" ]; then
  echo "Missing Qwen3.6-27B model configuration: ${FRONTIERCS_MODEL_CONFIG_SCRIPT}" >&2
  exit 2
fi

if [ "$#" -gt 0 ]; then
  exec bash "${SCRIPT_DIR}/run_frontiercs_ttt_multinode.sh" "$@"
fi

exec bash "${SCRIPT_DIR}/run_frontiercs_ttt_episode_qwen3.5_4B.sh"
