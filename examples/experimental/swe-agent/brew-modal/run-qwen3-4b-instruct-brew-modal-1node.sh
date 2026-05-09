#!/usr/bin/env bash
# One-node defaults for the Brew Modal SWE-agent training example.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export NUM_NODES="${NUM_NODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export LOCAL_RAY_GPUS="${LOCAL_RAY_GPUS:-${GPUS_PER_NODE}}"
export RAY_MODE="${RAY_MODE:-local}"
export RAY_API_SERVER="${RAY_API_SERVER:-http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}}"
export RUN_ID="${RUN_ID:-miles_r2egym_ohcore_qwen3_4b_1node_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run-qwen3-4b-instruct-brew-modal-common.sh"
