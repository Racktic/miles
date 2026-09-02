#!/usr/bin/env bash
# Build a homogeneous multi-node Ray cluster and submit Frontier-CS training.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROLE="${1:-${FRONTIERCS_RAY_ROLE:-}}"
HEAD_ADDR="${2:-${FRONTIERCS_RAY_HEAD_ADDR:-}}"

if [ "${ROLE}" != "head" ] && [ "${ROLE}" != "worker" ]; then
  echo "Usage: $0 <head|worker> <head-address>" >&2
  exit 2
fi
if [ -z "${HEAD_ADDR}" ]; then
  echo "A head address is required as argument 2 or FRONTIERCS_RAY_HEAD_ADDR." >&2
  exit 2
fi

NGPU="${FRONTIERCS_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
GPUS_PER_NODE="${FRONTIERCS_ACTOR_GPUS_PER_NODE:-${NGPU}}"
RAY_NODE_GPUS="${FRONTIERCS_RAY_NODE_GPUS:-${GPUS_PER_NODE}}"
GCS_PORT="${RAY_GCS_PORT:-6379}"
DASH_PORT="${RAY_DASH_PORT:-8265}"
CONNECT_TIMEOUT="${FRONTIERCS_RAY_CONNECT_TIMEOUT_SECONDS:-600}"

if ! [[ "${NGPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRONTIERCS_NGPU must be a positive integer; detected ${NGPU}." >&2
  exit 2
fi
if ! [[ "${RAY_NODE_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRONTIERCS_RAY_NODE_GPUS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] \
  || [ "${RAY_NODE_GPUS}" -lt "${GPUS_PER_NODE}" ]; then
  echo "FRONTIERCS_ACTOR_GPUS_PER_NODE must be positive and no larger than FRONTIERCS_RAY_NODE_GPUS." >&2
  exit 2
fi
if [ "${RAY_NODE_GPUS}" -gt "${NGPU}" ]; then
  echo "Ray cannot advertise ${RAY_NODE_GPUS} GPUs with only ${NGPU} visible." >&2
  exit 2
fi
if ! [[ "${CONNECT_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRONTIERCS_RAY_CONNECT_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi

if [ "${FRONTIERCS_RESET_RAY:-0}" = "1" ]; then
  ray stop --force >/dev/null 2>&1 || true
fi

if [ "${ROLE}" = "worker" ]; then
  deadline=$((SECONDS + CONNECT_TIMEOUT))
  echo "[frontiercs-ray] waiting for head ${HEAD_ADDR}:${GCS_PORT}"
  until (echo >/dev/tcp/"${HEAD_ADDR}"/"${GCS_PORT}") >/dev/null 2>&1; do
    if [ "${SECONDS}" -ge "${deadline}" ]; then
      echo "Timed out waiting for Ray head ${HEAD_ADDR}:${GCS_PORT}." >&2
      exit 1
    fi
    sleep 5
  done

  RAY_WORKER_ARGS=(
    --address="${HEAD_ADDR}:${GCS_PORT}"
    --num-gpus="${RAY_NODE_GPUS}"
    --disable-usage-stats
  )
  if [ -n "${FRONTIERCS_RAY_NODE_IP:-}" ]; then
    RAY_WORKER_ARGS+=(--node-ip-address="${FRONTIERCS_RAY_NODE_IP}")
  fi
  if [ -n "${RAY_OBJECT_STORE_MEMORY:-}" ]; then
    RAY_WORKER_ARGS+=(--object-store-memory="${RAY_OBJECT_STORE_MEMORY}")
  fi
  echo "[frontiercs-ray] joining ${HEAD_ADDR}:${GCS_PORT} with ${RAY_NODE_GPUS} GPUs"
  ray start "${RAY_WORKER_ARGS[@]}"

  cleanup_worker_ray() {
    ray stop --force >/dev/null 2>&1 || true
  }
  trap cleanup_worker_ray EXIT
  trap 'exit 0' INT TERM
  missing_checks=0
  missing_limit="${FRONTIERCS_RAY_HEAD_MISSING_CHECKS:-6}"
  if ! [[ "${missing_limit}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FRONTIERCS_RAY_HEAD_MISSING_CHECKS must be a positive integer." >&2
    exit 2
  fi
  while true; do
    sleep "${FRONTIERCS_RAY_WORKER_MONITOR_SECONDS:-5}"
    if (echo >/dev/tcp/"${HEAD_ADDR}"/"${GCS_PORT}") >/dev/null 2>&1; then
      missing_checks=0
    else
      missing_checks=$((missing_checks + 1))
      if [ "${missing_checks}" -ge "${missing_limit}" ]; then
        echo "[frontiercs-ray] head is gone; stopping worker Ray services"
        exit 0
      fi
    fi
  done
fi

ACTOR_NUM_NODES="${FRONTIERCS_ACTOR_NUM_NODES:-}"
if ! [[ "${ACTOR_NUM_NODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "Set FRONTIERCS_ACTOR_NUM_NODES to the total number of GPU nodes." >&2
  exit 2
fi

export MASTER_ADDR="${HEAD_ADDR}"
export FRONTIERCS_NGPU="${NGPU}"
export FRONTIERCS_ACTOR_GPUS_PER_NODE="${GPUS_PER_NODE}"
export FRONTIERCS_RAY_NODE_GPUS="${RAY_NODE_GPUS}"
export FRONTIERCS_START_RAY=1
export FRONTIERCS_RAY_CLUSTER_ADDRESS="${HEAD_ADDR}:${GCS_PORT}"
export FRONTIERCS_RAY_ADDRESS="http://${HEAD_ADDR}:${DASH_PORT}"
export FRONTIERCS_ROLLOUT_NUM_GPUS="${FRONTIERCS_ROLLOUT_NUM_GPUS:-$((ACTOR_NUM_NODES * GPUS_PER_NODE))}"
export FRONTIERCS_STOP_RAY_ON_EXIT="${FRONTIERCS_STOP_RAY_ON_EXIT:-1}"

exec bash "${SCRIPT_DIR}/run_frontiercs_ttt_episode_qwen3.5_4B.sh"
