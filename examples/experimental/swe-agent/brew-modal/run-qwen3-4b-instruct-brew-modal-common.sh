#!/usr/bin/env bash
# Shared Miles Megatron + SGLang launcher for Brew Modal SWE-agent rollouts.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../../../.." >/dev/null 2>&1 && pwd)"
TRAIN_MILES_DIR="${TRAIN_MILES_DIR:-${MILES_DIR}}"

RUN_ID="${RUN_ID:-miles_swe_brew_modal_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${MILES_DIR}/runs/${RUN_ID}}"
DATASET_KIND="${DATASET_KIND:-r2e-gym}"
RAW_PROMPT_DATA="${RAW_PROMPT_DATA:-/root/r2egym_train64.raw.jsonl}"
PROMPT_DATA="${PROMPT_DATA:-${RUN_DIR}/prompt_data.brew_modal.jsonl}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"

HF_CHECKPOINT="${HF_CHECKPOINT:-/root/qwen3-4B-Instruct-2507}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-/root/qwen3-4B-Instruct-2507_torch_dist}"
SAVE_PATH="${SAVE_PATH:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-raw}"

NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
LOCAL_RAY_GPUS="${LOCAL_RAY_GPUS:-${GPUS_PER_NODE}}"
RAY_MODE="${RAY_MODE:-local}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-8899}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_API_SERVER="${RAY_API_SERVER:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_${RUN_ID:0:24}}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
STOP_LOCAL_RAY_ON_EXIT="${STOP_LOCAL_RAY_ON_EXIT:-0}"
SUBMIT_MODE="${SUBMIT_MODE:-ray-job}"

BREW_ADAPTER_URL="${BREW_ADAPTER_URL:-${SWE_AGENT_GYM_URL:-http://127.0.0.1:11001}}"
MAX_ITERATIONS="${MAX_ITERATIONS:-25}"

NUM_ROLLOUT="${NUM_ROLLOUT:-5}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-65536}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-4}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-Tinyverl0.6}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_ID}}"
USE_WANDB="${USE_WANDB:-1}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_BREW_HEALTH_CHECK="${SKIP_BREW_HEALTH_CHECK:-0}"
SKIP_RAY_STATUS_CHECK="${SKIP_RAY_STATUS_CHECK:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${RUN_DIR}/failed_trajectories" "${RUN_DIR}/wandb"

export PYTHONUNBUFFERED=1
export SWE_AGENT_GYM_URL="${BREW_ADAPTER_URL}"
export SAVE_FAILED_TRAJ_DIR="${RUN_DIR}/failed_trajectories"
export WANDB_MODE
export MASTER_ADDR

exec > >(tee -a "${RUN_DIR}/train.log") 2>&1

echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "TRAIN_MILES_DIR=${TRAIN_MILES_DIR}"
echo "DATASET_KIND=${DATASET_KIND}"
echo "RAW_PROMPT_DATA=${RAW_PROMPT_DATA}"
echo "PROMPT_DATA=${PROMPT_DATA}"
echo "RAY_MODE=${RAY_MODE}"
echo "SUBMIT_MODE=${SUBMIT_MODE}"
echo "RAY_API_SERVER=${RAY_API_SERVER}"
echo "RAY_TEMP_DIR=${RAY_TEMP_DIR}"
echo "BREW_ADAPTER_URL=${BREW_ADAPTER_URL}"
echo "HF_CHECKPOINT=${HF_CHECKPOINT}"
echo "TORCH_DIST_CKPT=${TORCH_DIST_CKPT}"
echo "MEGATRON_TO_HF_MODE=${MEGATRON_TO_HF_MODE}"
echo "NUM_NODES=${NUM_NODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"

if [[ ! -d "${HF_CHECKPOINT}" ]]; then
  echo "Missing HF_CHECKPOINT directory: ${HF_CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -d "${TORCH_DIST_CKPT}" ]]; then
  echo "Missing TORCH_DIST_CKPT directory: ${TORCH_DIST_CKPT}" >&2
  exit 1
fi

if [[ ! -d "${TRAIN_MILES_DIR}" ]]; then
  echo "Missing TRAIN_MILES_DIR directory: ${TRAIN_MILES_DIR}" >&2
  exit 1
fi

if [[ ! -f "${RAW_PROMPT_DATA}" ]]; then
  echo "Missing RAW_PROMPT_DATA file: ${RAW_PROMPT_DATA}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_prompt_data.py" \
  --input "${RAW_PROMPT_DATA}" \
  --output "${PROMPT_DATA}" \
  --dataset-kind "${DATASET_KIND}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --limit "${PROMPT_LIMIT}"

"${PYTHON_BIN}" - "${PROMPT_DATA}" "${ROLLOUT_BATCH_SIZE}" <<'PY'
import json
import sys

path = sys.argv[1]
min_rows = int(sys.argv[2])
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
if len(rows) < min_rows:
    raise SystemExit(f"expected at least {min_rows} rows, got {len(rows)}")

required = {
    "runner": "oh-core",
    "runner_entrypoint": "run_oh_core",
    "env_type": "modal",
}
for index, row in enumerate(rows[:min_rows]):
    metadata = row.get("metadata") or {}
    for key, expected in required.items():
        actual = metadata.get(key)
        if actual != expected:
            raise SystemExit(f"row {index} {key} mismatch: {actual!r} != {expected!r}")
    if not metadata.get("dataset"):
        raise SystemExit(f"row {index} missing dataset")
    if not metadata.get("instance_id"):
        raise SystemExit(f"row {index} missing instance_id")
print("prompt_data_verified", len(rows), [row["metadata"]["instance_id"] for row in rows[:5]])
PY

if [[ "${SKIP_BREW_HEALTH_CHECK}" != "1" ]]; then
  "${PYTHON_BIN}" - "${BREW_ADAPTER_URL}" <<'PY'
import json
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
with urllib.request.urlopen(url, timeout=10) as resp:
    body = resp.read().decode("utf-8")
    print(f"adapter_health {resp.status} {body}")
    payload = json.loads(body)
if payload.get("status") not in {"ok", "healthy"}:
    raise SystemExit(f"adapter health check failed: {payload}")
PY
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} detected_nvlink_refs=${NVLINK_COUNT}"

source "${TRAIN_MILES_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

RUNTIME_PYTHONPATH="/root/Megatron-LM:${SCRIPT_DIR}:${RUN_DIR}:${TRAIN_MILES_DIR}"
if [[ "${MILES_DIR}" != "${TRAIN_MILES_DIR}" ]]; then
  RUNTIME_PYTHONPATH="${RUNTIME_PYTHONPATH}:${MILES_DIR}"
fi
export PYTHONPATH="${RUNTIME_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export NCCL_TIMEOUT_MS=36000000

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${TORCH_DIST_CKPT}"
)
if [[ -n "${SAVE_PATH}" ]]; then
  CKPT_ARGS+=(--save "${SAVE_PATH}" --save-interval "${SAVE_INTERVAL}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --metadata-key metadata
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.01
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-context-length "${ROLLOUT_MAX_CONTEXT_LEN}"
  --sglang-tool-call-parser "${SGLANG_TOOL_CALL_PARSER}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
  --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

CUSTOM_ARGS=(
  --custom-generate-function-path brew_modal_compat.generate
  --custom-rm-path brew_modal_compat.reward_func
  --rollout-function-path brew_modal_compat.generate_rollout
  --dynamic-sampling-filter-path brew_modal_compat.dynamic_filter
)

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-mode "${WANDB_MODE}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${RUN_DIR}/wandb"
    --disable-wandb-random-suffix
  )
  if [[ -n "${WANDB_KEY:-}" ]]; then
    WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
  fi
fi

RUNTIME_ENV_JSON="${RUN_DIR}/runtime_env.json"
"${PYTHON_BIN}" - "${RUNTIME_ENV_JSON}" "${RUNTIME_PYTHONPATH}" "${RUN_DIR}" "${BREW_ADAPTER_URL}" "${MASTER_ADDR}" "${HAS_NVLINK}" "${WANDB_MODE}" <<'PY'
import json
import sys

out, runtime_pythonpath, run_dir, adapter_url, master_addr, has_nvlink, wandb_mode = sys.argv[1:]
payload = {
    "env_vars": {
        "PYTHONPATH": runtime_pythonpath,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "SWE_AGENT_GYM_URL": adapter_url,
        "SAVE_FAILED_TRAJ_DIR": f"{run_dir}/failed_trajectories",
        "WANDB_MODE": wandb_mode,
        "MASTER_ADDR": master_addr,
        "NCCL_NVLS_ENABLE": has_nvlink,
        "NCCL_TIMEOUT_MS": "36000000",
    }
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f)
print(json.dumps(payload))
PY

if [[ "${RAY_MODE}" == "local" ]]; then
  mkdir -p "${RAY_TEMP_DIR}"
  if [[ "${STOP_LOCAL_RAY_ON_EXIT}" == "1" ]]; then
    trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
  fi
  RAY_START_ARGS=(
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${LOCAL_RAY_GPUS}" \
    --disable-usage-stats \
    --temp-dir="${RAY_TEMP_DIR}" \
    --port="${RAY_PORT}"
  )
  if [[ "${SUBMIT_MODE}" == "direct" ]]; then
    RAY_START_ARGS+=(--include-dashboard=False)
  else
    RAY_START_ARGS+=(--dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}")
  fi
  ray start "${RAY_START_ARGS[@]}"
  export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
elif [[ "${RAY_MODE}" == "existing" ]]; then
  export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"
else
  echo "RAY_MODE must be local or existing, got: ${RAY_MODE}" >&2
  exit 1
fi

if [[ "${SKIP_RAY_STATUS_CHECK}" != "1" ]]; then
  ray status | tee "${RUN_DIR}/ray_status_before_submit.txt"
  if [[ "${RAY_MODE}" == "existing" && "${REQUIRE_IDLE_GPUS}" == "1" ]]; then
    EXPECTED_GPUS=$((NUM_NODES * GPUS_PER_NODE))
    grep -q "0.0/${EXPECTED_GPUS}.0 GPU" "${RUN_DIR}/ray_status_before_submit.txt"
  fi
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" train.py
  --train-backend megatron
  --actor-num-nodes "${NUM_NODES}"
  --actor-num-gpus-per-node "${GPUS_PER_NODE}"
  --num-gpus-per-node "${GPUS_PER_NODE}"
  --colocate
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${WANDB_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
)

echo "Training command:"
printf ' %q' "${TRAIN_CMD[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; not submitting Ray job."
  exit 0
fi

if [[ "${SUBMIT_MODE}" == "direct" ]]; then
  cd "${TRAIN_MILES_DIR}"
  "${TRAIN_CMD[@]}"
  echo "Training completed: ${RUN_ID}"
  exit 0
elif [[ "${SUBMIT_MODE}" != "ray-job" ]]; then
  echo "SUBMIT_MODE must be ray-job or direct, got: ${SUBMIT_MODE}" >&2
  exit 1
fi

TRAIN_CMD_STR="$(printf ' %q' "${TRAIN_CMD[@]}")"
ray job submit --address="${RAY_API_SERVER}" \
  --runtime-env-json="$(cat "${RUNTIME_ENV_JSON}")" \
  -- /bin/bash -lc "cd $(printf '%q' "${TRAIN_MILES_DIR}") &&${TRAIN_CMD_STR}"

echo "Training completed: ${RUN_ID}"
