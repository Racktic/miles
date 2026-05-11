#!/bin/bash

# Run this script inside the container on all 4 nodes. The head node starts Ray
# and submits the Miles job; worker nodes join Ray and stay alive.
set -euo pipefail

if [[ $# -gt 0 && "${1}" != --* ]]; then
   export NANOROLLOUT_URL="${1}"
   shift
fi

EXTRA_TRAIN_ARGS=("$@")

export PYTHONUNBUFFERED=1
export NANOROLLOUT_URL="${NANOROLLOUT_URL:-http://127.0.0.1:11000}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${MILES_DIR}"

NUM_NODES=4
GPUS_PER_NODE=8
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
NODE_IP="${NODE_IP:-$(hostname -I | awk '{print $1}')}"
if [[ "${NODE_RANK}" != "0" && -z "${MASTER_ADDR:-}" ]]; then
   echo "MASTER_ADDR must be set on worker nodes." >&2
   exit 1
fi
export MASTER_ADDR="${MASTER_ADDR:-${NODE_IP}}"

RAY_PORT="${RAY_PORT:-8899}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RUN_NAME="${RUN_NAME:-miles_nanorollout_qwen3_4b_4node_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-runs/${RUN_NAME}}"
[[ "${RUN_DIR}" = /* ]] || RUN_DIR="${MILES_DIR}/${RUN_DIR}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/r_${SLURM_JOB_ID:-$$}_${NODE_RANK}}"

cleanup() {
   ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

ray stop --force >/dev/null 2>&1 || true

if [[ "${NODE_RANK}" != "0" ]]; then
   deadline=$((SECONDS + 900))
   until ray status --address="${MASTER_ADDR}:${RAY_PORT}" >/dev/null 2>&1; do
      if (( SECONDS >= deadline )); then
         echo "Timed out waiting for Ray head at ${MASTER_ADDR}:${RAY_PORT}" >&2
         exit 1
      fi
      sleep 5
   done

   ray start \
      --address="${MASTER_ADDR}:${RAY_PORT}" \
      --node-ip-address "${NODE_IP}" \
      --num-gpus "${GPUS_PER_NODE}" \
      --disable-usage-stats \
      --temp-dir="${RAY_TEMP_DIR}"

   echo "Worker node rank ${NODE_RANK} joined Ray at ${MASTER_ADDR}:${RAY_PORT}; waiting for head job."
   while ray status --address="${MASTER_ADDR}:${RAY_PORT}" >/dev/null 2>&1; do
      sleep 60
   done
   echo "Ray head at ${MASTER_ADDR}:${RAY_PORT} is gone; worker node rank ${NODE_RANK} exiting."
   exit 0
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
NVLINK_COUNT="${NVLINK_COUNT:-0}"
if [ "${NVLINK_COUNT}" -gt 0 ]; then
   HAS_NVLINK=1
else
   HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

ray start \
   --head \
   --node-ip-address "${NODE_IP}" \
   --num-gpus "${GPUS_PER_NODE}" \
   --disable-usage-stats \
   --temp-dir="${RAY_TEMP_DIR}" \
   --port="${RAY_PORT}" \
   --dashboard-host=0.0.0.0 \
   --dashboard-port="${RAY_DASHBOARD_PORT}"

expected_gpus=$((NUM_NODES * GPUS_PER_NODE))
deadline=$((SECONDS + 900))
while true; do
   status="$(ray status --address="${MASTER_ADDR}:${RAY_PORT}" 2>/dev/null || true)"
   gpus="$(
      awk '
        / GPU$/ && $1 ~ /\// {
          split($1, parts, "/")
          gsub(/[^0-9.]/, "", parts[2])
          print int(parts[2])
          exit
        }
      ' <<<"${status}"
   )"
   gpus="${gpus:-0}"
   echo "ray_cluster_wait gpus=${gpus}/${expected_gpus}"
   if (( gpus >= expected_gpus )); then
      break
   fi
   if (( SECONDS >= deadline )); then
      echo "Timed out waiting for ${expected_gpus} Ray GPUs" >&2
      ray status --address="${MASTER_ADDR}:${RAY_PORT}" || true
      exit 1
   fi
   sleep 5
done

source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/root/Megatron-LM}"
MEGATRON_TO_HF_MODE="${MEGATRON_TO_HF_MODE:-bridge}"
HF_CHECKPOINT="${HF_CHECKPOINT:-models/Qwen/Qwen3-4B-Instruct-2507}"
TORCH_DIST_CKPT="${TORCH_DIST_CKPT:-${HF_CHECKPOINT%/}_torch_dist}"
if [[ "${MEGATRON_TO_HF_MODE}" == "raw" ]]; then
   REF_CHECKPOINT="${TORCH_DIST_CKPT}"
else
   REF_CHECKPOINT="${HF_CHECKPOINT}"
fi

mkdir -p "${RUN_DIR}/failed_trajectories" "${RUN_DIR}/wandb"
export SAVE_FAILED_TRAJ_DIR="${RUN_DIR}/failed_trajectories"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-miles-nanorollout}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"

RUNTIME_PYTHONPATH="${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${MILES_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${RUNTIME_PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export NCCL_TIMEOUT_MS=36000000

CKPT_ARGS=(
   --model-name Qwen3-4B-Instruct-2507
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_CHECKPOINT}"
)
if [[ -n "${LOAD_PATH:-}" ]]; then
   CKPT_ARGS+=(--load "${LOAD_PATH}")
fi
if [[ -n "${SAVE_PATH:-}" ]]; then
   CKPT_ARGS+=(--save "${SAVE_PATH}" --save-interval "${SAVE_INTERVAL:-100}")
fi

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA:-examples/nanorollout/data/skyrl_v0_293.jsonl}"
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-3000}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-64}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-16}"
   --rollout-temperature 1.0
   --rollout-max-context-len 65536
   --rollout-max-response-len 4096
   --rollout-skip-special-tokens
   --rollout-health-check-first-wait 60
   --global-batch-size "${GLOBAL_BATCH_SIZE:-1024}"
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --skip-eval-before-train
   --eval-prompt-data swe_bench_verified "${EVAL_PROMPT_DATA:-examples/nanorollout/data/swe_bench_verified.jsonl}"
   --eval-input-key prompt
   --n-samples-per-eval-prompt 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
   --log-probs-chunk-size 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
   --tis-clip 2.0
   --tis-clip-low 0.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-mode "${WANDB_MODE}"
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP:-${RUN_NAME}}"
      --wandb-experiment-name "${WANDB_EXPERIMENT_NAME:-${RUN_NAME}}"
      --wandb-dir "${RUN_DIR}/wandb"
      --disable-wandb-random-suffix
   )
   if [[ -n "${WANDB_ENTITY}" ]]; then
      WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY}")
   fi
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-context-length 65536
   --sglang-tool-call-parser qwen
   --sglang-mem-fraction-static 0.65
   --sglang-enable-metrics
)

AGENT_ARGS=(
   --agent-runner oh-lite
   --agent-max-iterations 35
   --agent-task-timeout 6000
   --agent-step-timeout 600
   --agent-eval-timeout 600
   --agent-env-timeout 120
   --agent-create-timeout 600
   --agent-filter-overlong true
)

MISC_ARGS=(
   --wandb-experiment-name "${WANDB_EXPERIMENT_NAME:-${RUN_NAME}}"
   --megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.nanorollout.generate_with_nanorollout.generate
   --custom-rm-path examples.nanorollout.generate_with_nanorollout.reward_func
   --rollout-function-path examples.nanorollout.generate_with_nanorollout.generate_rollout
   --dynamic-sampling-filter-path examples.nanorollout.generate_with_nanorollout.dynamic_filter
   --tito
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NANOROLLOUT_URL\": \"${NANOROLLOUT_URL}\",
    \"SAVE_FAILED_TRAJ_DIR\": \"${SAVE_FAILED_TRAJ_DIR}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT}\",
    \"WANDB_ENTITY\": \"${WANDB_ENTITY}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_TIMEOUT_MS\": \"36000000\"
  }
}"

ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes "${NUM_NODES}" \
   --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
   --num-gpus-per-node "${GPUS_PER_NODE}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${AGENT_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}" \
   "${EXTRA_TRAIN_ARGS[@]}"
