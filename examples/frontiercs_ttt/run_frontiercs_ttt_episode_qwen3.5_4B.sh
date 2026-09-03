#!/bin/bash
# Full-parameter Frontier-CS training with complete group episodes as update units.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
FRONTIERCS_ROOT="${FRONTIERCS_ROOT:-}"
if [ -z "${FRONTIERCS_ROOT}" ] && [ -d "${MILES_DIR}/../Frontier-CS/algorithmic/problems" ]; then
  FRONTIERCS_ROOT="$(cd -- "${MILES_DIR}/../Frontier-CS" >/dev/null 2>&1 && pwd)"
fi
if [ -z "${FRONTIERCS_ROOT}" ] || [ ! -d "${FRONTIERCS_ROOT}/algorithmic/problems" ]; then
  echo "Set FRONTIERCS_ROOT to a Frontier-CS checkout containing algorithmic/problems." >&2
  exit 2
fi
FRONTIERCS_ROOT="$(cd -- "${FRONTIERCS_ROOT}" >/dev/null 2>&1 && pwd)"
PROMPT_DATA="${FRONTIERCS_PROMPT_DATA:-${SCRIPT_DIR}/data/problem_groups_30.jsonl}"
MEMORY_ROUNDS="${FRONTIERCS_MEMORY_ROUNDS:-4}"
CANDIDATES_PER_PROBLEM="${FRONTIERCS_CANDIDATES_PER_PROBLEM:-1}"
GROUP_SIZE="${FRONTIERCS_GROUP_SIZE:-3}"
GROUPS_PER_UPDATE="${FRONTIERCS_GROUPS_PER_UPDATE:-2}"
TRAIN_WRITE="${FRONTIERCS_TRAIN_WRITE:-1}"
ACT_MAX_NEW_TOKENS="${FRONTIERCS_ACT_MAX_NEW_TOKENS:-25600}"
WRITE_MAX_NEW_TOKENS="${FRONTIERCS_WRITE_MAX_NEW_TOKENS:-25600}"
WRITER_MAX_PROMPT_CHARS="${FRONTIERCS_WRITER_MAX_PROMPT_CHARS:-120000}"
DIAGNOSTICS_CHARS="${FRONTIERCS_DIAGNOSTICS_CHARS:-12000}"
JUDGE_TIMEOUT_SECONDS="${FRONTIERCS_JUDGE_TIMEOUT_SECONDS:-1800}"
JUDGE_POLL_SECONDS="${FRONTIERCS_JUDGE_POLL_SECONDS:-1}"
ACT_EXPLORE_BETA="${FRONTIERCS_ACT_EXPLORE_BETA:-0}"
EXPLORE_JUDGE_API_KEY="${FRONTIERCS_EXPLORE_JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
EXPLORE_JUDGE_API_BASE="${FRONTIERCS_EXPLORE_JUDGE_API_BASE:-https://api.openai.com/v1}"
EXPLORE_JUDGE_MODEL="${FRONTIERCS_EXPLORE_JUDGE_MODEL:-gpt-5-mini}"
EXPLORE_JUDGE_TIMEOUT="${FRONTIERCS_EXPLORE_JUDGE_TIMEOUT:-60}"
EXPLORE_JUDGE_CONCURRENCY="${FRONTIERCS_EXPLORE_JUDGE_CONCURRENCY:-64}"
SEQ_LENGTH="${FRONTIERCS_SEQ_LENGTH:-32768}"
MAX_TOKENS_PER_GPU="${FRONTIERCS_MAX_TOKENS_PER_GPU:-${SEQ_LENGTH}}"
NUM_EPOCHS="${FRONTIERCS_NUM_EPOCHS:-1}"
KL_MODE="${FRONTIERCS_KL_MODE:-loss}"
KL_COEF="${FRONTIERCS_KL_COEF:-${FRONTIERCS_KL:-0.01}}"
KL_TYPE="${FRONTIERCS_KL_TYPE:-k1}"
USE_UNBIASED_KL="${FRONTIERCS_USE_UNBIASED_KL:-0}"
REF_UPDATE_INTERVAL="${FRONTIERCS_REF_UPDATE_INTERVAL:-}"
RUN_ID="${FRONTIERCS_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
MODEL_LABEL="${FRONTIERCS_MODEL_LABEL:-qwen3.5-4B}"
OUTPUT_ROOT="${FRONTIERCS_OUTPUT_ROOT:-${FRONTIERCS_ROOT}/qwen_eval/results/frontiercs_ttt_rl}"
JUDGE_PORT="${FRONTIERCS_JUDGE_PORT:-8081}"
GOJUDGE_PORT="${FRONTIERCS_GOJUDGE_PORT:-5050}"
JUDGE_URL="${FRONTIERCS_JUDGE_URL:-http://127.0.0.1:${JUDGE_PORT}}"
if [ ! -f "${PROMPT_DATA}" ]; then
  echo "Missing group dataset: ${PROMPT_DATA}" >&2
  exit 2
fi
GROUP_COUNT="$(awk 'NF { count += 1 } END { print count + 0 }' "${PROMPT_DATA}")"
if [ "${GROUP_COUNT}" -lt 1 ]; then
  echo "The group dataset is empty: ${PROMPT_DATA}" >&2
  exit 2
fi

if [ "${GROUPS_PER_UPDATE}" -lt 1 ]; then
  echo "FRONTIERCS_GROUPS_PER_UPDATE must be at least 1" >&2
  exit 2
fi
if [ "${GROUP_SIZE}" -lt 1 ]; then
  echo "FRONTIERCS_GROUP_SIZE must be at least 1" >&2
  exit 2
fi
if [ "${MEMORY_ROUNDS}" -lt 1 ] || [ "${CANDIDATES_PER_PROBLEM}" -lt 1 ]; then
  echo "FRONTIERCS_MEMORY_ROUNDS and FRONTIERCS_CANDIDATES_PER_PROBLEM must be at least 1" >&2
  exit 2
fi
case "${TRAIN_WRITE}" in
  0|1) ;;
  *)
    echo "FRONTIERCS_TRAIN_WRITE must be 0 or 1" >&2
    exit 2
    ;;
esac
if ! EXPLORE_ENABLED="$(python3 -c 'import sys; value=float(sys.argv[1]); assert value >= 0; print(int(value > 0))' "${ACT_EXPLORE_BETA}" 2>/dev/null)"; then
  echo "FRONTIERCS_ACT_EXPLORE_BETA must be a non-negative number" >&2
  exit 2
fi
if [ "${EXPLORE_ENABLED}" = "1" ] && [ -z "${EXPLORE_JUDGE_API_KEY}" ]; then
  echo "Exploration reward is enabled but FRONTIERCS_EXPLORE_JUDGE_API_KEY/OPENAI_API_KEY is empty" >&2
  exit 2
fi
case "${KL_MODE}" in
  loss|none) ;;
  reward)
    echo "FRONTIERCS_KL_MODE=reward is not supported by the current custom-advantage GRPO path; use loss or none" >&2
    exit 2
    ;;
  *)
    echo "FRONTIERCS_KL_MODE must be loss or none" >&2
    exit 2
    ;;
esac
case "${KL_TYPE}" in
  k1|k2|k3|low_var_kl) ;;
  *)
    echo "FRONTIERCS_KL_TYPE must be k1, k2, k3, or low_var_kl" >&2
    exit 2
    ;;
esac
case "${USE_UNBIASED_KL}" in
  0|1) ;;
  *)
    echo "FRONTIERCS_USE_UNBIASED_KL must be 0 or 1" >&2
    exit 2
    ;;
esac
if [ -n "${REF_UPDATE_INTERVAL}" ] && ! [[ "${REF_UPDATE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRONTIERCS_REF_UPDATE_INTERVAL must be empty or a positive integer" >&2
  exit 2
fi

KL_ARGS=(--kl-loss-type "${KL_TYPE}" --kl-coef 0.0)
if [ "${KL_MODE}" = "loss" ]; then
  KL_ARGS+=(--use-kl-loss --kl-loss-coef "${KL_COEF}")
  if [ "${USE_UNBIASED_KL}" = "1" ]; then
    KL_ARGS+=(--use-unbiased-kl)
  fi
else
  KL_ARGS+=(--kl-loss-coef 0.0)
fi
if [ -n "${REF_UPDATE_INTERVAL}" ]; then
  KL_ARGS+=(--ref-update-interval "${REF_UPDATE_INTERVAL}")
fi
UPDATES_PER_EPOCH=$(((GROUP_COUNT + GROUPS_PER_UPDATE - 1) / GROUPS_PER_UPDATE))
NUM_UPDATES="${FRONTIERCS_NUM_UPDATES:-$((NUM_EPOCHS * UPDATES_PER_EPOCH))}"

NGPU="${FRONTIERCS_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${NGPU:-0}" -lt 1 ]; then
  echo "No visible GPU; enter an exclusive GPU allocation first." >&2
  exit 2
fi
TP="${FRONTIERCS_TP:-${NGPU}}"
PP="${FRONTIERCS_PP:-1}"
CP="${FRONTIERCS_CP:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
export PYTHONUNBUFFERED=1
AUTO_JUDGE_STARTED=0
RAY_HEAD_STARTED=0

cleanup_frontiercs_services() {
  if [ "${AUTO_JUDGE_STARTED:-0}" = "1" ]; then
    python3 "${SCRIPT_DIR}/host_judge.py" stop \
      --api-port "${JUDGE_PORT}" \
      --gojudge-port "${GOJUDGE_PORT}" \
      --cleanup
  fi
  if [ "${RAY_HEAD_STARTED:-0}" = "1" ] \
    && [ "${FRONTIERCS_STOP_RAY_ON_EXIT:-0}" = "1" ]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap cleanup_frontiercs_services EXIT

if ! curl -fsS "${JUDGE_URL}/health" >/dev/null && [ "${FRONTIERCS_AUTO_START_JUDGE:-0}" = "1" ]; then
  if [ "${JUDGE_URL}" != "http://127.0.0.1:${JUDGE_PORT}" ]; then
    echo "Automatic judge startup requires a local FRONTIERCS_JUDGE_URL matching FRONTIERCS_JUDGE_PORT" >&2
    exit 2
  fi
  python3 "${SCRIPT_DIR}/host_judge.py" start \
    --frontiercs-root "${FRONTIERCS_ROOT}" \
    --api-port "${JUDGE_PORT}" \
    --gojudge-port "${GOJUDGE_PORT}"
  AUTO_JUDGE_STARTED=1
fi
if ! curl -fsS "${JUDGE_URL}/health" >/dev/null; then
  echo "Frontier-CS judge is not healthy at ${JUDGE_URL}/health" >&2
  exit 2
fi

cd "${MILES_DIR}"
MODEL_CONFIG_SCRIPT="${FRONTIERCS_MODEL_CONFIG_SCRIPT:-${MILES_DIR}/scripts/models/qwen3.5-4B.sh}"
if [ ! -f "${MODEL_CONFIG_SCRIPT}" ]; then
  echo "Missing Miles model configuration: ${MODEL_CONFIG_SCRIPT}" >&2
  exit 2
fi
source "${MODEL_CONFIG_SCRIPT}"

HF_CHECKPOINT="${FRONTIERCS_HF_CHECKPOINT:-}"
TORCH_DIST="${FRONTIERCS_TORCH_DIST:-}"
if [ -z "${HF_CHECKPOINT}" ] || [ ! -d "${HF_CHECKPOINT}" ]; then
  echo "Set FRONTIERCS_HF_CHECKPOINT to the local Hugging Face checkpoint directory." >&2
  exit 2
fi
if [ -z "${TORCH_DIST}" ] || [ ! -d "${TORCH_DIST}" ]; then
  echo "Set FRONTIERCS_TORCH_DIST to the converted Miles distributed checkpoint directory." >&2
  exit 2
fi
SAVE_DIR="${FRONTIERCS_SAVE_DIR:-${OUTPUT_ROOT}/${RUN_ID}/checkpoints}"
LOG_DIR="${FRONTIERCS_LOG_DIR:-${OUTPUT_ROOT}/${RUN_ID}/logs}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/${RUN_ID}" "${SAVE_DIR}"

WANDB_ENV_FILE="${FRONTIERCS_WANDB_ENV_FILE:-}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -n "${WANDB_ENV_FILE}" ] && [ -f "${WANDB_ENV_FILE}" ]; then
  set -a
  . "${WANDB_ENV_FILE}"
  set +a
fi
WANDB_ARGS=()
WANDB_ENABLED=0
case "${FRONTIERCS_USE_WANDB:-1}" in
  ""|0|false|False|off|Off)
    ;;
  *)
    if [ -n "${WANDB_API_KEY:-}" ] || [ "${WANDB_MODE:-}" = "offline" ]; then
      WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT:-miles-frontier-cs}"
        --wandb-group "${WANDB_GROUP:-frontiercs-${MODEL_LABEL}-${RUN_ID}}"
        --disable-wandb-random-suffix
      )
      if [ -n "${FRONTIERCS_WANDB_RUN_ID:-}" ]; then
        WANDB_ARGS+=(--wandb-run-id "${FRONTIERCS_WANDB_RUN_ID}")
      fi
      if [ -n "${WANDB_TEAM:-}" ]; then
        WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
      fi
      if [ -n "${WANDB_MODE:-}" ]; then
        WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
      fi
      if [ -n "${FRONTIERCS_WANDB_DIR:-}" ]; then
        WANDB_ARGS+=(--wandb-dir "${FRONTIERCS_WANDB_DIR}")
      fi
      WANDB_ENABLED=1
    else
      echo "W&B disabled: no WANDB_API_KEY is available (set WANDB_MODE=offline for local logging)" >&2
    fi
    ;;
esac

echo "Frontier-CS complete-episode TTT run: ${RUN_ID}"
echo "model=${MODEL_LABEL} model_config=${MODEL_CONFIG_SCRIPT}"
echo "groups=${GROUP_COUNT} G=${GROUP_SIZE} groups_per_update=${GROUPS_PER_UPDATE} rounds=${MEMORY_ROUNDS} K=${CANDIDATES_PER_PROBLEM} train_write=${TRAIN_WRITE}"
echo "act_explore_beta=${ACT_EXPLORE_BETA} explore_judge_model=${EXPLORE_JUDGE_MODEL} terminal_memory_enabled=${EXPLORE_ENABLED}"
echo "epochs=${NUM_EPOCHS} optimizer_steps_per_epoch=${UPDATES_PER_EPOCH} total_optimizer_steps=${NUM_UPDATES}"
echo "kl_mode=${KL_MODE} kl_coef=${KL_COEF} kl_type=${KL_TYPE} unbiased_kl=${USE_UNBIASED_KL} ref_update_interval=${REF_UPDATE_INTERVAL:-frozen}"
echo "global_batch_size=derived dynamically from the complete episodes returned in each step"
echo "traces=${OUTPUT_ROOT}/${RUN_ID}"
echo "checkpoints=${SAVE_DIR}"
echo "wandb_enabled=${WANDB_ENABLED} wandb_project=${WANDB_PROJECT:-miles-frontier-cs}"

if [ "${FRONTIERCS_RESET_RAY:-0}" = "1" ]; then
  ray stop --force >/dev/null 2>&1 || true
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASH_PORT="${RAY_DASH_PORT:-8265}"
RAY_ADDRESS="${FRONTIERCS_RAY_ADDRESS:-http://127.0.0.1:${RAY_DASH_PORT}}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-}"
ACTOR_NUM_NODES="${FRONTIERCS_ACTOR_NUM_NODES:-1}"
ACTOR_GPUS_PER_NODE="${FRONTIERCS_ACTOR_GPUS_PER_NODE:-${NGPU}}"
RAY_NODE_GPUS="${FRONTIERCS_RAY_NODE_GPUS:-${ACTOR_GPUS_PER_NODE}}"
ROLLOUT_NUM_GPUS="${FRONTIERCS_ROLLOUT_NUM_GPUS:-$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))}"
ROLLOUT_GPUS_PER_ENGINE="${FRONTIERCS_ROLLOUT_GPUS_PER_ENGINE:-1}"
if ! [[ "${ACTOR_NUM_NODES}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${ACTOR_GPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRONTIERCS_ACTOR_NUM_NODES and FRONTIERCS_ACTOR_GPUS_PER_NODE must be positive integers." >&2
  exit 2
fi
if ! [[ "${RAY_NODE_GPUS}" =~ ^[1-9][0-9]*$ ]] || [ "${RAY_NODE_GPUS}" -lt "${ACTOR_GPUS_PER_NODE}" ]; then
  echo "FRONTIERCS_RAY_NODE_GPUS must be an integer at least as large as FRONTIERCS_ACTOR_GPUS_PER_NODE." >&2
  exit 2
fi
if [ "${RAY_NODE_GPUS}" -gt "${NGPU}" ]; then
  echo "FRONTIERCS_RAY_NODE_GPUS cannot exceed the ${NGPU} locally visible GPUs." >&2
  exit 2
fi
if ! [[ "${TP}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${PP}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${CP}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FRONTIERCS_TP, FRONTIERCS_PP, and FRONTIERCS_CP must be positive integers." >&2
  exit 2
fi
TOTAL_ACTOR_GPUS=$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))
MODEL_PARALLEL_SIZE=$((TP * PP * CP))
if [ $((TOTAL_ACTOR_GPUS % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
  echo "Total actor GPUs (${TOTAL_ACTOR_GPUS}) must be divisible by TP*PP*CP (${MODEL_PARALLEL_SIZE})." >&2
  exit 2
fi
DP_SIZE=$((TOTAL_ACTOR_GPUS / MODEL_PARALLEL_SIZE))
SAMPLES_PER_EPISODE=$((GROUP_SIZE * MEMORY_ROUNDS * CANDIDATES_PER_PROBLEM + TRAIN_WRITE * (MEMORY_ROUNDS - 1)))
SAMPLES_PER_UPDATE=$((GROUPS_PER_UPDATE * SAMPLES_PER_EPISODE))
if [ "${SAMPLES_PER_UPDATE}" -lt "${DP_SIZE}" ] \
  || [ $((SAMPLES_PER_UPDATE % DP_SIZE)) -ne 0 ]; then
  echo "One update produces ${SAMPLES_PER_UPDATE} samples, which is not divisible by DP=${DP_SIZE}." >&2
  echo "Adjust FRONTIERCS_GROUPS_PER_UPDATE or TP/PP/CP so Miles does not trim complete-episode samples." >&2
  exit 2
fi
echo "cluster_nodes=${ACTOR_NUM_NODES} gpus_per_node=${ACTOR_GPUS_PER_NODE} total_gpus=${TOTAL_ACTOR_GPUS} TP=${TP} PP=${PP} CP=${CP} DP=${DP_SIZE}"
echo "samples_per_episode=${SAMPLES_PER_EPISODE} samples_per_update=${SAMPLES_PER_UPDATE} samples_per_dp_rank=$((SAMPLES_PER_UPDATE / DP_SIZE))"
if [ "${FRONTIERCS_START_RAY:-1}" = "1" ]; then
  RAY_HEAD_ARGS=(
    --head
    --node-ip-address "${MASTER_ADDR}"
    --num-gpus "${RAY_NODE_GPUS}"
    --disable-usage-stats
    --port "${RAY_GCS_PORT:-6379}"
    --dashboard-host=0.0.0.0
    --dashboard-port "${RAY_DASH_PORT}"
  )
  if [ -n "${RAY_OBJECT_STORE_MEMORY}" ]; then
    RAY_HEAD_ARGS+=(--object-store-memory "${RAY_OBJECT_STORE_MEMORY}")
  fi
  ray start "${RAY_HEAD_ARGS[@]}"
  RAY_HEAD_STARTED=1
fi

if [ "${FRONTIERCS_WAIT_FOR_RAY_CLUSTER:-1}" = "1" ]; then
  RAY_CLUSTER_ADDRESS="${FRONTIERCS_RAY_CLUSTER_ADDRESS:-}"
  if [ -z "${RAY_CLUSTER_ADDRESS}" ]; then
    if [ "${FRONTIERCS_START_RAY:-1}" = "1" ]; then
      RAY_CLUSTER_ADDRESS="${MASTER_ADDR}:${RAY_GCS_PORT:-6379}"
    else
      RAY_CLUSTER_ADDRESS="auto"
    fi
  fi
  RAY_WAIT_ARGS=(
    --address "${RAY_CLUSTER_ADDRESS}"
    --expected-nodes "${ACTOR_NUM_NODES}"
    --expected-gpus "$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))"
    --minimum-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
    --timeout-seconds "${FRONTIERCS_RAY_WAIT_TIMEOUT_SECONDS:-600}"
    --poll-seconds "${FRONTIERCS_RAY_WAIT_POLL_SECONDS:-5}"
  )
  if [ "${FRONTIERCS_CHECK_RAY_PATHS:-1}" = "1" ]; then
    for required_path in \
      "${MILES_DIR}" \
      "${FRONTIERCS_ROOT}" \
      "${PROMPT_DATA}" \
      "${HF_CHECKPOINT}" \
      "${TORCH_DIST}" \
      "${SAVE_DIR}" \
      "${OUTPUT_ROOT}/${RUN_ID}"; do
      RAY_WAIT_ARGS+=(--required-path "${required_path}")
    done
  fi
  python3 "${SCRIPT_DIR}/wait_for_ray_cluster.py" "${RAY_WAIT_ARGS[@]}"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_DIR}:${FRONTIERCS_ROOT}\",
    \"FRONTIERCS_ROOT\": \"${FRONTIERCS_ROOT}\",
    \"FRONTIERCS_RUN_ID\": \"${RUN_ID}\",
    \"FRONTIERCS_OUTPUT_ROOT\": \"${OUTPUT_ROOT}\",
    \"FRONTIERCS_JUDGE_URL\": \"${JUDGE_URL}\",
    \"FRONTIERCS_MEMORY_ROUNDS\": \"${MEMORY_ROUNDS}\",
    \"FRONTIERCS_CANDIDATES_PER_PROBLEM\": \"${CANDIDATES_PER_PROBLEM}\",
    \"FRONTIERCS_GROUP_SIZE\": \"${GROUP_SIZE}\",
    \"FRONTIERCS_TRAIN_WRITE\": \"${TRAIN_WRITE}\",
    \"FRONTIERCS_ACT_MAX_NEW_TOKENS\": \"${ACT_MAX_NEW_TOKENS}\",
    \"FRONTIERCS_WRITE_MAX_NEW_TOKENS\": \"${WRITE_MAX_NEW_TOKENS}\",
    \"FRONTIERCS_WRITER_MAX_PROMPT_CHARS\": \"${WRITER_MAX_PROMPT_CHARS}\",
    \"FRONTIERCS_DIAGNOSTICS_CHARS\": \"${DIAGNOSTICS_CHARS}\",
    \"FRONTIERCS_JUDGE_TIMEOUT_SECONDS\": \"${JUDGE_TIMEOUT_SECONDS}\",
    \"FRONTIERCS_JUDGE_POLL_SECONDS\": \"${JUDGE_POLL_SECONDS}\",
    \"FRONTIERCS_ACT_ADVANTAGE_MODE\": \"${FRONTIERCS_ACT_ADVANTAGE_MODE:-temporal_problem_relative}\",
    \"FRONTIERCS_WRITE_REWARD_MODE\": \"${FRONTIERCS_WRITE_REWARD_MODE:-delta}\",
    \"FRONTIERCS_WRITE_ADVANTAGE_MODE\": \"${FRONTIERCS_WRITE_ADVANTAGE_MODE:-direct}\",
    \"FRONTIERCS_WRITE_ADVANTAGE_SCALE\": \"${FRONTIERCS_WRITE_ADVANTAGE_SCALE:-1.0}\",
    \"FRONTIERCS_ACT_CODE_CONTEXT\": \"${FRONTIERCS_ACT_CODE_CONTEXT:-none}\",
    \"FRONTIERCS_ENABLE_THINKING\": \"${FRONTIERCS_ENABLE_THINKING:-1}\",
    \"FRONTIERCS_ACT_EXPLORE_BETA\": \"${ACT_EXPLORE_BETA}\",
    \"FRONTIERCS_EXPLORE_JUDGE_API_KEY\": \"${EXPLORE_JUDGE_API_KEY}\",
    \"FRONTIERCS_EXPLORE_JUDGE_API_BASE\": \"${EXPLORE_JUDGE_API_BASE}\",
    \"FRONTIERCS_EXPLORE_JUDGE_MODEL\": \"${EXPLORE_JUDGE_MODEL}\",
    \"FRONTIERCS_EXPLORE_JUDGE_TIMEOUT\": \"${EXPLORE_JUDGE_TIMEOUT}\",
    \"FRONTIERCS_EXPLORE_JUDGE_CONCURRENCY\": \"${EXPLORE_JUDGE_CONCURRENCY}\",
    \"FRONTIERCS_TASK_BASELINE_ARTIFACT\": \"${FRONTIERCS_TASK_BASELINE_ARTIFACT:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MILES_DISABLE_MTP\": \"1\",
    \"MILES_DISABLE_FINAL_SAVE\": \"${MILES_DISABLE_FINAL_SAVE:-}\",
    \"PYTHONDONTWRITEBYTECODE\": \"1\"
  }
}"

ray job submit --address="${RAY_ADDRESS}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --ref-load "${TORCH_DIST}" \
  --load "${SAVE_DIR}" \
  --save "${SAVE_DIR}" \
  --save-interval "${FRONTIERCS_SAVE_INTERVAL:-1}" \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --metadata-key metadata \
  --num-rollout "${NUM_UPDATES}" \
  --rollout-batch-size "${GROUPS_PER_UPDATE}" \
  --rollout-shuffle \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len "${ACT_MAX_NEW_TOKENS}" \
  --rollout-temperature "${FRONTIERCS_TEMPERATURE:-1.0}" \
  --global-batch-size "${FRONTIERCS_NOMINAL_GLOBAL_BATCH_SIZE:-${GROUPS_PER_UPDATE}}" \
  --use-dynamic-global-batch-size \
  --balance-data \
  --custom-generate-function-path examples.frontiercs_ttt.frontiercs_episode_rollout.generate_episode \
  --custom-config-path "${SCRIPT_DIR}/frontiercs_episode_config.yaml" \
  --custom-reward-post-process-path examples.frontiercs_ttt.frontiercs_advantage.reward_post_process \
  --custom-rollout-log-function-path examples.frontiercs_ttt.frontiercs_metrics.log_rollout_data \
  --advantage-estimator grpo \
  --use-rollout-logprobs \
  "${KL_ARGS[@]}" \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr "${FRONTIERCS_LR:-1e-6}" \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --tensor-model-parallel-size "${TP}" \
  --pipeline-model-parallel-size "${PP}" \
  --context-parallel-size "${CP}" \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --sequence-parallel \
  --accumulate-allreduce-grads-in-fp32 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --seq-length "${SEQ_LENGTH}" \
  --max-position-embeddings "${SEQ_LENGTH}" \
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
  --log-probs-chunk-size "${FRONTIERCS_LOGP_CHUNK:-512}" \
  --train-backend megatron \
  --distributed-timeout-minutes "${FRONTIERCS_DIST_TIMEOUT_MIN:-120}" \
  --attention-backend flash \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}" \
  --num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
  --sglang-mem-fraction-static "${FRONTIERCS_SGLANG_MEM_FRACTION:-0.5}" \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
  --pin-rollout-manager-to-head \
  "${WANDB_ARGS[@]}" \
  --colocate
