#!/bin/bash
# Run inside the Miles Apptainer image on an exclusive GPU node.
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
PROMPT_DATA="${FRONTIERCS_PROMPT_DATA:-${SCRIPT_DIR}/data/problem_groups_smoke.jsonl}"
MEMORY_ROUNDS="${FRONTIERCS_MEMORY_ROUNDS:-2}"
CANDIDATES_PER_PROBLEM="${FRONTIERCS_CANDIDATES_PER_PROBLEM:-1}"
RUN_ID="${FRONTIERCS_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${FRONTIERCS_OUTPUT_ROOT:-${FRONTIERCS_ROOT}/qwen_eval/results/frontiercs_ttt_rl}"
JUDGE_PORT="${FRONTIERCS_JUDGE_PORT:-8081}"
GOJUDGE_PORT="${FRONTIERCS_GOJUDGE_PORT:-5050}"
JUDGE_URL="${FRONTIERCS_JUDGE_URL:-http://127.0.0.1:${JUDGE_PORT}}"

NGPU="${FRONTIERCS_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${NGPU:-0}" -lt 1 ]; then
  echo "No visible GPU; set FRONTIERCS_NGPU only after entering a GPU allocation." >&2
  exit 2
fi
TP="${FRONTIERCS_TP:-${NGPU}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
export PYTHONUNBUFFERED=1

if [ ! -f "${PROMPT_DATA}" ]; then
  echo "Missing group dataset: ${PROMPT_DATA}" >&2
  exit 2
fi
GROUP_COUNT="${FRONTIERCS_ROLLOUT_BATCH_SIZE:-$(awk 'NF { count += 1 } END { print count + 0 }' "${PROMPT_DATA}")}"
if [ "${GROUP_COUNT}" -lt 1 ]; then
  echo "The Frontier-CS group dataset is empty: ${PROMPT_DATA}" >&2
  exit 2
fi
AUTO_JUDGE_STARTED=0
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
  trap 'if [ "${AUTO_JUDGE_STARTED}" = "1" ]; then python3 "${SCRIPT_DIR}/host_judge.py" stop --api-port "${JUDGE_PORT}" --gojudge-port "${GOJUDGE_PORT}" --cleanup; fi' EXIT
fi
if ! curl -fsS "${JUDGE_URL}/health" >/dev/null; then
  echo "Frontier-CS judge is not healthy at ${JUDGE_URL}/health" >&2
  echo "Start it first, or set FRONTIERCS_AUTO_START_JUDGE=1 for the portable host judge." >&2
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
LOG_DIR="${SCRIPT_DIR}/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/${RUN_ID}" "${SAVE_DIR}"

echo "Frontier-CS TTT run: ${RUN_ID}"
echo "groups=${GROUP_COUNT} K=${CANDIDATES_PER_PROBLEM} S=${MEMORY_ROUNDS}"
echo "traces: ${OUTPUT_ROOT}/${RUN_ID}"
echo "checkpoints: ${SAVE_DIR}"

# Avoid killing an unrelated Ray session by default.  On a fresh exclusive
# allocation, set FRONTIERCS_RESET_RAY=1 if a stale session must be cleared.
if [ "${FRONTIERCS_RESET_RAY:-0}" = "1" ]; then
  ray stop --force >/dev/null 2>&1 || true
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_DASH_PORT="${RAY_DASH_PORT:-8265}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4000000000}"
ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NGPU}" \
  --disable-usage-stats \
  --port "${RAY_GCS_PORT:-6379}" \
  --object-store-memory "${RAY_OBJECT_STORE_MEMORY}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASH_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_DIR}:${FRONTIERCS_ROOT}\",
    \"FRONTIERCS_ROOT\": \"${FRONTIERCS_ROOT}\",
    \"FRONTIERCS_RUN_ID\": \"${RUN_ID}\",
    \"FRONTIERCS_OUTPUT_ROOT\": \"${OUTPUT_ROOT}\",
    \"FRONTIERCS_JUDGE_URL\": \"${JUDGE_URL}\",
    \"FRONTIERCS_MEMORY_ROUNDS\": \"${MEMORY_ROUNDS}\",
    \"FRONTIERCS_CANDIDATES_PER_PROBLEM\": \"${CANDIDATES_PER_PROBLEM}\",
    \"FRONTIERCS_ACT_ADVANTAGE_MODE\": \"${FRONTIERCS_ACT_ADVANTAGE_MODE:-raw}\",
    \"FRONTIERCS_WRITE_REWARD_MODE\": \"${FRONTIERCS_WRITE_REWARD_MODE:-delta}\",
    \"FRONTIERCS_WRITE_ADVANTAGE_MODE\": \"${FRONTIERCS_WRITE_ADVANTAGE_MODE:-direct}\",
    \"FRONTIERCS_ACT_CODE_CONTEXT\": \"${FRONTIERCS_ACT_CODE_CONTEXT:-none}\",
    \"FRONTIERCS_ENABLE_THINKING\": \"${FRONTIERCS_ENABLE_THINKING:-1}\",
    \"FRONTIERCS_ACT_EXPLORE_BETA\": \"${FRONTIERCS_ACT_EXPLORE_BETA:-0}\",
    \"FRONTIERCS_TASK_BASELINE_ARTIFACT\": \"${FRONTIERCS_TASK_BASELINE_ARTIFACT:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"MILES_DISABLE_MTP\": \"1\",
    \"PYTHONDONTWRITEBYTECODE\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASH_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --ref-load "${TORCH_DIST}" \
  --load "${SAVE_DIR}" \
  --save "${SAVE_DIR}" \
  --save-interval 1 \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --metadata-key metadata \
  --num-rollout "${MEMORY_ROUNDS}" \
  --rollout-batch-size "${GROUP_COUNT}" \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len "${FRONTIERCS_ACT_MAX_NEW_TOKENS:-25600}" \
  --rollout-temperature "${FRONTIERCS_TEMPERATURE:-1.0}" \
  --global-batch-size "${GROUP_COUNT}" \
  --use-dynamic-global-batch-size \
  --balance-data \
  --custom-generate-function-path examples.frontiercs_ttt.frontiercs_rollout.generate \
  --custom-config-path "${SCRIPT_DIR}/frontiercs_config.yaml" \
  --custom-reward-post-process-path examples.frontiercs_ttt.frontiercs_advantage.reward_post_process \
  --advantage-estimator grpo \
  --use-rollout-logprobs \
  --use-kl-loss \
  --kl-loss-coef "${FRONTIERCS_KL:-0.01}" \
  --kl-loss-type k1 \
  --kl-coef 0.0 \
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
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --sequence-parallel \
  --accumulate-allreduce-grads-in-fp32 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --seq-length "${FRONTIERCS_SEQ_LENGTH:-32768}" \
  --max-position-embeddings "${FRONTIERCS_SEQ_LENGTH:-32768}" \
  --max-tokens-per-gpu "${FRONTIERCS_MAX_TOKENS_PER_GPU:-32768}" \
  --log-probs-chunk-size "${FRONTIERCS_LOGP_CHUNK:-512}" \
  --train-backend megatron \
  --distributed-timeout-minutes "${FRONTIERCS_DIST_TIMEOUT_MIN:-120}" \
  --attention-backend flash \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-softmax-in-fp32 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static "${FRONTIERCS_SGLANG_MEM_FRACTION:-0.5}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NGPU}" \
  --rollout-num-gpus "${NGPU}" \
  --colocate
