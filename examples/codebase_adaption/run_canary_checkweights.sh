#!/bin/bash
set -ex

NGPU="${CODEBASE_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; [ "${NGPU:-0}" -ge 1 ] || NGPU=4
TP="${CODEBASE_TP:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

ray stop --force 2>/dev/null || true; sleep 2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${MILES_DIR}"

export CLBENCH_ROOT="${CLBENCH_ROOT:-/home/qixinx/continual-learning-bench}"
# Merged baseline (231 swe_bench_cl @ avg@4 + 19 codebase) so gain=reward-baseline
# is defined for BOTH the swecl train pool and the codebase heldout eval. The old
# default pointed at the codebase-only trace, which left every swecl train id at
# baseline 0 (gain==reward) — see scripts/swecl/gen_train_episodes.py. env wins over
# the YAML codebase_baseline_artifact (codebase_rollout.py:74), so set it correctly here.
export CODEBASE_BASELINE_ARTIFACT="${CODEBASE_BASELINE_ARTIFACT:-${SCRIPT_DIR}/data/baseline_merged.json}"
export CLBENCH_CONTAINER_BACKEND="${CLBENCH_CONTAINER_BACKEND:-singularity}"
export MSWEA_SINGULARITY_EXECUTABLE="${MSWEA_SINGULARITY_EXECUTABLE:-apptainer}"
export CLBENCH_SIF_DIR="${CLBENCH_SIF_DIR:-/data/user_data/qixinx/clbench/sifs}"
export CODEBASE_PYDEPS="${CODEBASE_PYDEPS:-/data/user_data/qixinx/miles_pydeps/codebase_py312_clean}"
if [ -d "${CODEBASE_PYDEPS}" ]; then
  export PYTHONPATH="${CODEBASE_PYDEPS}:${PYTHONPATH:-}"
fi
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
export SSL_CERT_DIR="${SSL_CERT_DIR:-/etc/ssl/certs}"

# Prevent nested Apptainer calls from inheriting the outer training container's
# bind list. The clbench backend creates writable sandboxes whose filesystems do
# not necessarily contain host mountpoints like /data.
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH

source "${MILES_DIR}/scripts/models/${CODEBASE_MODEL_SCRIPT:-qwen3.5-9B.sh}"

CKPT="${CODEBASE_HF_CKPT:-/data/user_data/qixinx/Qwen3.5-9B}"
TORCH_DIST="${CODEBASE_TORCH_DIST:-/data/user_data/qixinx/Qwen3.5-9B_torch_dist}"
RUN_ID="${CODEBASE_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SAVE_DIR="${CODEBASE_SAVE_DIR:-/data/group_data/rl/yuxiaoq/qixinx/codebase_adaption_runs/${RUN_ID}/ckpt}"
CODEBASE_RUN_DIR="${SCRIPT_DIR}/logs/${RUN_ID}"
CODEBASE_TRAJ_DIR="${CODEBASE_TRAJ_DIR:-${CODEBASE_RUN_DIR}/traj}"
mkdir -p "${SAVE_DIR}" "${CODEBASE_TRAJ_DIR}" "${CODEBASE_RUN_DIR}"
echo "=== Codebase run id=${RUN_ID}  traj -> ${CODEBASE_TRAJ_DIR} ==="

set +x
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb.env" ]; then
  set -a
  . "${HOME}/.wandb.env"
  set +a
fi
WANDB_ARGS=()
if [ -n "${CODEBASE_USE_WANDB:-}" ] && [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-miles-codebase-adaption}"
    --wandb-group "${WANDB_GROUP:-codebase-qwen3.5-9B-${RUN_ID}}"
    --disable-wandb-random-suffix
  )
  if [ -n "${CODEBASE_WANDB_RUN_ID:-}" ]; then
    WANDB_ARGS+=(--wandb-run-id "${CODEBASE_WANDB_RUN_ID}")
  fi
  if [ -n "${WANDB_TEAM:-}" ]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
  if [ -n "${WANDB_MODE:-}" ]; then
    WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
  fi
else
  WANDB_API_KEY=""
fi
set -x

CKPT_ARGS=(
  --hf-checkpoint ${CKPT}
  --load ${SAVE_DIR}
  --save ${SAVE_DIR}
)
case "${CODEBASE_SAVE_INTERVAL:-0}" in
  ""|0|none|None|false|False|off|Off)
    export MILES_DISABLE_FINAL_SAVE=1
    CKPT_ARGS+=(--save-interval 1000000000)
    ;;
  *)
    CKPT_ARGS+=(--save-interval "${CODEBASE_SAVE_INTERVAL}")
    ;;
esac

ROLLOUT_ARGS=(
  --prompt-data ${CODEBASE_PROMPT_DATA:-${SCRIPT_DIR}/data/swecl_train_episodes.jsonl}
  --input-key prompt
  --metadata-key metadata
  --rollout-shuffle
  --num-rollout ${CODEBASE_NUM_ROLLOUT:-1}
  --rollout-batch-size ${CODEBASE_ROLLOUT_BATCH_SIZE:-2}
  --n-samples-per-prompt ${CODEBASE_N_SAMPLES:-8}
  --rollout-max-response-len ${CODEBASE_MAX_RESP:-1200}
  --rollout-temperature ${CODEBASE_TEMP:-1}
  --global-batch-size ${CODEBASE_GLOBAL_BATCH_SIZE:-16}
  --balance-data
  --log-multi-turn
)
case "${CODEBASE_DYNAMIC_GLOBAL_BATCH:-1}" in
  ""|0|none|None|false|False|off|Off)
    ;;
  *)
    ROLLOUT_ARGS+=(--use-dynamic-global-batch-size)
    ;;
esac

CUSTOM_ARGS=(
  --custom-generate-function-path examples.codebase_adaption.codebase_rollout.generate
  --custom-config-path examples/codebase_adaption/codebase_config.yaml
  --custom-reward-post-process-path examples.codebase_adaption.codebase_advantage.reward_post_process
  --custom-rollout-log-function-path examples.codebase_adaption.codebase_metrics.log_rollout_data
  --custom-eval-rollout-log-function-path examples.codebase_adaption.codebase_metrics.log_eval_rollout_data
)

EVAL_ARGS=()
case "${CODEBASE_EVAL_INTERVAL:-3}" in
  ""|0|none|None|false|False|off|Off)
    ;;
  *)
    EVAL_ARGS=(
      --eval-interval ${CODEBASE_EVAL_INTERVAL:-3}
      --eval-prompt-data heldout ${CODEBASE_EVAL_PROMPT_DATA:-${SCRIPT_DIR}/data/heldout_episodes.jsonl}
      --n-samples-per-eval-prompt ${CODEBASE_N_EVAL_SAMPLES:-1}
      --skip-eval-before-train
    )
    ;;
esac

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef ${CODEBASE_KL:-0.01}
  --kl-loss-type k1
  --ref-load ${TORCH_DIST}
  --kl-coef 0.00
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${CODEBASE_LR:-1e-6}
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

PERF_ARGS=(
  --tensor-model-parallel-size ${TP}
  --pipeline-model-parallel-size 1
  --context-parallel-size ${CODEBASE_CP:-1}
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --accumulate-allreduce-grads-in-fp32
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --seq-length ${CODEBASE_SEQ_LEN:-32768}
  --max-position-embeddings ${CODEBASE_SEQ_LEN:-32768}
  --max-tokens-per-gpu ${CODEBASE_MAX_TOK_PER_GPU:-20480}
  --log-probs-chunk-size ${CODEBASE_LOGP_CHUNK:-512}
)

MEGATRON_ARGS=(
  --train-backend megatron
  --distributed-timeout-minutes ${CODEBASE_DIST_TIMEOUT_MIN:-180}
  --attention-backend flash
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static ${CODEBASE_SGLANG_MEM:-0.5}
)

MISC_ARGS=(
  --check-weight-update-equal
  --actor-num-nodes 1
  --actor-num-gpus-per-node ${NGPU}
  --rollout-num-gpus ${NGPU}
  --colocate
)

# --object-store-memory: cap Ray's plasma store. Without it Ray reads the host's
# physical RAM (not the cgroup limit) and reserves ~30% (observed 200GB) in
# /dev/shm, which counts against our cgroup and OOM-kills the job. alchemy sets
# 4GB; we use a bit more for the larger multi-turn rollout samples.
ray start --head --num-gpus="${NGPU}" --disable-usage-stats --object-store-memory ${CODEBASE_OBJECT_STORE_MEM:-16000000000}
set +x
if [ -n "${WANDB_API_KEY:-}" ]; then
  set +x
fi
RAY_JOB_ARGS=()
if [ -n "${CODEBASE_RAY_NO_WAIT:-}" ]; then
  RAY_JOB_ARGS+=(--no-wait)
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_DIR}:${PYTHONPATH:-}\",
    \"CLBENCH_ROOT\": \"${CLBENCH_ROOT}\",
    \"CODEBASE_BASELINE_ARTIFACT\": \"${CODEBASE_BASELINE_ARTIFACT}\",
    \"CLBENCH_CONTAINER_BACKEND\": \"${CLBENCH_CONTAINER_BACKEND}\",
    \"MSWEA_SINGULARITY_EXECUTABLE\": \"${MSWEA_SINGULARITY_EXECUTABLE}\",
    \"CLBENCH_SIF_DIR\": \"${CLBENCH_SIF_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"${CUDA_DEVICE_MAX_CONNECTIONS}\",
    \"SSL_CERT_FILE\": \"${SSL_CERT_FILE}\",
    \"SSL_CERT_DIR\": \"${SSL_CERT_DIR}\",
    \"PYTHONDONTWRITEBYTECODE\": \"1\",
    \"MILES_DISABLE_FINAL_SAVE\": \"${MILES_DISABLE_FINAL_SAVE:-}\",
    \"CODEBASE_TRAJ_DIR\": \"${CODEBASE_TRAJ_DIR}\",
    \"CODEBASE_NUM_ACTS_CAP\": \"${CODEBASE_NUM_ACTS_CAP:-0}\",
    \"CODEBASE_MAX_STEPS_PER_ISSUE\": \"${CODEBASE_MAX_STEPS_PER_ISSUE:-}\",
    \"CODEBASE_NO_MEMORY\": \"${CODEBASE_NO_MEMORY:-}\",
    \"CODEBASE_CONTEXT_MAX_TOKENS\": \"${CODEBASE_CONTEXT_MAX_TOKENS:-240000}\",
    \"CODEBASE_CONTEXT_RESERVE_TOKENS\": \"${CODEBASE_CONTEXT_RESERVE_TOKENS:-500}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"
  }
}"
ray job submit --address="http://127.0.0.1:8265" \
  "${RAY_JOB_ARGS[@]}" \
  --runtime-env-json "${RUNTIME_ENV_JSON}" \
  -- python3 "${MILES_DIR}/train.py" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${CUSTOM_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${MEGATRON_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]} \
  ${WANDB_ARGS[@]}
