#!/bin/bash
set -ex

# 提交侧 shell 可能携带宿主(RHEL9)证书路径变量, sbatch --export=ALL 会把它们带进
# Ubuntu 容器, 而容器内该路径不存在 → python/wandb 的全部 https 报
# SSLError(FileNotFoundError)/x509 unknown authority(2026-07-18 事故, 见 RUNBOOK W8)。
# 容器内一律使用镜像自带 CA。
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO PIP_CERT

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
# --save-hf 默认开启(2026-07-16 用户要求): 每次存 ckpt 时经 Megatron AutoBridge
# 额外导出完整 HF 模型(含未训练的 vision 塔, 官方机制, 见 notes/RUNBOOK.md W7),
# 评测/分发即取即用。缺省路径 = ckpt 的兄弟目录 hf/iter_{rollout_id};
# CODEBASE_SAVE_HF 可覆盖路径, 设为 off/0/none 可关闭。
SAVE_HF_DEFAULT="${SAVE_DIR%/ckpt}/hf/iter_{rollout_id}"
CODEBASE_SAVE_HF="${CODEBASE_SAVE_HF:-${SAVE_HF_DEFAULT}}"
case "${CODEBASE_SAVE_HF}" in
  ""|0|none|None|false|False|off|Off) ;;
  *) CKPT_ARGS+=(--save-hf "${CODEBASE_SAVE_HF}") ;;
esac
case "${CODEBASE_SAVE_INTERVAL:-0}" in
  ""|0|none|None|false|False|off|Off)
    export MILES_DISABLE_FINAL_SAVE=1
    CKPT_ARGS+=(--save-interval 1000000000)
    ;;
  *)
    CKPT_ARGS+=(--save-interval "${CODEBASE_SAVE_INTERVAL}")
    ;;
esac

# CODEBASE_ROLLOUT_SHUFFLE=0 disables dataset shuffling so episodes are consumed
# in file order (sequential offset per epoch) — required for step-level
# curriculum data files (easy episodes first). Default 1 keeps legacy behavior.
if [[ "${CODEBASE_ROLLOUT_SHUFFLE:-1}" == "1" ]]; then
  ROLLOUT_SHUFFLE_ARG="--rollout-shuffle"
else
  ROLLOUT_SHUFFLE_ARG=""
fi

ROLLOUT_ARGS=(
  --prompt-data ${CODEBASE_PROMPT_DATA:-${SCRIPT_DIR}/data/swecl_train_episodes.jsonl}
  --input-key prompt
  --metadata-key metadata
  ${ROLLOUT_SHUFFLE_ARG}
  --num-rollout ${CODEBASE_NUM_ROLLOUT:-1}
  --rollout-batch-size ${CODEBASE_ROLLOUT_BATCH_SIZE:-2}
  --n-samples-per-prompt ${CODEBASE_N_SAMPLES:-8}
  --rollout-max-response-len ${CODEBASE_MAX_RESP:-2500}
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
# DAPO-style zero-std group dropping (2026-07-21): wraps generate_rollout to
# physically remove constant-advantage sibling groups (~40-50% dead groups that
# dilute the effective step size) before the dynamic-gbs computation.
if [[ "${CODEBASE_DROP_ZERO_STD_GROUPS:-0}" == "1" ]]; then
  CUSTOM_ARGS+=(--rollout-function-path examples.codebase_adaption.codebase_rollout_filter.generate_rollout)
fi

EVAL_ARGS=()
case "${CODEBASE_EVAL_INTERVAL:-3}" in
  ""|0|none|None|false|False|off|Off)
    ;;
  *)
    EVAL_ARGS=(
      --eval-interval ${CODEBASE_EVAL_INTERVAL:-3}
      --eval-prompt-data heldout ${CODEBASE_EVAL_PROMPT_DATA:-${SCRIPT_DIR}/data/heldout_episodes.jsonl}
      --n-samples-per-eval-prompt ${CODEBASE_N_EVAL_SAMPLES:-1}
    )
    # CODEBASE_EVAL_BEFORE_TRAIN=1: drop --skip-eval-before-train so eval fires at
    # rollout_id==0 BEFORE any training step. Used for ckpt evaluation: load a frozen
    # ckpt weights-only (+ --start-rollout-id 0 via CODEBASE_TRAIN_EXTRA_ARGS), eval
    # runs on the exact synced weights; the one wasted rollout after it persists
    # nothing (save disabled). Default unchanged: skip flag kept.
    if [[ "${CODEBASE_EVAL_BEFORE_TRAIN:-0}" != "1" ]]; then
      EVAL_ARGS+=(--skip-eval-before-train)
    fi
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
)
# CODEBASE_NO_OFFLOAD=1 drops the HybridDeviceOptimizer (cpu-offload) stack and
# matches the stock scripts/run-qwen3.5-4B.sh optimizer config. Both A_log-style
# fp32-bucket corruption and the dp_reshardable common.pt step poisoning
# (notes/CKPT_RESUME_BUGS_0715.md) live exclusively in the offload path; without
# it checkpoints support native full resume. 4B@TP2/DP4 adds ~6GB/GPU fp32
# optimizer shards — first train step after switching is the memory validation,
# fall back to offload on OOM. First relaunch after switching MUST stay
# weights-only (--no-load-optim): older ckpts hold HDO-format optimizer state.
if [[ "${CODEBASE_NO_OFFLOAD:-0}" != "1" ]]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

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
  # episode 并发闸 = 本值 x 卡数(miles semaphore)。默认 512 与 miles 默认一致(训练行为零变化);
  # eval 全量(1036 episode 一次性投放)必须限流, wrapper 设 4 -> 32 并发(2026-07-13 用户批准)。
  --sglang-server-concurrency ${CODEBASE_SGLANG_CONCURRENCY:-512}
)
# 崩塌排查金丝雀的显式引擎旗标通道(如 --sglang-disable-cuda-graph);默认空 = 行为零变化。
if [ -n "${CODEBASE_SGLANG_EXTRA_ARGS:-}" ]; then
  SGLANG_ARGS+=(${CODEBASE_SGLANG_EXTRA_ARGS})
fi

MISC_ARGS=(
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
# 监督式提交(2026-07-18, 用户批准): ray job submit 的前台日志尾随客户端依赖
# dashboard 的 WebSocket, dashboard 打嗝(HTTP 500 / close 1006)会让客户端崩溃,
# 在 sbatch 下等于整个训练被 SLURM 连坐杀掉(7/17 两起事故)。监督模式改为:
# --no-wait 提交 + 后台自动重连的日志尾随 + 前台轮询 ray job status 判生死
# (容忍 dashboard 瞬断)。本脚本保持前台运行以维持容器与 ray 头存活。
RAY_SUPERVISED=0
case "${CODEBASE_RAY_SUPERVISED:-0}" in
  1|true|True|yes|on) RAY_SUPERVISED=1 ;;
esac
if [ "${RAY_SUPERVISED}" = "1" ]; then
  RAY_SUBMISSION_ID="cb-${RUN_ID}-$(date +%s)"
  RAY_JOB_ARGS+=(--no-wait --submission-id "${RAY_SUBMISSION_ID}")
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
    \"CODEBASE_WRITE_REWARD_MODE\": \"${CODEBASE_WRITE_REWARD_MODE:-}\",
    \"CODEBASE_WRITE_FORMAT_BONUS\": \"${CODEBASE_WRITE_FORMAT_BONUS:-0.1}\",
    \"CODEBASE_TRAIN_ACT_ONLY\": \"${CODEBASE_TRAIN_ACT_ONLY:-}\",
    \"CODEBASE_WRITE_PROMPT_V3\": \"${CODEBASE_WRITE_PROMPT_V3:-}\",
    \"CODEBASE_WRITE_THINKING\": \"${CODEBASE_WRITE_THINKING:-}\",
    \"PYTHONDONTWRITEBYTECODE\": \"1\",
    \"MILES_DISABLE_FINAL_SAVE\": \"${MILES_DISABLE_FINAL_SAVE:-}\",
    \"CODEBASE_TRAJ_DIR\": \"${CODEBASE_TRAJ_DIR}\",
    \"CODEBASE_NUM_ACTS_CAP\": \"${CODEBASE_NUM_ACTS_CAP:-0}\",
    \"CODEBASE_MAX_STEPS_PER_ISSUE\": \"${CODEBASE_MAX_STEPS_PER_ISSUE:-}\",
    \"CODEBASE_NO_MEMORY\": \"${CODEBASE_NO_MEMORY:-}\",
    \"CODEBASE_CONTEXT_MAX_TOKENS\": \"${CODEBASE_CONTEXT_MAX_TOKENS:-240000}\",
    \"CODEBASE_CONTEXT_RESERVE_TOKENS\": \"${CODEBASE_CONTEXT_RESERVE_TOKENS:-500}\",
    \"TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC\": \"${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}\",
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
  ${WANDB_ARGS[@]} \
  ${CODEBASE_TRAIN_EXTRA_ARGS:-}

# ---- 监督循环(仅 CODEBASE_RAY_SUPERVISED=1;上面的 submit 因 --no-wait 秒回)----
if [ "${RAY_SUPERVISED}" = "1" ]; then
  set +e
  set +x
  echo "[supervise] submission_id=${RAY_SUBMISSION_ID}; 日志尾随自动重连 + status 轮询开始"
  TAIL_STOP="/tmp/.ray-tail-stop-${RAY_SUBMISSION_ID}"
  # 2026-07-18 orchard 冒烟实测的收尾 bug 修复: job 成功结束后 `ray job status` 会持续
  # 失败(原因未明, 疑似 dashboard 被重拉日志压垮), 旧逻辑 10min 后误报 UNREACHABLE
  # (exit 1), 且尾随子进程每 5s 重拉一遍全量 console log 造成日志洪泛。
  # 修法: 尾随流里检测 ray cli 的终态句("Job '<id>' succeeded/failed/stopped"),
  # 写入 TERM_FLAG 后不再重拉; status 轮询优先信任 TERM_FLAG。
  TERM_FLAG="/tmp/.ray-terminal-${RAY_SUBMISSION_ID}"
  rm -f "${TAIL_STOP}" "${TERM_FLAG}"
  # 重连续传: `ray job logs -f` 每次重连都会从头重放全量日志, 运行中频繁瞬断会让
  # console log 重复膨胀(3 天 run 可能达 GB 级, /home 配额扛不住)。用行数游标跳过
  # 已输出的前缀, 只透传新增行。
  TAIL_SEEN="/tmp/.ray-tail-seen-${RAY_SUBMISSION_ID}"
  echo 0 > "${TAIL_SEEN}"
  (
    while [ ! -f "${TAIL_STOP}" ]; do
      ray job logs --address="http://127.0.0.1:8265" -f "${RAY_SUBMISSION_ID}" 2>/dev/null \
        | while IFS= read -r line; do
            # n = 本次 dump 内的行序号(每次重连从 1 重数); 只透传超过历史游标的行
            n=$((${n:-0}+1))
            if [ ${n} -gt $(cat "${TAIL_SEEN}") ]; then
              printf '%s\n' "$line"
              echo ${n} > "${TAIL_SEEN}"
            fi
            case "$line" in
              *"Job '${RAY_SUBMISSION_ID}' succeeded"*) echo SUCCEEDED > "${TERM_FLAG}" ;;
              *"Job '${RAY_SUBMISSION_ID}' failed"*)    echo FAILED    > "${TERM_FLAG}" ;;
              *"Job '${RAY_SUBMISSION_ID}' stopped"*)   echo STOPPED   > "${TERM_FLAG}" ;;
            esac
          done
      [ -f "${TERM_FLAG}" ] && break
      [ -f "${TAIL_STOP}" ] && break
      echo "[supervise] 日志尾随断开, 5s 后重连(已见 $(cat "${TAIL_SEEN}") 行, 重放部分将跳过)"
      sleep 5
    done
  ) &
  TAIL_PID=$!
  FINAL=""
  unreachable=0
  # 状态查询走 dashboard HTTP API(JSON 的 status 字段是大写枚举)。不要 grep
  # `ray job status` 的人类输出——失败时它打小写 "Job '...' failed", 无 FAILED 枚举,
  # 会被永远解析为空 → 误走不可达分支 → 健康 run 也会 10 分钟被误杀(2026-07-18 事故)。
  _poll_status() {
    curl -s --max-time 10 "http://127.0.0.1:8265/api/jobs/" \
      | python3 -c "import json,sys;print(next((j.get('status','') for j in json.load(sys.stdin) if j.get('submission_id')=='${RAY_SUBMISSION_ID}'),''))" 2>/dev/null
  }
  while :; do
    # TERM_FLAG(尾随流里识别的终态)优先: 即刻定案, 也让尾随子进程不再重拉全量日志;
    # 常规路径走 dashboard HTTP API(上面的 _poll_status)。
    if [ -f "${TERM_FLAG}" ]; then
      FINAL=$(cat "${TERM_FLAG}"); break
    fi
    st=$(_poll_status)
    if [ -z "${st}" ]; then
      unreachable=$((unreachable+1))
      if [ ${unreachable} -ge 20 ]; then
        # dashboard 死 ≠ 训练死(2026-07-20 事故: dashboard 单独死亡, 旧逻辑误杀了
        # 仍在产轨迹的健康训练, 损失 r64-70)。旁路活性判据: traj 落盘新鲜度
        # (文件系统信号, 不依赖 dashboard)。45min 覆盖一个完整 rollout+train 周期。
        fresh=$(find "${CODEBASE_TRAJ_DIR}" -newermt "-45 minutes" -print -quit 2>/dev/null)
        if [ -n "${fresh}" ]; then
          echo "[supervise] status 不可达但 traj 仍在更新, 判定训练存活, 继续守望(dashboard 已放弃)"
          unreachable=0
          sleep 60; continue
        fi
        echo "[supervise] status 连续 ${unreachable} 次不可达且 traj 停更>45min, 判定训练已死"
        FINAL="UNREACHABLE"; break
      fi
      sleep 30; continue
    fi
    unreachable=0
    case "${st}" in
      SUCCEEDED|FAILED|STOPPED)
        sleep 15
        st2=$(_poll_status)
        if [ "${st2}" = "${st}" ]; then FINAL="${st}"; break; fi
        ;;
    esac
    sleep 60
  done
  touch "${TAIL_STOP}"
  kill "${TAIL_PID}" 2>/dev/null
  echo "[supervise] 训练终态: ${FINAL}"
  case "${FINAL}" in
    SUCCEEDED) exit 0 ;;
    *)         exit 1 ;;
  esac
fi
