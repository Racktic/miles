#!/usr/bin/env bash
# SWE-smith pass@k(avg@2)采集 —— eval-only 模式(2026-07-20, 用户要求)。
# 机制: CODEBASE_EVAL_BEFORE_TRAIN=1 让 eval 先于一切训练步执行(RUNBOOK W7 模式),
# 1262 单题 episode 作为 eval 数据一次投放, 全程无引擎切换/无 log_probs/无训练步。
# eval 落盘后仅剩 1 个 2-episode 的迷你训练 rollout(LR=0, 几分钟)然后自然退出。
# ⚠ 千级 episode 一次投放必须 SGLANG_CONCURRENCY=4(RUNBOOK W6, 防打爆沙箱构建)。
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_TRAIN_TASK=swe_smith          # 迷你训练 rollout 也走 swe_smith(避免 240 池混入)
export CODEBASE_TRAIN_DATASET=data/swe_smith/pilot.jsonl
export CODEBASE_EVAL_TASK=swe_smith
export CODEBASE_EVAL_DATASET=data/swe_smith/pilot.jsonl
export CODEBASE_TRAIN_ACT_ONLY=1
export CODEBASE_LR=0.0
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_RUN_ID="${CODEBASE_RUN_ID:-swesmith-passk-evalonly}"
export CODEBASE_NUM_ROLLOUT=1
# 废步最小化(照抄 eval_ckpt_19q 2026-07-16 定案): train_iters = 1*1*4//4 = 1 >= 1,
# 且 gbs=4 被 DP=4 整除; 我此前的 2*2//16=0 会崩 lr 调度器。
export CODEBASE_ROLLOUT_BATCH_SIZE=1
export CODEBASE_N_SAMPLES=4
export CODEBASE_GLOBAL_BATCH_SIZE=4
export CODEBASE_EVAL_INTERVAL=1000000         # 禁掉废步之后的周期 eval(防双跑)
export CODEBASE_EVAL_BEFORE_TRAIN=1
export CODEBASE_MAX_RESP=32768                # eval-only 口径: 无每轮生成上限(训练口径 2500; 用户指正)
export CODEBASE_USE_WANDB=
export CODEBASE_EVAL_PROMPT_DATA="${CODEBASE_EVAL_PROMPT_DATA:-${SD}/data/swesmith_passk_pilot.jsonl}"
export CODEBASE_N_EVAL_SAMPLES=2
export CODEBASE_SGLANG_CONCURRENCY=4
export CODEBASE_SAVE_INTERVAL=0
export CODEBASE_SEQ_LEN=24576
export CODEBASE_MAX_TOK_PER_GPU=20480
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_PROMPT_DATA="${CODEBASE_PROMPT_DATA:-${SD}/data/swesmith_passk_pilot.jsonl}"
export CODEBASE_BASELINE_ARTIFACT="${SD}/data/baseline_4b_textfmt.json"
NGPU_DETECTED=$(nvidia-smi -L 2>/dev/null | wc -l); export CODEBASE_NGPU=${CODEBASE_NGPU:-${NGPU_DETECTED:-8}}
export CODEBASE_TP=2

NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_SAVE_DIR="${NODE_RUN_DIR}/ckpt"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"
source "${SD}/scripts/cluster_orchard_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((CODEBASE_NGPU-1)))}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
SCR="${CODEBASE_SCR:-/tmp/qixinx}"
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

bash "${SD}/scripts/ensure_torch_dist_local.sh"

cd "$SD"
echo "=== SWE-smith pass@k EVAL-ONLY RUN=${CODEBASE_RUN_ID} NGPU=${CODEBASE_NGPU} on $(hostname) ==="
bash ./launch_codebase_adaption_apptainer.sh
