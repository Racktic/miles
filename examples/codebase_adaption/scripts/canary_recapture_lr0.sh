#!/usr/bin/env bash
# lr=0 金丝雀 x sglang recapture 补丁(PR #27140 移植版)验证(2026-07-14)。
# 目的: 权重零更新跑 2 个 rollout。此前该配置下 rollout_1 必崩(ACT=0.0000);
# 打开 CODEBASE_SGLANG_RECAPTURE_PATCH 后若 rollout_1 恢复健康(ACT≈0.21-0.23),
# 则补丁生效且根因(sleep/wake 后 CUDA graph 回放陈旧 mamba 状态指针)定案。
# 判据:
#   1. server 日志出现 "Recapturing TP worker CUDA graphs after weight update"
#   2. rollout_1 的 codebase_samples/act_mean_raw_reward 与 completed 数与 rollout_0 同量级
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_SGLANG_RECAPTURE_PATCH=1        # 本实验的唯一自变量

export CODEBASE_RUN_ID="swecl-4b-recapture-lr0"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swecl-4b-textfmt"

export CODEBASE_NUM_ROLLOUT=2                   # rollout_0(健康基准) + rollout_1(判决)
export CODEBASE_EVAL_INTERVAL=0
export CODEBASE_SAVE_INTERVAL=0
export CODEBASE_LR=0.0                          # 权重零更新
export CODEBASE_SEQ_LEN=24576                   # 32768 已两次在 step1 OOM, 用户裁定回 24576
export CODEBASE_MAX_TOK_PER_GPU=24576
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_HF_CKPT="/data/user_data/qixinx/Qwen3.5-4B"
export CODEBASE_TORCH_DIST="/data/user_data/qixinx/Qwen3.5-4B_torch_dist"
export CODEBASE_PROMPT_DATA="${SD}/data/formal_episodes.jsonl"
export CODEBASE_BASELINE_ARTIFACT="${SD}/data/baseline_4b_textfmt.json"
export CODEBASE_NGPU=8 CODEBASE_TP=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR=/scratch/qixinx
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

cd "$SD"
echo "=== recapture-patch lr=0 canary RUN=${CODEBASE_RUN_ID} on $(hostname) (worktree: $SD) ==="
bash ./launch_codebase_adaption_apptainer.sh
