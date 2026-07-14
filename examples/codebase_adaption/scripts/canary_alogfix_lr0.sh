#!/usr/bin/env bash
# lr=0 金丝雀 #6: A_log bf16 修复复验(2026-07-14)。
# 判决书(#5 探针): before train() -87.121 健康 -> after train() +59.931 -> backup 中毒。
# 凶手 = optimizer.step 对孤儿 fp32 参数(A_log, 全模型唯一)的错误回写(lr=0 也发生)。
# 修复 = plugin 中 A_log 改随 config.dtype(bf16)。判据: after train() 不变 + rollout_1 健康
# + actor==ref logprob 16 位相等 + 轨迹格式正常。探针保留以验证。
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_SGLANG_RECAPTURE_PATCH=1   # 保留(无graph时自动空转), 保证与金丝雀#1单变量差异


export CODEBASE_RUN_ID="swecl-4b-alogfix-lr0"
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
