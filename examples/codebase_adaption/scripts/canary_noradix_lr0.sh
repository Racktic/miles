#!/usr/bin/env bash
# lr=0 金丝雀(备用): --sglang-disable-radix-cache(2026-07-14)。
# 目的: 切断 mamba 状态快照跨请求复用(#24954 污染传播路径)。
# SIF 已核实缺 #24954 与 #26941 两个 mamba 修复且 ping-pong 机制激活(memory_pool.py:513/592)。
# 与其他金丝雀的唯一差异 = CODEBASE_SGLANG_EXTRA_ARGS。
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_SGLANG_RECAPTURE_PATCH=1   # 保留(无graph时自动空转), 保证与金丝雀#1单变量差异
export CODEBASE_SGLANG_EXTRA_ARGS="--sglang-disable-radix-cache"   # 本实验唯一自变量

export CODEBASE_RUN_ID="swecl-4b-noradix-lr0"
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
