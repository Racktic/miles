#!/usr/bin/env bash
# lr=0 金丝雀 #5: A_log 探针定位实验(2026-07-14)。
# 判决书: 金丝雀#4 张量指纹显示 sync-1→sync-2 唯一变动 = layers.0.linear_attn.A_log
# (-87.121→+59.931, 8 引擎一致, lr=0)。本实验在 miles actor.py 的 train_actor 各节点
# 加 [alog-dbg] 探针(GPU 参数 + CPU backup 字典), 日志直接点名改写 A_log 的操作。
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_SGLANG_RECAPTURE_PATCH=1   # 保留(无graph时自动空转), 保证与金丝雀#1单变量差异


export CODEBASE_RUN_ID="swecl-4b-alogdbg-lr0"
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
