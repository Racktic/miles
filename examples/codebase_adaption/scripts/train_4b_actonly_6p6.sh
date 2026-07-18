#!/usr/bin/env bash
# ACT-only x 6p6(4中段+2难题)数据 正式训练(2026-07-14 用户决策)。
# 与上一个 run(swecl-4b-alogfix-formal)的差异, 仅两点:
#   1. 数据: episodes_6p6_hard.jsonl —— 6+6 块, 每块 = 4 中段 anchor(易→难) + 2 难题
#      (能力沟优先/硬核补足), 262 个不重复 episode, 覆盖 218 题(mid 78 + gap 23 + hardcore 117), 9 迁移配对按存货加权;
#   2. CODEBASE_TRAIN_ACT_ONLY=1 —— WRITE 照常写/用记忆, 但不训练不给 reward(只训 ACT)。
# 背景: WRITE reward 的均值回归 bias 已定案(notes/WRITE_COLLAPSE_ANALYSIS_0714.md),
# reward 重设计推迟; 本实验先看纯 ACT 训练下记忆行为的自然演化与难题解锁情况。
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_TRAIN_ACT_ONLY=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1        # 保留(指纹仪表+recapture, 无害)
# 2026-07-15 用户拍板: 本组 run 全程保持 cpu-offload 不变(组内不换优化器实现, 方法学干净;
# 本组续训本来就一律 weights-only, 开关与否不影响状态丢失)。若中途重启, 记得:
#   CODEBASE_TRAIN_EXTRA_ARGS="--no-load-optim --no-load-rng"
# **之后的新实验统一 CODEBASE_NO_OFFLOAD=1**(弃用 cpu-offload: A_log 写坏与 ckpt resume
# 双毒均独居其代码路径, 详见 notes/CKPT_RESUME_BUGS_0715.md; 新 run 的 ckpt 将原生全量可恢复,
# 首个训练步 = 显存验证, 约 +6GB/卡, OOM 则退回)。
# export CODEBASE_NO_OFFLOAD=1   # 本组不启用; 新实验 wrapper 请打开
# 2026-07-14 首跑事故: 6p6 数据样本更长(响应均值14.9k/最长23.4k), 首个训练步 backward
# 超过 PyTorch NCCL watchdog 默认心跳 480s, rank0 被误杀(SIGABRT)。按官方提示调大。
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_RUN_ID="swecl-4b-actonly-6p6"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swecl-4b-textfmt"

export CODEBASE_NUM_ROLLOUT=120
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=8
export CODEBASE_SEQ_LEN=24576
export CODEBASE_MAX_TOK_PER_GPU=24576
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_HF_CKPT="/data/user_data/qixinx/Qwen3.5-4B"
export CODEBASE_TORCH_DIST="/data/user_data/qixinx/Qwen3.5-4B_torch_dist"
export CODEBASE_PROMPT_DATA="${SD}/data/episodes_6p6_hard.jsonl"
export CODEBASE_BASELINE_ARTIFACT="${SD}/data/baseline_4b_textfmt.json"
export CODEBASE_NGPU=8 CODEBASE_TP=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR=/scratch/qixinx
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

cd "$SD"
echo "=== ACT-only 6p6 RUN=${CODEBASE_RUN_ID} on $(hostname) (code: $SD) ==="
bash ./launch_codebase_adaption_apptainer.sh
