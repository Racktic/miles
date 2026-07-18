#!/usr/bin/env bash
# WRITE=delta 消融组: R(M_k)=gain[k+1]-gain[k](均值回归对照)。
# 与 actonly-6p6 的差异, 仅三点:
#   1. WRITE 参与训练, reward = gain[k+1] - gain[k](downstream_improve_rewards, window=1)
#   2. CODEBASE_NO_OFFLOAD=1 —— 弃用 cpu-offload(A_log/ckpt 双 bug 根除, 新标准配置);
#   3. --save-hf —— 每个 ckpt 自动导出完整 HF(评测即取即用; 首次启用需张量清点验证)。
# 起点 = 原版 Qwen3.5-4B(用户拍板)。
# 报警规则(预登记, 只报不动训练——用户红线):
#   A) anchor(slot0-3)成功率连续 2 个 8-rollout 窗口下降且累计 >10pp;
#   B) 样本截断率 >35% 持续 2 个窗口;
#   C) A_log 哨兵偏离 -87.121;D) WRITE 合规率或失败后合规率骤降。
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_WRITE_REWARD_MODE=delta
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1        # 指纹仪表+recapture, 保留
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_RUN_ID="swecl-4b-write-delta-6p6"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swecl-4b-textfmt"

export CODEBASE_NUM_ROLLOUT=120
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=8
# --save-hf 已是 run 脚本默认(hf/iter_{rollout_id} 兄弟目录), 无需在此设置
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
echo "=== ACT+WRITE 6p6 RUN=${CODEBASE_RUN_ID} on $(hostname) (code: $SD) ==="
bash ./launch_codebase_adaption_apptainer.sh
