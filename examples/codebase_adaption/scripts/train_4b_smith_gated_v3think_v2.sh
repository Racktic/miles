#!/usr/bin/env bash
# SWE-smith v2 数据首训(2026-07-21): gated + v3 + thinking, 只换数据不换其他。
# 与 train_4b_write_gated_v3think_6p6.sh 的差异, 仅三点:
#   1. 训练数据 = SWE-smith 33 repo / 333 条 6+6 v2 episode(实测难度梯度,
#      题目跨 episode 零重复; 120 rollout×2 episode/rollout=240 抽取 < 333,
#      每条最多见 1 次 —— 反背题实验主变量);
#   2. CODEBASE_TRAIN_TASK=swe_smith(判分链: 恢复测试 via test_patch + F2P/P2P;
#      workspace 初始化含 git 中和, 防 git log 泄漏 Bug Patch);
#      eval 侧不动 —— 仍是 swecl heldout, 作为跨数据集迁移信号(训练全程不见 swecl);
#   3. MAX_TOK_PER_GPU=24576(用户拍板 2026-07-21: 回到 24576, 再 OOM 再说)。
# 用法(flame-31):
#   cd ~/miles/examples/codebase_adaption
#   nohup bash scripts/train_4b_smith_gated_v3think_v2.sh > logs/smith-4b-gated-v3think-v2.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export CODEBASE_WRITE_REWARD_MODE=gated_downstream
export CODEBASE_WRITE_FORMAT_BONUS=0.1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/top32.jsonl

export CODEBASE_RUN_ID="smith-4b-gated-v3think-v2"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_NUM_ROLLOUT=120
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=8
export CODEBASE_SEQ_LEN=24576
export CODEBASE_MAX_TOK_PER_GPU=20480   # 24576 在 r16 训练步 OOM(2026-07-21, 差 1.1G, 同 v3think r9 签名), 降回实测稳定值
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"

# 节点本地大文件落点(先于 cluster env 设置, 覆盖其 /home 缺省)
NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_SAVE_DIR="${NODE_RUN_DIR}/ckpt"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"

source "${SD}/scripts/cluster_orchard_env.sh"
export CODEBASE_PROMPT_DATA="${SD}/data/swesmith_episodes_6p6_v2.jsonl"
# gated 模式 reward 不吃 baseline(仅 traj 里的 gain 统计字段用); 指到 passk_v1,
# 试点 1256 题有实测值, 其余缺省 0。
export CODEBASE_BASELINE_ARTIFACT="/project/flame/qixinx/swe_smith/passk_v1/baseline.json"
export CODEBASE_NGPU=8 CODEBASE_TP=2

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR="${CODEBASE_SCR:-/tmp/qixinx}"
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

bash "${SD}/scripts/ensure_torch_dist_local.sh"

cd "$SD"
echo "=== SMITH v2 GATED+V3+THINK RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
