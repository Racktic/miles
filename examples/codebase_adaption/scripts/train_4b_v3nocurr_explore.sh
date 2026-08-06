#!/usr/bin/env bash
# v3nocurr explore run (2026-08-03, user-specified 方案③):
#   - EXACT copy of train_4b_v3nocurr_gated.sh (same v3 pool + shuffle, 191
#     rollouts = one epoch, gated_downstream WRITE reward, filter on) with ONE
#     delta: the ACT exploration reward is enabled (CODEBASE_ACT_EXPLORE_BETA=0.3).
#     At episode end an LLM judge (gpt-5-mini) scores each memory delta
#     M_{k-1}->M_k on 4 dims; the score is standardized within the same
#     ("act", group, trial) GRPO groups and added as adv += 0.3 * explore_adv
#     to that trial's ACT sample. Task rewards and WRITE reward are untouched.
#   - Third single-delta arm alongside grace12 (ACT step grace) and memwin3
#     (WRITE windowed delta), all sharing the gated base for attribution.
# Usage: nohup bash scripts/train_4b_v3nocurr_explore.sh > logs/smith-4b-v3nocurr-explore.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Ray stability mitigation (proven on flame-20: extends node lifespan ~3x).
export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000

export CODEBASE_WRITE_REWARD_MODE=gated_downstream
export CODEBASE_WRITE_FORMAT_BONUS=0.1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# --- ACT exploration reward (the single delta vs the gated arm) ---
export CODEBASE_ACT_EXPLORE_BETA=0.3
# Judge key comes from the repo-level .env (OPENAI_API_KEY -> gpt-5-mini default).
if [ -f "/home/qixinx/miles/.env" ]; then
  set -a; source /home/qixinx/miles/.env; set +a
fi
export CODEBASE_JUDGE_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY missing from /home/qixinx/miles/.env}"
export CODEBASE_JUDGE_API_BASE="https://api.openai.com/v1"
export CODEBASE_JUDGE_MODEL="gpt-5-mini"
export CODEBASE_JUDGE_TIMEOUT=60
export CODEBASE_JUDGE_CONCURRENCY=64

export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/top53.jsonl
# CODEBASE_ROLLOUT_SHUFFLE intentionally NOT set: default 1 = shuffled.
export CODEBASE_DROP_ZERO_STD_GROUPS=1

export CODEBASE_RUN_ID="smith-4b-v3nocurr-explore"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_NUM_ROLLOUT=191
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=8
export CODEBASE_SEQ_LEN=24576
export CODEBASE_MAX_TOK_PER_GPU=20480   # proven-stable value (24576 OOMed twice on think-class runs)
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"

NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_SAVE_DIR="${NODE_RUN_DIR}/ckpt"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"

source "${SD}/scripts/cluster_orchard_env.sh"
export CODEBASE_PROMPT_DATA="${SD}/data/swesmith_episodes_v3_curriculum.jsonl"
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
echo "=== V3NOCURR EXPLORE RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
