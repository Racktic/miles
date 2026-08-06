#!/usr/bin/env bash
# v3nocurr memwin3 run (2026-08-02, user-specified):
#   - EXACT copy of train_4b_v3nocurr_gated.sh (same v3 pool + shuffle, 191
#     rollouts = one epoch, filter on) with ONE delta: the WRITE reward mode is
#     gated_windowed — R(M_k) = format_ok * (mean(reward[k+1..k+3]) -
#     mean(reward[k-2..k])) + 0.1 * format_ok. Prices a 3-trial horizon so
#     cross-task memory accumulation finally has a gradient (gated_downstream's
#     next-trial-only pricing hollowed memory into a one-task rolling buffer:
#     cross-repo keep 28% -> 0.5%). Format gate + bonus kept (drift anchor).
#   - ACT reward unchanged (no STEP_GRACE): isolates the WRITE-side fix; the
#     ACT-side fix (grace12) runs as its own arm on flame-31.
# Usage (direct): nohup bash scripts/train_4b_v3nocurr_memwin3.sh > logs/smith-4b-v3nocurr-memwin3.console.log 2>&1 &
# Usage (queued): sbatch scripts/sbatch_v3nocurr_memwin3.sbatch
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000

export CODEBASE_WRITE_REWARD_MODE=gated_windowed
export CODEBASE_WRITE_WINDOW=3
export CODEBASE_WRITE_FORMAT_BONUS=0.1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_TRAIN_TASK=swe_smith
export CODEBASE_TRAIN_DATASET=data/swe_smith/top53.jsonl
# CODEBASE_ROLLOUT_SHUFFLE intentionally NOT set: default 1 = shuffled.
export CODEBASE_DROP_ZERO_STD_GROUPS=1

export CODEBASE_RUN_ID="smith-4b-v3nocurr-memwin3"
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
echo "=== V3NOCURR MEMWIN3 RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
