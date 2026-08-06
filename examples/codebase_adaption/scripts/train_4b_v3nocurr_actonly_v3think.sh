#!/usr/bin/env bash
# v3nocurr ACT-only run (2026-07-30, user-specified):
#   - data: the full v3 pool (382 episodes, same jsonl as v3curr) but with
#     CODEBASE_ROLLOUT_SHUFFLE left at its default (1) — episode order is
#     random, so the per-rollout difficulty mix is uniform over time
#     (~50% easy / 34% mid / 16% hard; expected success ~26%). No tier cliffs.
#   - CODEBASE_TRAIN_ACT_ONLY=1: WRITE prompt v3 + WRITE-only thinking stay
#     ON (memory is still written and consumed every trial) but WRITE samples
#     get no gradient — only ACT trains.
#   - CODEBASE_DROP_ZERO_STD_GROUPS=1 (with min_keep batch guard). DELTA vs
#     the old v3curr actonly, which ran without the filter: the v3nocurr pair
#     (this run vs gated) must differ ONLY in CODEBASE_TRAIN_ACT_ONLY, so the
#     filter is ON in both.
# Counterpart: train_4b_v3nocurr_gated.sh (WRITE trained, gated_downstream).
# Usage: nohup bash scripts/train_4b_v3nocurr_actonly_v3think.sh > logs/smith-4b-v3nocurr-actonly.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Ray stability mitigation (proven on flame-20: extends node lifespan ~3x).
export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000

export CODEBASE_TRAIN_ACT_ONLY=1
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

export CODEBASE_RUN_ID="smith-4b-v3nocurr-actonly-v3think"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

# 191 rollouts x 2 episodes = 382 = exactly one full epoch of the v3 pool
# (user decision 2026-07-30): the shuffled arm consumes the same data
# multiset as a full curriculum pass would — order is the only variable.
# At 120 the shuffled arm would eat 3x more hard instances than v3curr's
# first 240 episodes did (15.8% vs 5.4%), confounding reward comparisons.
export CODEBASE_NUM_ROLLOUT=191
export CODEBASE_EVAL_INTERVAL=8
# Interval 8 (user decision 2026-07-30); tighten only if a crash-loop reappears.
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
echo "=== V3NOCURR ACT-ONLY (+v3think WRITE untrained) RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
