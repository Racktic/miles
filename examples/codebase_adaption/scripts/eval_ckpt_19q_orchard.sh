#!/usr/bin/env bash
# Orchard variant of eval_ckpt_19q.sh — Megatron-load eval of a frozen ckpt on
# the 19-question heldout (tablib 9 + tenacity 10).
#   replace:   19q x 5 official orderings, memory mode (n=95)
#   stateless: 19q single-instance no_memory x avg@4 (n=76)
# Differences from the babel original: orchard paths (HF ckpt on /project/flame,
# torch_dist on node-local /tmp via ensure script), cluster_orchard_env sourcing,
# node-local run dir. Weights-only load (--no-load-optim), eval-before-train,
# the trailing rollout_0 train step is a harmless waste step — kill after eval
# lands or let it exit.
# Usage (on the node holding the ckpt):
#   EVAL_MODE=replace  EVAL_CKPT_DIR=/tmp/qixinx/runs/<run>/ckpt bash scripts/eval_ckpt_19q_orchard.sh
#   EVAL_MODE=stateless EVAL_CKPT_DIR=... EVAL_TAG=iter47 bash scripts/eval_ckpt_19q_orchard.sh
# To pin a specific iteration, point EVAL_CKPT_DIR at a directory whose
# latest_checkpointed_iteration.txt names it (copy the pointer file if needed).
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${EVAL_MODE:?EVAL_MODE=stateless|replace}"
CKPT="${EVAL_CKPT_DIR:?EVAL_CKPT_DIR required (dir containing latest_checkpointed_iteration.txt)}"
TAG="${EVAL_TAG:-iter$(cat "$CKPT/latest_checkpointed_iteration.txt")}"

export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_SAVE_DIR="$CKPT"
export CODEBASE_SAVE_INTERVAL=0
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export CODEBASE_TRAIN_EXTRA_ARGS="--no-load-optim --no-load-rng --start-rollout-id 0"
export CODEBASE_NUM_ROLLOUT=1
export CODEBASE_EVAL_BEFORE_TRAIN=1
export CODEBASE_EVAL_INTERVAL=1000000
export CODEBASE_SEQ_LEN=24576 CODEBASE_MAX_TOK_PER_GPU=20480
export CODEBASE_NGPU=8 CODEBASE_TP=2
export CODEBASE_SGLANG_CONCURRENCY=4
export CODEBASE_MAX_RESP=32768
# Waste-step constraint (2026-07-16): train_iters = 1*1*4//4 = 1, gbs % DP == 0.
export CODEBASE_ROLLOUT_BATCH_SIZE=1 CODEBASE_N_SAMPLES=4
export CODEBASE_GLOBAL_BATCH_SIZE=4
export CODEBASE_PROMPT_DATA="${SD}/data/episodes_6p6_hard.jsonl"

case "$MODE" in
  stateless)
    export CODEBASE_EVAL_PROMPT_DATA="${SD}/data/baseline19_textfmt_eval.jsonl"
    export CODEBASE_N_EVAL_SAMPLES=4
    ;;
  replace)
    export CODEBASE_EVAL_PROMPT_DATA="${SD}/data/replace_textfmt_eval.jsonl"
    export CODEBASE_N_EVAL_SAMPLES=1
    ;;
  *) echo "unknown EVAL_MODE=$MODE" >&2; exit 1;;
esac

export CODEBASE_RUN_ID="eval-${TAG}-${MODE}"
export CODEBASE_USE_WANDB=

NODE_RUN_DIR="/tmp/qixinx/runs/${CODEBASE_RUN_ID}"
export CODEBASE_TRAJ_DIR="${NODE_RUN_DIR}/traj"
export CODEBASE_TORCH_DIST="/tmp/qixinx/models/Qwen3.5-4B_torch_dist"
mkdir -p "${NODE_RUN_DIR}"

source "${SD}/scripts/cluster_orchard_env.sh"
# cluster env sets CODEBASE_HF_CKPT to the orchard copy (engine loads pristine
# vision weights; language weights come from $CKPT via --load).

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR="${CODEBASE_SCR:-/tmp/qixinx}"
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

bash "${SD}/scripts/ensure_torch_dist_local.sh"

cd "$SD"
echo "=== eval_ckpt_19q(orchard) MODE=$MODE CKPT=$CKPT TAG=$TAG RUN=$CODEBASE_RUN_ID on $(hostname) ==="
bash ./launch_codebase_adaption_apptainer.sh
