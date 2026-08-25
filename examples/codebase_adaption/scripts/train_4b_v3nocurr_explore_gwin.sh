#!/usr/bin/env bash
# v3nocurr explore-gwin run (2026-08-13, user-specified): explore-mbfb arm with
# WRITE reward switched to the windowed differential mode.
#   - EXACT copy of train_4b_v3nocurr_explore_mbfb.sh with ONE experiment delta:
#     CODEBASE_WRITE_REWARD_MODE=gated_windowed (K=3, full format gating):
#       R(M_k) = format_ok * ( mean(reward[k+1..k+3]) - mean(reward[k-2..k]) + 0.1 )
#     User confirmed 2026-08-13: full gating (delta term also gated), K=3.
#     Rationale: gated_downstream prices only the next trial and trained memory
#     into a one-task rolling buffer; the windowed delta prices cross-task
#     accumulation. Known accepted caveat: delta family mean-reversion bias,
#     attenuated by K=3 + within-group normalization.
#   - ACT side unchanged: downstream + exploration reward beta=0.3, gpt-5-mini
#     judge on memory deltas (known farmable; single-variable run for WRITE).
#   - Ops delta (not experimental): gRPC keepalive envs to mitigate the
#     ActorUnavailableError family (idle-connection kills, 6 crashes on the
#     mbfb run); untested mitigation, zero expected training effect.
# Usage: nohup bash scripts/train_4b_v3nocurr_explore_gwin.sh > /tmp/qixinx/smith-4b-v3nocurr-explore-gwin.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Ray stability mitigation (proven on flame-20: extends node lifespan ~3x).
export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000
# Keepalive against idle-TCP reclaim (GCP ~10min): keeps actor gRPC channels warm.
export RAY_grpc_client_keepalive_time_ms=60000
export RAY_grpc_client_keepalive_timeout_ms=30000
# OOM investigation (2026-08-13). Both runs died at iteration 0 inside the
# ref-logprob forward (model.py:265), requesting 11.37 GiB on both nodes.
# That size is the fp32 full-vocab logits tensor for a packed micro-batch at
# the cap: ~20090 tok * 151936 vocab * 4 B = 11.37 GiB. It is allocated by the
# model forward itself, BEFORE any chunked log-prob code runs.
# Tried and REVERTED, neither works:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -- sglang's
#     TorchMemorySaver refuses to run with it and kills the rollout engine.
#   - CODEBASE_LOGP_CHUNK=256 -- math_utils.py:881 only chunks a logits.clone()
#     that is already bounded to chunk_size rows (~155 MB at 512), so it cannot
#     touch the 11.37 GiB forward allocation.
# Context: seq_length is 24576 (below) while max_tokens_per_gpu is 20480, so a
# sample in the 20480..24576 band bypasses the first-fit cap (utils/data.py:265)
# and forms its own oversized micro-batch. Explore-arm lengths (mean 16406,
# max 23213) are the first to reach that band.

export CODEBASE_WRITE_REWARD_MODE=gated_windowed
export CODEBASE_WRITE_WINDOW=3
export CODEBASE_WRITE_FORMAT_BONUS=0.1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# --- ACT exploration reward (unchanged from explore-mbfb) ---
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
export CODEBASE_MULTIBLOCK_FEEDBACK=1

export CODEBASE_RUN_ID="smith-4b-v3nocurr-explore-gwin"
# Date-stamped, decoupled from RUN_ID: an existing wandb id would resume into the old run.
export CODEBASE_WANDB_RUN_ID="smith-4b-v3nocurr-explore-gwin-0813"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_NUM_ROLLOUT=191
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=4
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
echo "=== V3NOCURR EXPLORE-GWIN RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
