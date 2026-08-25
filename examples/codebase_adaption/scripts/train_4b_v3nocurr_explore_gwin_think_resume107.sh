#!/usr/bin/env bash
# Resume the explore-gwin-think run from the HF backup of iter_0000107 after the
# 2026-08-19 node reclaim destroyed both the torch_dist checkpoints and the
# node-local run dirs. Training config is a VERBATIM copy of
# train_4b_v3nocurr_explore_gwin_think.sh; everything new lives in the seeding
# block below. What this restart is and is not:
#   - Weights: exact (HF iter_0000107 from Racktic/swecl-qwen35-ckpt, converted
#     back to torch_dist on the node).
#   - Data order: exact. rollout_shuffle permutations depend only on
#     (rollout_seed=42, epoch_id) and consumption is a pure counter, so the
#     dataset state at iter107 is reconstructed closed-form: 108 rollouts x
#     rollout_batch_size 2 = 216 prompts consumed; dataset has 382 episodes ->
#     epoch_id=0, sample_offset=216, sample_group_index=216,
#     sample_index=216*8=1728.
#   - Optimizer: LOST (Adam moments were only in torch_dist). --no-load-optim
#     warm restart; expect a brief update-scale transient after rollout 108.
#   - traj/ and hf/ history before 108: lost; new files start at rollout 108.
# The seed flow (download/convert/state-fabrication + the --start-rollout-id
# override) runs ONLY when the ckpt dir is not already seeded; after the first
# successful save (iter >= 108) a crash restart of this same script follows the
# normal full-resume path with no extra args -- so the standing rule "never pass
# --start-rollout-id on crash restarts" is preserved.
# Usage: nohup bash scripts/train_4b_v3nocurr_explore_gwin_think_resume107.sh > /tmp/qixinx/smith-4b-v3nocurr-explore-gwin-think.console.log 2>&1 &
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- verbatim env from train_4b_v3nocurr_explore_gwin_think.sh ----------------
export RAY_task_events_report_interval_ms=0
export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=30000
export RAY_grpc_client_keepalive_time_ms=60000
export RAY_grpc_client_keepalive_timeout_ms=30000

export CODEBASE_WRITE_REWARD_MODE=gated_windowed
export CODEBASE_WRITE_WINDOW=3
export CODEBASE_ACT_THINKING=1
export CODEBASE_WRITE_FORMAT_BONUS=0.1
export CODEBASE_WRITE_PROMPT_V3=1
export CODEBASE_WRITE_THINKING=1
export CODEBASE_NO_OFFLOAD=1
export CODEBASE_SGLANG_RECAPTURE_PATCH=1
export CODEBASE_RAY_SUPERVISED=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export CODEBASE_ACT_EXPLORE_BETA=0.3
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
export CODEBASE_DROP_ZERO_STD_GROUPS=1
export CODEBASE_MULTIBLOCK_FEEDBACK=1

export CODEBASE_RUN_ID="smith-4b-v3nocurr-explore-gwin-think"
# Same wandb id as the original run: the curve continues at rollout/step 108.
export CODEBASE_WANDB_RUN_ID="smith-4b-v3nocurr-explore-gwin-think-0816"
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swesmith-4b"

export CODEBASE_NUM_ROLLOUT=191
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=4
export CODEBASE_SEQ_LEN=20480
export CODEBASE_MAX_TOK_PER_GPU=20480
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

# ---------------- resume-from-HF seeding (idempotent) ----------------
RESUME_ITER=107
START_ROLLOUT=$((RESUME_ITER + 1))
HF_REPO="Racktic/swecl-qwen35-ckpt"
HF_SUBDIR="${CODEBASE_RUN_ID}/iter_0000107"
HFDL="/tmp/qixinx/hfdl/${CODEBASE_RUN_ID}-iter107"
SIF="${CODEBASE_MILES_SIF:-/project/flame/qixinx/images/miles_dev-202606081341.sif}"

TRACKER="${CODEBASE_SAVE_DIR}/latest_checkpointed_iteration.txt"
NEED_SEED=1
if [ -f "${TRACKER}" ]; then
  CUR_IT="$(cat "${TRACKER}" 2>/dev/null || echo 0)"
  # A post-resume save exists (or the seed completed earlier): normal full
  # resume, no seeding and no start-rollout override.
  if [ "${CUR_IT}" -ge "${START_ROLLOUT}" ] 2>/dev/null; then
    NEED_SEED=0
    echo "[resume107] ckpt tracker at iter ${CUR_IT} >= ${START_ROLLOUT}: normal resume path"
  fi
fi

if [ "${NEED_SEED}" = "1" ]; then
  if [ ! -f "${CODEBASE_SAVE_DIR}/.seed107_done" ]; then
    # HF hub cache must live on the node disk: ~/.cache/huggingface sits under
    # the 25G home quota and a 8G ckpt download blows it (seen 2026-08-19).
    export HF_HOME="/tmp/qixinx/hf_home"
    mkdir -p "${HF_HOME}"
    export HF_TOKEN="${HF_TOKEN:-$(grep HF_TOKEN /home/qixinx/miles/.env | cut -d= -f2)}"
    [ -n "${HF_TOKEN}" ] || { echo "[resume107] FATAL: no HF_TOKEN"; exit 1; }

    echo "[resume107] downloading ${HF_REPO}/${HF_SUBDIR} -> ${HFDL}"
    mkdir -p "${HFDL}"
    python3 - "$HF_REPO" "$HF_SUBDIR" "$HFDL" <<'PY'
import os, sys, shutil
from huggingface_hub import snapshot_download
repo, subdir, dst = sys.argv[1:4]
path = snapshot_download(repo, allow_patterns=[f"{subdir}/*"], token=os.environ["HF_TOKEN"])
src = os.path.join(path, subdir)
for f in os.listdir(src):
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print("downloaded:", sorted(os.listdir(dst)))
PY
    # Weights sanity: the 4B ckpt ships 2 safetensors shards.
    N_ST=$(ls "${HFDL}"/*.safetensors 2>/dev/null | wc -l)
    [ "${N_ST}" -ge 1 ] || { echo "[resume107] FATAL: no safetensors in ${HFDL}"; exit 1; }

    echo "[resume107] converting HF -> torch_dist into ${CODEBASE_SAVE_DIR}"
    source /home/qixinx/apps/apptainer/env.sh
    mkdir -p "${CODEBASE_SAVE_DIR}"
    apptainer exec --nv --bind /project/flame,/home/qixinx,/tmp "${SIF}" bash -c '
      cd /home/qixinx/miles && source scripts/models/qwen3.5-4B.sh &&
      PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint '"${HFDL}"' \
        --save '"${CODEBASE_SAVE_DIR}"
    [ -f "${TRACKER}" ] || { echo "[resume107] FATAL: converter left no tracker file"; exit 1; }

    # Reconstruct the dataset-consumption state the lost
    # global_dataset_state_dict_107.pt held (values derived in the header).
    mkdir -p "${CODEBASE_SAVE_DIR}/rollout"
    apptainer exec --bind /home/qixinx,/tmp "${SIF}" python3 - "${CODEBASE_SAVE_DIR}/rollout/global_dataset_state_dict_${RESUME_ITER}.pt" <<'PY'
import sys, torch
torch.save(
    {"sample_offset": 216, "epoch_id": 0, "sample_group_index": 216,
     "sample_index": 1728, "metadata": {}},
    sys.argv[1],
)
print("wrote", sys.argv[1])
PY
    touch "${CODEBASE_SAVE_DIR}/.seed107_done"
    rm -rf "${HFDL}"
  else
    echo "[resume107] seed already done (marker present), launching"
  fi
  # Converted ckpt carries no optimizer/rng state and its tracker says
  # iteration 0, so the start rollout must be forced exactly once here --
  # mirrors the sanctioned eval-flow flag set (eval_ckpt_19q_orchard.sh).
  export CODEBASE_TRAIN_EXTRA_ARGS="${CODEBASE_TRAIN_EXTRA_ARGS:-} --no-load-optim --no-load-rng --start-rollout-id ${START_ROLLOUT}"
fi

cd "$SD"
echo "=== V3NOCURR EXPLORE-GWIN-THINK RESUME@${START_ROLLOUT} RUN=${CODEBASE_RUN_ID} on $(hostname) (ckpt/traj: ${NODE_RUN_DIR}) ==="
bash ./launch_codebase_adaption_apptainer.sh
