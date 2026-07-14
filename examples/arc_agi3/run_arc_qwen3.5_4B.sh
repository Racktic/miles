#!/bin/bash
# 2-GPU (TP=2) colocated VLM GRPO: ARC-AGI-3 memory agent on Qwen3.5-4B.
#
# Run INSIDE the miles apptainer container. From the host:
#   apptainer exec --nv \
#     --bind /data,/home/qixinx/miles,/home/qixinx/ARC-AGI-3-Agents \
#     /data/user_data/qixinx/images/miles_dev-202606081341.sif \
#     bash /home/qixinx/miles/examples/arc_agi3/run_arc_qwen3.5_4B.sh
#
# NOTE: a few container-internal bits (Megatron on PYTHONPATH, exact arg compat) are confirmed
# during the first smoke run; see examples/arc_agi3/README.md.
set -ex

# Auto-adapt to the allocation: 2 GPUs on rl RTX-PRO-6000-96GB, 4 on general L40S. TP = #GPUs (colocate).
NGPU="${ARC_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; [ "${NGPU:-0}" -ge 1 ] || NGPU=2
TP="${ARC_TP:-$NGPU}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
echo "GPUs=$NGPU  TP=$TP  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1

# clean stragglers from a PRIOR run — only THIS ray session. Do NOT blanket `pkill python/ray` here:
# apptainer shares the host PID namespace on this cluster, so that would kill other users' jobs.
ray stop --force 2>/dev/null || true; sleep 2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${MILES_DIR}"

# Some RHEL-family hosts export SSL_CERT_FILE/SSL_CERT_DIR at /etc/pki/... paths that DON'T exist in
# the Debian-based container, and a dependency does ssl.create_default_context(cafile=os.environ[...])
# -> FileNotFoundError. Point them at the container's real CA bundle.
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs

# Qwen3.5-4B megatron (text-backbone) spec; the VLM vision tower is handled by the qwen3_5 bridge.
source "${MILES_DIR}/scripts/models/qwen3.5-4B.sh"     # -> MODEL_ARGS

# MTP-stripped config (weights symlinked): Qwen3.5-4B ships a dense MTP head whose megatron.bridge
# weight mapping crashes the load; MTP is unused for RL. config.json here has mtp_num_hidden_layers
# removed so neither megatron nor sglang build it. See examples/arc_agi3/README.md.
CKPT=/data/user_data/qixinx/qwen3.5-4B-nomtp
ARC_REPO=/home/qixinx/ARC-AGI-3-Agents

# ARC online API creds for the rollout workers (do not echo the key).
set +x
if [ -f "${ARC_REPO}/.env" ]; then set -a; . "${ARC_REPO}/.env"; set +a; fi
ARC_API_KEY="${ARC_API_KEY:-}"
ARC_BASE_URL="${ARC_BASE_URL:-https://three.arcprize.org}"
ARC_BASE_URL="${ARC_BASE_URL%/}"            # strip trailing slash (arc-agi 0.9.8 double-slash -> 404)
ARC_SCHEME="${SCHEME:-https}"; ARC_HOST="${HOST:-three.arcprize.org}"; ARC_PORT="${PORT:-443}"
OPERATION_MODE="${OPERATION_MODE:-online}"

# Per-episode human-readable rollout trajectories (memory/action/reward per turn; no images).
# Each run gets its own logs/<run-id>/ so experiments don't overwrite each other. Pass ARC_RUN_ID
# (e.g. a descriptive name) or it defaults to the SLURM job id / a timestamp.
ARC_RUN_ID="${ARC_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
ARC_RUN_DIR="${SCRIPT_DIR}/logs/${ARC_RUN_ID}"
ARC_TRAJ_DIR="${ARC_TRAJ_DIR:-${ARC_RUN_DIR}/traj}"
mkdir -p "${ARC_TRAJ_DIR}"
echo "=== ARC run id=${ARC_RUN_ID}  (log+traj under examples/arc_agi3/logs/${ARC_RUN_ID}/) ==="

# wandb (optional): key from the env, else from ~/.wandb.env (so it need not be pasted into a command).
# Enabled only if a key is found. Override project/group via WANDB_PROJECT / WANDB_GROUP.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb.env" ]; then set -a; . "${HOME}/.wandb.env"; set +a; fi
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-miles-arc-agi3}"
WANDB_GROUP="${WANDB_GROUP:-qwen3.5-4B}"
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY}" ]; then
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT}" --wandb-group "${WANDB_GROUP}" --wandb-key "${WANDB_API_KEY}")
  echo "wandb: ENABLED (project=${WANDB_PROJECT} group=${WANDB_GROUP})"
else
  echo "wandb: disabled (export WANDB_API_KEY to enable)"
fi
echo "rollout trajectories -> ${ARC_TRAJ_DIR}"
set -x

CKPT_ARGS=(
   --hf-checkpoint ${CKPT}
   --load ${CKPT}
   --megatron-to-hf-mode bridge
)
# Checkpoint saving is ON by default (interval 1000). Set ARC_NO_SAVE=1 (e.g. for dry-runs) to skip it
# entirely: with no --save-interval, should_run_periodic_action() returns False even on the final
# rollout, so no save_model() at all (saves the ~340s checkpoint write).
if [ -z "${ARC_NO_SAVE:-}" ]; then
   CKPT_ARGS+=(--save /data/user_data/qixinx/arc_ckpt/qwen3.5-4B --save-interval "${ARC_SAVE_INTERVAL:-1000}")
fi

ROLLOUT_ARGS=(
   --prompt-data ${MILES_DIR}/examples/arc_agi3/data/arc_games.jsonl
   --input-key prompt
   --metadata-key metadata
   # NOTE: data/arc_games.jsonl prompts are conversation-list (text-only) so the VLM Dataset path is
   # happy WITHOUT --apply-chat-template (keeps seed.prompt a list) and WITHOUT a prompt-length filter
   # (no --rollout-max-context-len, else filter_long_prompt re-parses a templated string and crashes).
   # The rollout ignores seed.prompt entirely and builds its own per-turn messages.
   --rollout-shuffle
   --num-rollout ${ARC_NUM_ROLLOUT:-3}   # override for dry-runs, e.g. ARC_NUM_ROLLOUT=1
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096   # 4096 fits reasoning's long tail (~2200 tok) + memory + action.
                                     # 2048 length-truncated only 8% of turns BUT 77% of the (post-
                                     # ACTION6-fix, now ~9-turn) episodes — one overrun kills the episode.
   --rollout-temperature 1
   --global-batch-size 16          # variable turn-samples/episode; small to guarantee a full step
   --log-multi-turn                # log multi-turn rollout metrics to wandb/tensorboard
)

# Observation mode: ARC_OBS_MODE=objects -> text-only object rollout (no VLM); default -> VLM (image+matrix).
ARC_GEN_FN="examples.arc_agi3.arc_rollout.generate"
if [ "${ARC_OBS_MODE:-}" = "objects" ]; then
   ARC_GEN_FN="examples.arc_agi3.arc_rollout_objects.generate"
   echo "OBS_MODE=objects: text-only object rollout (no VLM)"
fi
CUSTOM_ARGS=(
   --custom-generate-function-path ${ARC_GEN_FN}
   --custom-config-path examples/arc_agi3/arc_config.yaml
   --custom-reward-post-process-path examples.arc_agi3.arc_advantage.reward_post_process
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # Qwen3.5 VLM uses GatedDeltaNet (linear attn), and megatron's GDN rejects PACKED (thd) sequences.
   # Use non-packed bshd + a fixed micro-batch (dynamic batch size implies thd packing).
   --qkv-format bshd
   --micro-batch-size 1
)

MEGATRON_ARGS=(
   --train-backend megatron
   --attention-backend flash
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   # Optimizer = fp32 Adam, FULLY on-GPU (no offload, no bf16 hacks — clean). Auto-scales: 4×L40S TP=4
   # (~25/46GB/card) or 2×RTX-PRO-6000-96GB TP=2 (~40/96GB/card). The real constraint was CPU mem for
   # the colocate LOAD phase (~50GB peak) — give the node ≥96G (cgroup here = 96G).
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node ${NGPU}
   --rollout-num-gpus ${NGPU}
   --colocate
)

# Dry-run fast mode (ARC_DRY_FAST=1): disable sglang cuda-graph capture (~98s faster startup, safe —
# only makes decode slower). NOTE: do NOT add --debug-skip-weight-update here: in colocate mode
# sglang's onloaded weights come from that sync, so skipping it makes rollout emit pure GARBAGE.
DRYFAST_ARGS=()
if [ -n "${ARC_DRY_FAST:-}" ]; then
   DRYFAST_ARGS=(--sglang-disable-cuda-graph)
   echo "DRY-FAST on: skip cuda-graph capture (faster startup)"
fi

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NGPU} --disable-usage-stats \
   --port=${RAY_GCS_PORT:-6379} --object-store-memory 4000000000 \
   --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASH_PORT:-8265}

# PYTHONPATH: miles repo (for examples.arc_agi3.*) + ARC repo (for arc_rl). Megatron is expected to be
# importable in the container already; if not, append its path here.
set +x
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_DIR}:${ARC_REPO}\",
    \"PYTHONUSERBASE\": \"/data/user_data/qixinx/arc_userbase\",
    \"ARC_AGI3_REPO\": \"${ARC_REPO}\",
    \"ARC_API_KEY\": \"${ARC_API_KEY}\",
    \"ARC_BASE_URL\": \"${ARC_BASE_URL}\",
    \"SCHEME\": \"${ARC_SCHEME}\",
    \"HOST\": \"${ARC_HOST}\",
    \"PORT\": \"${ARC_PORT}\",
    \"OPERATION_MODE\": \"${OPERATION_MODE}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SSL_CERT_FILE\": \"/etc/ssl/certs/ca-certificates.crt\",
    \"SSL_CERT_DIR\": \"/etc/ssl/certs\",
    \"MILES_DISABLE_MTP\": \"1\",
    \"PYTHONDONTWRITEBYTECODE\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"garbage_collection_threshold:0.8,max_split_size_mb:256\",
    \"ARC_TRAJ_DIR\": \"${ARC_TRAJ_DIR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"
  }
}"
# keep tracing OFF through the submit so ARC_API_KEY / WANDB_API_KEY don't leak into the log
ray job submit --address="http://127.0.0.1:${RAY_DASH_PORT:-8265}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MEGATRON_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${DRYFAST_ARGS[@]}
