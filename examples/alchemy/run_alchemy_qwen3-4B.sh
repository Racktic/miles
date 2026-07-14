#!/bin/bash
# Symbolic Alchemy two-phase (ACT->REWRITE) TEXT rollout smoke on Qwen3-4B-Instruct-2507 (TP=#GPUs, colocate).
# Adapted minimally from examples/arc_agi3/run_arc_qwen3.5_4B.sh. Text-only: no VLM, no images, no
# ARC online API. The ONLY external dep is dm_alchemy, installed in a custom PYTHONUSERBASE that MUST
# reach every ray worker (see RUNTIME_ENV_JSON below) or the generate procs raise ModuleNotFoundError.
#
# Run INSIDE the miles apptainer container. From the host:
#   SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
#   apptainer exec --nv --bind /data,/home/qixinx \
#     "$SIF" bash /home/qixinx/miles/examples/alchemy/run_alchemy_qwen3.5_4B.sh
set -ex

NGPU="${ALCHEMY_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; [ "${NGPU:-0}" -ge 1 ] || NGPU=4
TP="${ALCHEMY_TP:-$NGPU}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
echo "GPUs=$NGPU  TP=$TP  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export PYTHONUNBUFFERED=1

# Stop ONLY this ray session's stragglers. Do NOT blanket pkill (shared host PID namespace).
ray stop --force 2>/dev/null || true; sleep 2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${MILES_DIR}"

# RHEL host SSL_CERT_* point at /etc/pki paths absent in the Debian container -> point at the real bundle.
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs

# Qwen3-4B-Instruct-2507 megatron spec -> MODEL_ARGS
source "${MILES_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"

# HF checkpoint (tokenizer + megatron->sglang weight bridge). VL config; loaded via mbridge.
# HF checkpoint (tokenizer + the megatron->sglang weight bridge). The stock public Qwen3-4B-Instruct-2507 is a VL+MTP
# config, but that's harmless: the raw/--spec build path below is a pure text GPT decoder that ignores
# vision/mtp in config.json entirely (verified: torch_dist has 378 flat decoder.layers tensors, no mtp/vision).
CKPT=/data/user_data/qixinx/Qwen3-4B-Instruct-2507
# megatron torch_dist ckpt for --load/--ref-load. Built once via examples/alchemy/convert_ckpt.sh
# (tools/convert_hf_to_torch_dist.py). NEVER point --load/--ref-load at the HF dir: with default raw mode
# that path falls back to megatron.bridge.AutoBridge -> Qwen35VLBridge -> 'NoneType has no megatron_module'.
TORCH_DIST=/data/user_data/qixinx/Qwen3-4B-Instruct-2507_torch_dist

ALCHEMY_RUN_ID="${ALCHEMY_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
ALCHEMY_RUN_DIR="${SCRIPT_DIR}/logs/${ALCHEMY_RUN_ID}"
ALCHEMY_TRAJ_DIR="${ALCHEMY_TRAJ_DIR:-${ALCHEMY_RUN_DIR}/traj}"   # 默认落在 logs/<run>/traj（在 /home，不写 /data）
mkdir -p "${ALCHEMY_TRAJ_DIR}" "${ALCHEMY_RUN_DIR}"
echo "=== Alchemy run id=${ALCHEMY_RUN_ID}  traj -> ${ALCHEMY_TRAJ_DIR} ==="
echo "=== training streams: ACT=${ALCHEMY_TRAIN_ACT:-1}  WRITE=${ALCHEMY_TRAIN_WRITE:-1} ==="
# checkpoints persist to /data (big, NFS). --load==--save: a fresh dir cold-starts (no tracker -> loads
# from --ref-load), later runs resume. Saved as megatron torch_dist iter_xxxxxx.
ALCHEMY_SAVE_DIR="${ALCHEMY_SAVE_DIR:-/data/user_data/qixinx/alchemy_runs/${ALCHEMY_RUN_ID}/ckpt}"
echo "=== checkpoints -> ${ALCHEMY_SAVE_DIR} ==="

# wandb optional (disabled unless a key is present); don't leak the key into the log.
set +x
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.wandb.env" ]; then set -a; . "${HOME}/.wandb.env"; set +a; fi
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY}" ] && [ -n "${ALCHEMY_USE_WANDB:-}" ]; then
  WANDB_ARGS=(--use-wandb --wandb-project "${WANDB_PROJECT:-miles-alchemy}" --wandb-group "${WANDB_GROUP:-qwen3-4B}" --wandb-key "${WANDB_API_KEY}")
  if [ -n "${ALCHEMY_WANDB_RUN_ID:-}" ]; then
    WANDB_ARGS+=(--wandb-run-id "${ALCHEMY_WANDB_RUN_ID}")
  fi
  echo "wandb: ENABLED"
else
  WANDB_API_KEY=""   # smoke: keep wandb off by default; also keeps the key out of the runtime env
  echo "wandb: disabled (set ALCHEMY_USE_WANDB=1 + WANDB_API_KEY to enable)"
fi
set -x

CKPT_ARGS=(
   --hf-checkpoint ${CKPT}
   # --ref-load (below) holds the base weights + KL ref. --load MUST be a FRESH dir with no
   # latest_checkpointed_iteration.txt so miles cold-starts (arguments.py: finetune=True, iteration->0,
   # weights from --ref-load, start_rollout_id=0). Pointing --load at the torch_dist made miles think it
   # was RESUMING at iteration 0 -> start_rollout_id=0+1=1 -> range(1, num_rollout=1) empty -> 0 train
   # steps (load+weight-sync then exit, falsely "SUCCEEDED"). This mirrors reference run-qwen3.5-4B.sh
   # (--load <empty miles save dir>, --ref-load <torch_dist>).
   --load ${ALCHEMY_SAVE_DIR}
   --save ${ALCHEMY_SAVE_DIR}
   --save-interval ${ALCHEMY_SAVE_INTERVAL:-10}
   # NO --megatron-to-hf-mode bridge: "bridge" builds the model via megatron.bridge (Qwen35VLBridge ->
   # language_model./vision_model. nested keys), which mismatches the flat decoder.layers torch_dist that
   # convert_hf_to_torch_dist.py produces via --spec + mbridge -> all text weights skipped + BytesIO crash.
   # Default "raw" builds the flat GPTModel from --spec get_qwen3_5_spec, matching the ckpt (like every
   # reference scripts/run-qwen3.5-4B.sh). --load==--save: fresh dir cold-starts, later runs resume.
)

ROLLOUT_ARGS=(
   --prompt-data ${ALCHEMY_PROMPT_DATA:-${MILES_DIR}/examples/alchemy/data/alchemy_episodes.jsonl}
   --input-key prompt
   --metadata-key metadata
   # prompts are conversation-list (text-only): no --apply-chat-template, no prompt-length filter.
   # The rollout ignores seed.prompt and builds its own per-turn ACT/REWRITE messages.
   # --rollout-shuffle: ON by default. Set ALCHEMY_SHUFFLE=0 for CURRICULUM (data fed in jsonl order,
   # i.e. easy->hard from make_curriculum.py); shuffle would destroy that ordering.
   $([ "${ALCHEMY_SHUFFLE:-1}" = "0" ] || echo --rollout-shuffle)
   --num-rollout ${ALCHEMY_NUM_ROLLOUT:-1}
   --rollout-batch-size ${ALCHEMY_ROLLOUT_BATCH_SIZE:-2}
   --n-samples-per-prompt ${ALCHEMY_N_SAMPLES:-8}
   --rollout-max-response-len ${ALCHEMY_MAX_RESP:-2560}
   --rollout-temperature 1
   # train_iters = num_rollout*rollout_batch*n_samples // global_batch  (Megatron, model.py:65).
   # Keep global_batch <= num_rollout*rollout_batch*n_samples or train_iters rounds to 0 -> the
   # OptimizerParamScheduler "lr_decay_steps > 0" assert fails.
   # With rollout_batch=2, n_samples=8: train_iters = num_rollout*16 // 16 = num_rollout.
   --global-batch-size ${ALCHEMY_GLOBAL_BATCH_SIZE:-16}
   # dynamic global batch: set global_batch = ALL of this rollout's samples -> exactly ONE optimizer step
   # per rollout (real GRPO), instead of splitting the variable sample count into num_samples//global_batch
   # tiny steps. (--global-batch-size above still used for the train_iters/lr-scheduler math.)
   # NOTE: this is MUTUALLY EXCLUSIVE with --disable-rollout-trim-samples -- the dynamic-gbs code lives
   # inside `if not disable_rollout_trim_samples` (rollout_data_conversion.py). With dynamic on + dp_size=1
   # (TP=4) the "trim" is a no-op (dynamic_gbs = num_samples), so no GRPO group is ever cut.
   --use-dynamic-global-batch-size
   # balance-data: with dp_size=2 (8 GPUs, TP=4) the default STRIDED split (range(i,N,dp)) gives equal
   # sample COUNTS but random per-rank token loads -> our variable-length packed trials (500..19.5k) land
   # unevenly -> one dp rank packs more microbatches than the other -> a cross-dp collective desyncs and
   # the (reload-default 10min) NCCL allreduce watchdog kills the job. Karmarkar-Karp balanced partition
   # (equal_size=True) gives equal counts AND similar total tokens -> matched microbatch counts -> in sync.
   --balance-data
   --log-multi-turn
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.alchemy.alchemy_rollout.generate
   --custom-config-path ${ALCHEMY_CUSTOM_CONFIG_PATH:-examples/alchemy/alchemy_config.yaml}
   --custom-reward-post-process-path examples.alchemy.alchemy_advantage.reward_post_process
   --custom-rollout-log-function-path examples.alchemy.alchemy_metrics.log_rollout_data
   --custom-eval-rollout-log-function-path examples.alchemy.alchemy_metrics.log_eval_rollout_data
)

EVAL_ARGS=()
if [ -n "${ALCHEMY_EVAL_INTERVAL:-}" ]; then
   EVAL_ARGS=(
      --eval-interval ${ALCHEMY_EVAL_INTERVAL}
      --eval-prompt-data hard20 ${ALCHEMY_EVAL_PROMPT_DATA:-${MILES_DIR}/examples/alchemy/data/hard_set_20_eval.jsonl}
      --n-samples-per-eval-prompt ${ALCHEMY_N_EVAL_SAMPLES:-1}
      --skip-eval-before-train
   )
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   # KL-to-ref on BOTH streams (same coef; pure GRPO collapses the writer's syntax). Needs a megatron
   # ref ckpt — ${CKPT} must be a valid megatron checkpoint (verify before a real run).
   --use-kl-loss
   --kl-loss-coef ${ALCHEMY_KL:-0.01}
   --kl-loss-type k1
   --ref-load ${TORCH_DIST}
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${ALCHEMY_LR:-1e-6}
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   # Optimizer CPU offload: the train-side resident (~16GB) is mostly distributed-optimizer state
   # (fp32 master/m/v). Offloading it to CPU widens the per-microbatch forward headroom (free 63GB ->
   # ~73GB) so the single longest packed sample's fla/GDN + logit peak (~58GB) fits. miles forces
   # use_distributed_optimizer=True (arguments.py:14), which these require. Mirrors reference
   # scripts/run_qwen3_5_35b_a3b_mtp_cp2_ep8.py. overlap d2h/h2d keeps it from killing throughput.
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP}
   --pipeline-model-parallel-size 1
   --context-parallel-size ${ALCHEMY_CP:-1}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   # Official run-qwen3.5-4B.sh stability flags (we'd been missing them):
   # sequence-parallel shards activations (incl layernorm/dropout regions) along the seq dim across the
   # TP group -> frees activation memory (needs TP>1, which we have). accumulate grads in fp32 -> stable
   # gradient all-reduce. Both are standard for this model in the reference recipe.
   --sequence-parallel
   --accumulate-allreduce-grads-in-fp32
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # Qwen3.5's hf_attention REQUIRES thd/packed seqs: get_packed_seq_params() only builds PackedSeqParams
   # when qkv_format==thd (parallel.py:88), else None -> hf_attention.py:191 `assert packed_seq_params is
   # not None` crashes the train forward. miles default is thd; the old --qkv-format bshd (copied from the
   # ARC script) forced None. thd pairs with --use-dynamic-batch-size (bshd would need --micro-batch-size
   # and forbids dynamic batch). Mirrors reference scripts/run-qwen3.5-4B.sh.
   --use-dynamic-batch-size
   # Multi-turn chat: in replace mode the longest single TRAIN sequence = one trial's full chat
   # (system + up to max_steps_per_trial turns of game-state+reasoning). 4096 (megatron default) is far
   # too small and silently truncates late-trial turns. Qwen3-4B-Instruct-2507 is natively 262144-ctx and uses RoPE,
   # so raising the position limit is ~free (no weight change). max-tokens-per-gpu is the real memory knob
   # and MUST be >= the longest single sequence, so keep it >= seq-length (lower it if a run OOMs).
   --seq-length ${ALCHEMY_SEQ_LEN:-32768}
   --max-position-embeddings ${ALCHEMY_SEQ_LEN:-32768}
   # max-tokens-per-gpu = the per-GPU MICROBATCH token budget. Must be >= the longest single packed sample
   # (longest packed ACT trial ~19.5k tokens). 32768 OOM'd in the train fwd/bwd because it packs ~2 long
   # samples; 20480 fits one longest sample per microbatch (half the peak).
   --max-tokens-per-gpu ${ALCHEMY_MAX_TOK_PER_GPU:-20480}
   # chunk the logit/cross-entropy over the sequence so the huge vocab (248320) x long packed sequence
   # doesn't materialise a giant logit tensor (the OOM driver after packing). -1 = off.
   --log-probs-chunk-size ${ALCHEMY_LOGP_CHUNK:-512}
)

MEGATRON_ARGS=(
   --train-backend megatron
   # default 10min gloo timeout is too short: our rollout phase is ~15-20min, so the training side's
   # collective (waiting for rollout data, esp. with dp>1) times out. Raise well above one rollout.
   --distributed-timeout-minutes ${ALCHEMY_DIST_TIMEOUT_MIN:-120}
   --attention-backend flash
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
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

# Default = cuda-graph ON (normal/real run: faster decode). Set ALCHEMY_DRYFAST=1 ONLY for quick smokes
# where you want faster startup at the cost of slower decode (skips cuda-graph capture).
DRYFAST_ARGS=()
if [ -n "${ALCHEMY_DRYFAST:-}" ]; then
   DRYFAST_ARGS=(--sglang-disable-cuda-graph)
   echo "DRY-FAST on (ALCHEMY_DRYFAST=1): skipping cuda-graph capture (faster startup, slower decode)"
else
   echo "cuda-graph ON (normal run); set ALCHEMY_DRYFAST=1 for a fast-startup smoke"
fi

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_NUM_CPUS_ARGS=()
if [ -n "${ALCHEMY_RAY_NUM_CPUS:-}" ]; then
   RAY_NUM_CPUS_ARGS=(--num-cpus "${ALCHEMY_RAY_NUM_CPUS}")
fi
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NGPU} "${RAY_NUM_CPUS_ARGS[@]}" --disable-usage-stats \
   --port=${RAY_GCS_PORT:-6379} --object-store-memory 4000000000 \
   --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASH_PORT:-8265}

# PYTHONUSERBASE MUST reach every ray worker so the generate procs can import dm_alchemy.
set +x
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MILES_DIR}\",
    \"PYTHONUSERBASE\": \"/data/user_data/qixinx/arc_userbase\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SSL_CERT_FILE\": \"/etc/ssl/certs/ca-certificates.crt\",
    \"SSL_CERT_DIR\": \"/etc/ssl/certs\",
    \"MILES_DISABLE_MTP\": \"1\",
    \"PYTHONDONTWRITEBYTECODE\": \"1\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"garbage_collection_threshold:0.8,max_split_size_mb:256\",
    \"ALCHEMY_TRAJ_DIR\": \"${ALCHEMY_TRAJ_DIR}\",
    \"ALCHEMY_TRAIN_ACT\": \"${ALCHEMY_TRAIN_ACT:-1}\",
    \"ALCHEMY_TRAIN_WRITE\": \"${ALCHEMY_TRAIN_WRITE:-1}\",
    \"ALCHEMY_FREEFORM\": \"${ALCHEMY_FREEFORM:-0}\",
    \"ALCHEMY_NO_MEMORY\": \"${ALCHEMY_NO_MEMORY:-0}\",
    \"ALCHEMY_NUM_TRIALS_CAP\": \"${ALCHEMY_NUM_TRIALS_CAP:-0}\",
    \"ALCHEMY_WRITE_SIGNAL\": \"${ALCHEMY_WRITE_SIGNAL:-transition_acc}\",
    \"ALCHEMY_WRITE_IMPROVE_K\": \"${ALCHEMY_WRITE_IMPROVE_K:-1}\",
    \"ALCHEMY_WRITE_K0_MODE\": \"${ALCHEMY_WRITE_K0_MODE:-improve}\",
    \"ALCHEMY_MEMORY_WINDOW_SIZE\": \"${ALCHEMY_MEMORY_WINDOW_SIZE:-1}\",
    \"ALCHEMY_ACT_EXPLORE_BETA\": \"${ALCHEMY_ACT_EXPLORE_BETA:-0}\",
    \"DEEPSEEK_API_KEY\": \"${DEEPSEEK_API_KEY:-}\",
    \"DEEPSEEK_API_BASE\": \"${DEEPSEEK_API_BASE:-https://api.deepseek.com}\",
    \"ALCHEMY_JUDGE_MODEL\": \"${ALCHEMY_JUDGE_MODEL:-deepseek-chat}\",
    \"ALCHEMY_JUDGE_CONCURRENCY\": \"${ALCHEMY_JUDGE_CONCURRENCY:-64}\",
    \"ALCHEMY_JUDGE_TIMEOUT\": \"${ALCHEMY_JUDGE_TIMEOUT:-60}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\"
  }
}"
ray job submit --address="http://127.0.0.1:${RAY_DASH_PORT:-8265}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MEGATRON_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${DRYFAST_ARGS[@]}
