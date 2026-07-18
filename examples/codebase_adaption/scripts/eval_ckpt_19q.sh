#!/usr/bin/env bash
# 对某个 Megatron ckpt 做 19 题(codebase heldout 全量)评测 —— "Megatron 加载式 eval"。
# 原理: weights-only 加载 ckpt + --start-rollout-id 0 + eval-before-train => 先 eval 后训练,
# eval 用的是与训练态完全一致的"原版 vision + ckpt 语言权重"引擎状态, 零 HF 转换零 vision 拼接。
# eval 完成后的那个 rollout_0 训练步是无害废步(存档禁用), sbatch 侧会在 eval 落盘后直接杀掉。
#
# 用法: EVAL_MODE=stateless|replace EVAL_CKPT_DIR=<含latest_txt的目录> bash scripts/eval_ckpt_19q.sh
#   stateless: 19 题单题 no_memory × avg@4   (baseline19_textfmt_eval.jsonl)
#   replace:   19 题 × 5 官方题序, memory 模式 (replace_textfmt_eval.jsonl)
set -euo pipefail
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${EVAL_MODE:?EVAL_MODE=stateless|replace}"
CKPT="${EVAL_CKPT_DIR:?EVAL_CKPT_DIR required (dir containing latest_checkpointed_iteration.txt)}"
TAG="${EVAL_TAG:-iter$(cat "$CKPT/latest_checkpointed_iteration.txt")}"

export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_HF_CKPT="/data/user_data/qixinx/Qwen3.5-4B"          # 引擎加载原版(vision 完整)
export CODEBASE_TORCH_DIST="/data/user_data/qixinx/Qwen3.5-4B_torch_dist"
export CODEBASE_SAVE_DIR="$CKPT"                                      # --load 指向定格 ckpt; 存档已禁用
export CODEBASE_SAVE_INTERVAL=0                                       # 不存档(MILES_DISABLE_FINAL_SAVE=1)
export CODEBASE_NO_OFFLOAD=1                                          # 权重-only 加载, 不需要也不该用 HDO
export CODEBASE_SGLANG_RECAPTURE_PATCH=1                              # 含权重指纹仪表, 校验同步
export CODEBASE_TRAIN_EXTRA_ARGS="--no-load-optim --no-load-rng --start-rollout-id 0"
export CODEBASE_NUM_ROLLOUT=1                                         # 仅为触发 eval-before-train
export CODEBASE_EVAL_BEFORE_TRAIN=1
export CODEBASE_EVAL_INTERVAL=1000000                                 # 禁掉训练后的周期 eval
export CODEBASE_SEQ_LEN=24576 CODEBASE_MAX_TOK_PER_GPU=24576
export CODEBASE_NGPU=8 CODEBASE_TP=2
export CODEBASE_SGLANG_CONCURRENCY=4                                  # 全量投放限流(2026-07-13 规矩)
export CODEBASE_MAX_RESP=32768                                        # eval 无每轮生成上限(与 7/14 官方评测口径一致; v4 漏设致 stateless 用了 2500)
# 废步最小化约束(2026-07-16 两次崩后定案): lr 调度器要求
#   train_iters = num_rollout*rollout_batch*n_samples // global_batch >= 1,
# 且 global_batch 需被 DP 整除 => 1 prompt × 4 samples / gbs=4 => train_iters=1
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
export CODEBASE_USE_WANDB=   # eval 不上 wandb

SCR=/scratch/qixinx
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

cd "$SD"
echo "=== eval_ckpt_19q MODE=$MODE CKPT=$CKPT RUN=$CODEBASE_RUN_ID on $(hostname) ==="
bash ./launch_codebase_adaption_apptainer.sh
