#!/usr/bin/env bash
# 新 scaffold(纯文本格式)测评套件 wrapper(2026-07-13, 用户批准计划)。
# 用法: eval_textfmt_suite.sh <baseline|baseline_small|icl|replace>
# eval-only(NUM_ROLLOUT=0): 无训练、无权重更新; 判分 = clbench 官方步数口径。
# 硬性要求: eval 不受训练侧长度限制 —— 样本截断在 evaluation 分支已全部 guard;
# 每轮生成上限设 32768(实效无上限, 远超实测最长响应, 仅受模型 262k 物理上下文约束)。
set -euo pipefail
SUITE="${1:?用法: $0 <baseline|baseline_small|icl|replace>}"
SD=/home/qixinx/miles/examples/codebase_adaption

case "$SUITE" in
  baseline)
    export CODEBASE_RUN_ID="bl-4b-textfmt"
    export CODEBASE_EVAL_PROMPT_DATA="${SD}/data/baseline_textfmt_eval.jsonl"
    export CODEBASE_N_EVAL_SAMPLES=4        # avg@4, 对齐旧 baseline 口径
    ;;
  baseline_small)
    export CODEBASE_RUN_ID="bl-4b-textfmt-small"
    export CODEBASE_EVAL_PROMPT_DATA="${SD}/data/baseline_textfmt_eval_small.jsonl"
    export CODEBASE_N_EVAL_SAMPLES=2        # 冒烟: 10 题 x2
    ;;
  baseline_patch)
    export CODEBASE_RUN_ID="bl-4b-textfmt-patch"
    export CODEBASE_EVAL_PROMPT_DATA="${SD}/data/baseline_textfmt_eval_patch.jsonl"
    export CODEBASE_N_EVAL_SAMPLES=4        # 补齐 l5-16 GLIBC 崩掉的缺口
    ;;
  icl)
    export CODEBASE_RUN_ID="icl-4b-textfmt"
    export CODEBASE_EVAL_PROMPT_DATA="${SD}/data/icl_textfmt_eval.jsonl"
    export CODEBASE_N_EVAL_SAMPLES=1        # 5 行官方题序 = 5 run
    ;;
  replace)
    export CODEBASE_RUN_ID="iclmem-4b-textfmt"
    export CODEBASE_EVAL_PROMPT_DATA="${SD}/data/replace_textfmt_eval.jsonl"
    export CODEBASE_N_EVAL_SAMPLES=1
    ;;
  *) echo "未知 suite: $SUITE"; exit 2;;
esac

export CODEBASE_NGPU=8 CODEBASE_TP=2
export CODEBASE_SGLANG_CONCURRENCY=4        # x8卡 = 32 episode 并发(用户批准 2026-07-13):
                                            # 全量 eval 一次投 1036 episode, 无闸会容器风暴打爆节点
export CODEBASE_NUM_ROLLOUT=0               # miles 原生 eval-only 模式
export CODEBASE_EVAL_INTERVAL=1
export CODEBASE_SAVE_INTERVAL=0
export CODEBASE_MAX_STEPS_PER_ISSUE=40
export CODEBASE_NUM_ACTS_CAP=0
export CODEBASE_MAX_RESP=32768              # eval 无生成上限(用户要求)
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="textfmt-evals"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_MODEL_SCRIPT="qwen3.5-4B.sh"
export CODEBASE_HF_CKPT="/data/user_data/qixinx/Qwen3.5-4B"
export CODEBASE_TORCH_DIST="/data/user_data/qixinx/Qwen3.5-4B_torch_dist"
export CODEBASE_PROMPT_DATA="${SD}/data/formal_episodes.jsonl"   # loader 需要, eval-only 不消费
export CODEBASE_BASELINE_ARTIFACT="${SD}/data/baseline_4b_merged.json"  # eval gain 仅日志用

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR=/scratch/qixinx
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

cd "$SD"
echo "=== textfmt eval suite=${SUITE} RUN=${CODEBASE_RUN_ID} on $(hostname) ==="
bash ./launch_codebase_adaption_apptainer.sh
