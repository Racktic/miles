#!/bin/bash
# WRITE reward 消融 —— 单个训练 run 提交到 rl 分区(不指定节点, 吃满整机内存, 无 cpu-offload)。
# 用法: sbatch --job-name=<jn> --export=ALL,WRAP=scripts/train_4b_write_delta_6p6.sh scripts/sbatch_write_ablation.sbatch
#   或直接见 submit 两行命令。
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
#SBATCH --account=aviralku
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/%x-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1
SD=/home/qixinx/miles/examples/codebase_adaption
export CODEBASE_RAY_SUPERVISED=1   # 监督式提交: dashboard 打嗝不再连坐训练(2026-07-18)
: "${WRAP:?WRAP=scripts/train_4b_write_{delta,downstream}_6p6.sh required}"
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) mem=ALL(--mem=0) wrapper=$WRAP ==="
cd "$SD"
bash "$WRAP"
