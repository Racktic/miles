#!/bin/bash
#SBATCH --job-name=swecl-4b-memwin3
#SBATCH -A aviralku
#SBATCH -p flame
#SBATCH -q flame-64gpu_qos
#SBATCH --nodes=1
#SBATCH -n1 -c200
#SBATCH --mem=1800G
#SBATCH --gres=gpu:8
#SBATCH --time=14-00:00:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/smith-4b-v3nocurr-memwin3.sbatch.out
# Queued arm for the memory windowed-delta WRITE reward (gated_windowed, K=3).
# Self-starting: when Slurm grants a node this launches training directly and
# keeps the allocation for the run's lifetime (crash recovery = ssh in and
# rerun train_4b_v3nocurr_memwin3.sh; the allocation survives script exits
# because this wrapper loops on a hold sleep after the first launch).
set -uo pipefail
source /home/qixinx/apps/apptainer/env.sh
cd /home/qixinx/miles/examples/codebase_adaption
mkdir -p logs
echo "=== sbatch memwin3 on $(hostname) at $(date) ==="
nohup bash scripts/train_4b_v3nocurr_memwin3.sh > logs/smith-4b-v3nocurr-memwin3.console.log 2>&1 &
# Hold the allocation regardless of the training process's fate so the node
# stays ours for restarts (mirrors the long-lived hold-job pattern of
# swecl-4b-rl / swecl-4b-smith-v3).
while true; do sleep 3600; done
