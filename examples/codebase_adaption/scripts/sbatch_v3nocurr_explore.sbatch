#!/bin/bash
#SBATCH --job-name=swecl-4b-explore
#SBATCH -A aviralku
#SBATCH -p flame
#SBATCH -q flame-64gpu_qos
#SBATCH --nodes=1
#SBATCH -n1 -c200
#SBATCH --mem=1800G
#SBATCH --gres=gpu:8
#SBATCH --time=14-00:00:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/smith-4b-v3nocurr-explore.sbatch.out
# Queued arm for the ACT exploration reward (方案③: gpt-5-mini judge on memory
# deltas, adv += 0.3 * explore_adv on ACT samples; gated base otherwise).
# Self-starting: when Slurm grants a node this launches training directly and
# holds the allocation for the run's lifetime (crash recovery = ssh in and
# rerun train_4b_v3nocurr_explore.sh; the allocation survives script exits
# because this wrapper loops on a hold sleep after the first launch).
set -uo pipefail
source /home/qixinx/apps/apptainer/env.sh
cd /home/qixinx/miles/examples/codebase_adaption
mkdir -p logs
echo "=== sbatch explore on $(hostname) at $(date) ==="
nohup bash scripts/train_4b_v3nocurr_explore.sh > logs/smith-4b-v3nocurr-explore.console.log 2>&1 &
# Hold the allocation regardless of the training process's fate so the node
# stays ours for restarts (mirrors the memwin3 / swecl-4b-rl hold pattern).
while true; do sleep 3600; done
