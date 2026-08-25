#!/bin/bash
# orchard L5 冒烟: 整节点 8 卡, delta wrapper, 1 rollout, 不存 ckpt, 无 wandb 上报。
# torch_dist 在节点本地现场转换(ensure_torch_dist_local.sh)。
#SBATCH -A aviralku
#SBATCH -p preempt
#SBATCH -q preempt_qos
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/%x-%j.out
set -uo pipefail
export PYTHONUNBUFFERED=1
SD=/home/qixinx/miles/examples/codebase_adaption
export CODEBASE_RAY_SUPERVISED=1
echo "=== $(date) job=$SLURM_JOB_ID node=$(hostname) SMOKE(L5) ==="

export CODEBASE_RUN_ID="swecl-4b-smoke-orchard"
export CODEBASE_NUM_ROLLOUT=1
export CODEBASE_SAVE_INTERVAL=0                      # 不存 ckpt(冒烟)
export CODEBASE_TORCH_DIST=/tmp/qixinx/models/Qwen3.5-4B_torch_dist
export CODEBASE_SAVE_DIR=/tmp/qixinx/smoke_runs/${CODEBASE_RUN_ID}/ckpt

echo "### L1: 8-GPU visibility in SIF"
source /home/qixinx/apps/apptainer/env.sh
apptainer exec --nv --bind /project/flame,/home/qixinx /project/flame/qixinx/images/miles_dev-202606081341.sif \
  python -c "import torch; n=torch.cuda.device_count(); print('cuda_devices=', n); exit(0 if n==8 else 1)" || { echo "L1_FAIL"; exit 1; }

echo "### A7: ensure torch_dist on node-local /tmp"
bash "$SD/scripts/ensure_torch_dist_local.sh" || { echo "A7_FAIL"; exit 1; }

echo "### L5: smoke rollout via delta wrapper"
cd "$SD"
bash scripts/train_4b_write_delta_6p6.sh
rc=$?
echo "=== $(date) SMOKE_EXIT rc=$rc ==="
exit $rc
