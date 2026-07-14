#!/bin/bash
# Submit the ARC-AGI-3 / Qwen3.5-4B GRPO run to the general partition on 4x L40S (TP=4).
# L40S nodes have ~768G CPU, so we request 256G and the colocate load phase never OOMs; and TP=4
# shrinks per-rank params to ~1B so the fp32 optimizer fits fully on-GPU (clean config, no offload).
#
#   sbatch examples/arc_agi3/sbatch_arc_qwen3.5_4B.sh
#   tail -f examples/arc_agi3/arc_smoke.log     # live log (symlinked to the -o file below)
#
#SBATCH --partition=general
#SBATCH --qos=normal
# No explicit --account: use the submitting user's DEFAULT account. (As of 2026-06-09 qixinx is only
# associated with the group account `aviralku`, so that's still what runs; request a personal account
# from the cluster admin to bill separately, then set it as your default and this picks it up.)
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=arc_qwen35
#SBATCH --output=/home/qixinx/miles/examples/arc_agi3/logs/sbatch-%j.log
#SBATCH --open-mode=truncate

set -x
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
apptainer exec --nv --bind /data,/home/qixinx \
  "$SIF" bash /home/qixinx/miles/examples/arc_agi3/run_arc_qwen3.5_4B.sh
echo "=== sbatch job finished with code $? at $(date) ==="
