#!/bin/bash
# 新 scaffold 测评 sbatch(2026-07-13)。eval-only, 无 Megatron 训练, 8 卡 sglang 推理。
# 用法: sbatch --export=ALL,SUITES="baseline_small" scripts/sbatch_textfmt_evals.sbatch   (冒烟)
#       sbatch --export=ALL,SUITES="baseline icl replace" scripts/sbatch_textfmt_evals.sbatch (全量)
#SBATCH --job-name=textfmt-evals
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
#SBATCH --account=aviralku
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=8:00:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/textfmt-evals-%j.console.log

set -uo pipefail
export PYTHONUNBUFFERED=1
SD=/home/qixinx/miles/examples/codebase_adaption
SUITES="${SUITES:-baseline icl replace}"
echo "===== textfmt evals: suites=[${SUITES}] node=$(hostname) job=${SLURM_JOB_ID} ====="

RC_ALL=0
for suite in ${SUITES}; do
  echo ""
  echo "########## suite: ${suite} 开始 $(date '+%H:%M:%S') ##########"
  bash "${SD}/scripts/eval_textfmt_suite.sh" "${suite}" 2>&1 | tee "${SD}/logs/textfmt-eval-${suite}-${SLURM_JOB_ID}.log"
  RC=${PIPESTATUS[0]}
  echo "########## suite: ${suite} 结束 rc=${RC} $(date '+%H:%M:%S') ##########"
  [ "${RC}" -ne 0 ] && RC_ALL=${RC}
done
exit ${RC_ALL}
