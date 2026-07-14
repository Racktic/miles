#!/bin/bash
# 9B 正式训练试跑 @ RTX_PRO_6000(babel-l5-16, 754G, Blackwell)。
# 同时验证两件事: ① Blackwell 架构 SIF 能否跑起来 ② 9B 真实内存峰值(754G 够不够)。
# 配置与 4B 正式 run 对称: num_rollout=120, max40, 5+5, n_samples8, TP2, seq20480, eval@8, ckpt@8。
# 内存日志器把 cgroup 峰值写到共享 .memtrace.csv(l5-16 的 cgroup 从别的节点读不到)。
#SBATCH --job-name=swecl-9b-formal
#SBATCH --partition=rl
#SBATCH --qos=rl_qos
#SBATCH --account=aviralku
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=750G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/swecl-9b-formal-r120.console.log

set -uo pipefail
SD=/home/qixinx/miles/examples/codebase_adaption
export PYTHONUNBUFFERED=1

# ── 实验参数(9B: 不覆盖 MODEL_SCRIPT/HF/TORCH_DIST, run 脚本默认即 9B)──
export CODEBASE_RUN_ID="swecl-9b-formal-r120"
export CODEBASE_NGPU=8 CODEBASE_TP=2
export CODEBASE_NUM_ROLLOUT=120
export CODEBASE_MAX_STEPS_PER_ISSUE=40
export CODEBASE_NUM_ACTS_CAP=0
export CODEBASE_N_SAMPLES=8
export CODEBASE_SEQ_LEN=20480 CODEBASE_MAX_TOK_PER_GPU=20480
export CODEBASE_EVAL_INTERVAL=8
export CODEBASE_SAVE_INTERVAL=8
export CODEBASE_USE_WANDB=1
export WANDB_PROJECT="miles-codebase-adaption" WANDB_GROUP="swecl-9b-formal"
export CODEBASE_WANDB_RUN_ID="${CODEBASE_RUN_ID}"
export CODEBASE_MODEL_SCRIPT="qwen3.5-9B.sh"
export CODEBASE_HF_CKPT="/data/user_data/qixinx/Qwen3.5-9B"
export CODEBASE_TORCH_DIST="/data/user_data/qixinx/Qwen3.5-9B_torch_dist"
export CODEBASE_PROMPT_DATA="${SD}/data/formal_episodes.jsonl"
export CODEBASE_BASELINE_ARTIFACT="${SD}/data/baseline_9b_merged.json"
export CODEBASE_EVAL_PROMPT_DATA="${SD}/data/heldout_episodes.jsonl"
# object-store 16G(754G 节点上更要 cap, 否则 Ray 抢 ~200G+)
export CODEBASE_OBJECT_STORE_MEM=16000000000

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
SCR=/scratch/qixinx
mkdir -p "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/tmp"
export APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache" TMPDIR="$SCR/tmp"
export APPTAINERENV_APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINERENV_APPTAINER_CACHEDIR="$SCR/apptainer_cache" APPTAINERENV_TMPDIR="$SCR/tmp"

# ── 内存日志器: 每 30s 记 cgroup used/peak/oom 到共享 csv(供跨节点监控)──
MEMCSV="${SD}/logs/swecl-9b-formal-r120.memtrace.csv"
FLOG="${SD}/logs/swecl-9b-formal-r120.console.log"
( CGB="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID}/step_extern/user"
  [ -f "$CGB/memory.current" ] || CGB="/sys/fs/cgroup/system.slice/slurmstepd.scope/job_${SLURM_JOB_ID}/step_batch/user"
  echo "ts,step,used_G,peak_G,oom_kill,limit_G" > "$MEMCSV"; PEAK=0
  LIM=$(( $(cat "$CGB/memory.max" 2>/dev/null || echo 0)/1073741824 ))
  while true; do
    U=$(head -1 "$CGB/memory.current" 2>/dev/null|tr -d '[:space:]'); U=${U:-0}; UG=$((U/1073741824))
    [ "$UG" -gt "$PEAK" ] 2>/dev/null && PEAK=$UG
    OK=$(grep -oE 'oom_kill [0-9]+' "$CGB/memory.events" 2>/dev/null|awk '{print $2}'); OK=${OK:-0}
    ST=$(grep -cE 'Timer train end' "$FLOG" 2>/dev/null); ST=${ST:-0}
    echo "$(date +%H:%M),${ST},${UG},${PEAK},${OK},${LIM}" >> "$MEMCSV"
    sleep 30
  done ) &
MEMLOGGER=$!

# ── /scratch 清扫(治 rollout 变慢/RAM 压力)──
( while true; do
    find "$SCR/tmp" -maxdepth 1 -type d -mmin +20 \
      \( -name 'minisweagent-*' -o -name 'clbench-ctmp-*' -o -name 'build-temp-*' -o -name 'bundle-temp-*' \) \
      -exec rm -rf {} + 2>/dev/null; sleep 300
  done ) &
SWEEP=$!

cd "$SD"
echo "=== 9B formal RUN=${CODEBASE_RUN_ID} on $(hostname) job=${SLURM_JOB_ID} 750G TP2 num_rollout=120 eval@8 ckpt@8 ==="
bash ./launch_codebase_adaption_apptainer.sh
RC=$?
echo "=== launch exit code: $RC (mem peak 见 $MEMCSV) ==="
kill $MEMLOGGER $SWEEP 2>/dev/null
exit $RC
