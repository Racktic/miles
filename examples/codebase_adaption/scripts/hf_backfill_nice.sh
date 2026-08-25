#!/usr/bin/env bash
# Upload finished-run checkpoints to the Hub from a node that is STILL TRAINING
# another arm. Plain ckpt_to_hf.sh is unsafe here: reading a 67G torch_dist
# checkpoint pulls it into page cache, page cache counts against the slurm
# cgroup, and on 2026-08-16 that OOM-killed a live run. This wrapper therefore
#   - runs one iteration at a time under nice/ionice idle priority,
#   - drops the page cache it just created (dd iflag=nocache) after each one,
#   - aborts if the cgroup climbs past a high-water mark,
# so a background backfill can share a node with training.
# Usage: nohup bash scripts/hf_backfill_nice.sh <RUN_ID> [iter ...] > /tmp/qixinx/hf_backfill.log 2>&1 &
set -uo pipefail
RUN="${1:?usage: hf_backfill_nice.sh <RUN_ID> [iter ...]}"
shift || true
SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="/tmp/qixinx/runs/${RUN}"
WORK="/tmp/qixinx/hfconv/${RUN}"
export HF_HOME="${HF_HOME:-/tmp/qixinx/hf_home}"
LIMIT_GB="${HF_BACKFILL_CGROUP_LIMIT_GB:-1450}"   # of 1814G; abort above this

cg_dir() { dirname "$(grep -l . /sys/fs/cgroup/system.slice/slurmstepd.scope/*/memory.current 2>/dev/null | head -1)"; }
CG="$(cg_dir)"
cg_gb() { [ -n "${CG}" ] && echo $(( $(cat "${CG}/memory.current") / 1024 / 1024 / 1024 )) || echo 0; }
evict() {  # drop page cache for the files we just read/wrote
  for d in "$@"; do
    [ -d "$d" ] || continue
    find "$d" -type f -size +1M 2>/dev/null | while read -r f; do
      dd if="$f" iflag=nocache count=0 status=none 2>/dev/null
    done
  done
}

if [ $# -gt 0 ]; then ITERS=("$@")
else mapfile -t ITERS < <(ls -d "${RUN_DIR}"/ckpt/iter_* 2>/dev/null | xargs -n1 basename | sort -r); fi

echo "[backfill] ${RUN}: ${#ITERS[@]} iterations queued, cgroup=$(cg_gb)G limit=${LIMIT_GB}G"
ok=0; skip=0; fail=0
for IT in "${ITERS[@]}"; do
  cur=$(cg_gb)
  if [ "${cur}" -gt "${LIMIT_GB}" ]; then
    echo "[backfill] ${IT}: PAUSE, cgroup ${cur}G > ${LIMIT_GB}G — evicting and waiting 5m"
    evict "${RUN_DIR}/ckpt" "${WORK}"
    sleep 300
    cur=$(cg_gb)
    [ "${cur}" -gt "${LIMIT_GB}" ] && { echo "[backfill] still ${cur}G — stopping to protect training"; break; }
  fi
  echo "[backfill] ${IT}: start (cgroup ${cur}G, $(date +%H:%M))"
  nice -n 19 ionice -c3 bash "${SD}/scripts/ckpt_to_hf.sh" "${RUN}" "${IT}" 2>&1 | grep -aE "ckpt_to_hf\]|UPLOAD FAILED|CONVERT FAILED"
  rc=${PIPESTATUS[0]}
  evict "${RUN_DIR}/ckpt/${IT}" "${WORK}"
  if [ "${rc}" -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
  echo "[backfill] ${IT}: done rc=${rc} (cgroup $(cg_gb)G, disk $(df -h /tmp | tail -1 | awk '{print $5}'))"
done
echo "[backfill] ${RUN} finished: ${ok} ok, ${fail} failed"
