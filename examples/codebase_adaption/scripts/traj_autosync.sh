#!/usr/bin/env bash
# Continuous trajectory backup for a running arm (2026-08-19 lesson: the node
# reclaim took every un-mirrored traj with it).
# Two destinations, because neither alone is safe:
#   - /project/flame backup dir: fast incremental rsync, but the share sits at
#     100% (~14G headroom, group-wide) so it can fill at any time.
#   - HF dataset repo: durable off-cluster copy; the whole traj tars to a few
#     hundred MB, uploaded on eval-point cadence only.
# Runs nice'd with tiny I/O (~50MB/rollout) — safe on a training node, unlike
# the ckpt uploads that once OOM-killed a run via page cache.
# Usage: nohup bash scripts/traj_autosync.sh <RUN_ID> > /tmp/qixinx/traj_autosync.log 2>&1 &
set -uo pipefail
RUN="${1:?usage: traj_autosync.sh <RUN_ID>}"
SRC="/tmp/qixinx/runs/${RUN}/traj"
DST="/project/flame/qixinx/backups/${RUN}-resume/traj"
WORK="/tmp/qixinx/trajsync/${RUN}"
REPO="Racktic/swecl-qwen35-traj"
INTERVAL="${TRAJ_SYNC_INTERVAL:-1800}"
HF_EVERY="${TRAJ_HF_EVERY:-4}"          # upload to HF every Nth cycle
export HF_HOME="${HF_HOME:-/tmp/qixinx/hf_home}"
export HF_TOKEN="${HF_TOKEN:-$(grep HF_TOKEN /home/qixinx/miles/.env | cut -d= -f2)}"
mkdir -p "${WORK}"

cycle=0
while true; do
  cycle=$((cycle + 1))
  if [ -d "${SRC}" ]; then
    mkdir -p "${DST}"
    if nice -n 19 ionice -c3 rsync -a "${SRC}/" "${DST}/" 2>/tmp/qixinx/trajsync_rsync.err; then
      echo "[traj_autosync $(date +%H:%M)] rsync ok: $(du -sh "${DST}" 2>/dev/null | cut -f1) rollouts=$(ls "${DST}/train" 2>/dev/null | wc -l)"
    else
      echo "[traj_autosync $(date +%H:%M)] rsync FAILED (likely /project full): $(tail -n1 /tmp/qixinx/trajsync_rsync.err)"
    fi

    if [ $((cycle % HF_EVERY)) -eq 1 ] && [ -n "${HF_TOKEN}" ]; then
      TAR="${WORK}/${RUN}-traj.tar.gz"
      if nice -n 19 ionice -c3 tar -czf "${TAR}" -C "$(dirname "${SRC}")" "$(basename "${SRC}")" 2>/dev/null; then
        python3 - "$REPO" "$TAR" "${RUN}/traj-latest.tar.gz" <<'PY' || echo "[traj_autosync] HF upload failed"
import os, sys
from huggingface_hub import HfApi
repo, local, path_in_repo = sys.argv[1:4]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo,
                repo_id=repo, repo_type="dataset")
print("[traj_autosync] HF ok:", path_in_repo, os.path.getsize(local) // 1024 // 1024, "MB")
PY
        rm -f "${TAR}"
      fi
    fi
  else
    echo "[traj_autosync $(date +%H:%M)] no traj dir yet at ${SRC}"
  fi
  sleep "${INTERVAL}"
done
