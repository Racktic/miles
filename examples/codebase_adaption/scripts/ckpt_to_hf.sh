#!/usr/bin/env bash
# Convert torch_dist checkpoints to HF format and upload them to
# Racktic/swecl-qwen35-ckpt, one iteration at a time.
#
# Each iteration is converted to a scratch dir, uploaded, then deleted, so the
# extra disk footprint stays at one HF copy (~8G) instead of the whole run.
# Iterations already present on the Hub are skipped, which makes reruns cheap
# and safe after an interruption.
#
# Usage: bash scripts/ckpt_to_hf.sh <RUN_ID> [iter ...]
#   RUN_ID  e.g. smith-4b-v3nocurr-explore-gwin
#   iter    zero-padded dir names (iter_0000003); default = all under ckpt/
set -uo pipefail

RUN="${1:?usage: ckpt_to_hf.sh <RUN_ID> [iter ...]}"
shift || true

SD="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MILES_SIF="${CODEBASE_MILES_SIF:-/project/flame/qixinx/images/miles_dev-202606081341.sif}"

# apptainer is a user-space install; a non-interactive shell won't have it on PATH.
HOST_APPTAINER_PREFIX="${CODEBASE_HOST_APPTAINER_PREFIX:-/home/qixinx/apps/apptainer}"
if [[ -f "${HOST_APPTAINER_PREFIX}/env.sh" ]]; then
  source "${HOST_APPTAINER_PREFIX}/env.sh"
fi
command -v apptainer >/dev/null || { echo "FATAL: apptainer not on PATH" >&2; exit 1; }
ORIGIN_HF="/project/flame/qixinx/models/Qwen3.5-4B"
RUN_DIR="/tmp/qixinx/runs/${RUN}"
WORK="/tmp/qixinx/hfconv/${RUN}"
REPO="Racktic/swecl-qwen35-ckpt"

export HF_TOKEN=$(grep HF_TOKEN /home/qixinx/miles/.env | cut -d= -f2)
[[ -n "${HF_TOKEN}" ]] || { echo "FATAL: no HF_TOKEN in miles/.env" >&2; exit 1; }
[[ -d "${RUN_DIR}/ckpt" ]] || { echo "FATAL: missing ${RUN_DIR}/ckpt" >&2; exit 1; }

if [[ $# -gt 0 ]]; then
  ITERS=("$@")
else
  mapfile -t ITERS < <(ls -d "${RUN_DIR}"/ckpt/iter_* 2>/dev/null | xargs -n1 basename | sort)
fi

# One Hub listing up front; per-iteration queries would be needlessly chatty.
mapfile -t DONE < <(python3 - "$REPO" "$RUN" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, run = sys.argv[1], sys.argv[2]
try:
    files = HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(repo)
except Exception as e:
    print(f"__LIST_FAILED__ {e}", file=sys.stderr)
    sys.exit(0)
# An iteration counts as uploaded only once its weights are there.
for it in sorted({
    f.split("/")[1] for f in files
    if f.startswith(run + "/") and f.endswith(".safetensors")
}):
    print(it)
PY
)
echo "[ckpt_to_hf] ${RUN}: ${#ITERS[@]} local, ${#DONE[@]} already on the Hub"

mkdir -p "${WORK}"
ok=0; skip=0; fail=0
for IT in "${ITERS[@]}"; do
  if [[ " ${DONE[*]} " == *" ${IT} "* ]]; then
    echo "[ckpt_to_hf] ${IT}: already uploaded, skipping"
    skip=$((skip+1)); continue
  fi
  SRC="${RUN_DIR}/ckpt/${IT}"
  DST="${WORK}/${IT}"
  [[ -d "${SRC}" ]] || { echo "[ckpt_to_hf] ${IT}: MISSING locally"; fail=$((fail+1)); continue; }

  echo "[ckpt_to_hf] ${IT}: converting"
  rm -rf "${DST}"
  if ! apptainer exec --bind /project/flame,/home/qixinx,/tmp "${MILES_SIF}" \
      python3 /home/qixinx/miles/tools/convert_torch_dist_to_hf.py \
        --input-dir "${SRC}" --output-dir "${DST}" --origin-hf-dir "${ORIGIN_HF}" --force; then
    echo "[ckpt_to_hf] ${IT}: CONVERT FAILED"
    rm -rf "${DST}"; fail=$((fail+1)); continue
  fi

  echo "[ckpt_to_hf] ${IT}: uploading"
  if python3 - "$REPO" "$RUN" "$IT" "$DST" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, run, it, path = sys.argv[1:5]
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    folder_path=path, path_in_repo=f"{run}/{it}", repo_id=repo, repo_type="model")
print("UPLOAD_OK")
PY
  then
    ok=$((ok+1))
  else
    echo "[ckpt_to_hf] ${IT}: UPLOAD FAILED (local ckpt untouched)"
    fail=$((fail+1))
  fi
  # Drop the HF copy either way; the torch_dist source is still on disk.
  rm -rf "${DST}"
done

rmdir "${WORK}" 2>/dev/null
echo "[ckpt_to_hf] ${RUN} done: ${ok} uploaded, ${skip} skipped, ${fail} failed"
[[ ${fail} -eq 0 ]]
