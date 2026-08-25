#!/usr/bin/env bash
set -euo pipefail

# Host-side launcher (orchard cluster edition). Miles runs in the outer
# development SIF; clbench then uses the host Apptainer runtime to start each
# issue SIF.
#
# orchard has no system apptainer. We use a self-contained unprivileged install
# (tools/install-unprivileged.sh) living on shared storage at
# /home/qixinx/apps/apptainer. Because it is relocatable and ships its own
# libexec/utils/etc under one prefix, the babel-era HOST_LIBS bind gymnastics
# (copying RHEL /lib64 deps into the container) is unnecessary here: the whole
# prefix rides in on the /home/qixinx bind and works as-is inside the SIF.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MILES_SIF="${CODEBASE_MILES_SIF:-/project/flame/qixinx/images/miles_dev-202606081341.sif}"
INNER_SCRIPT="${CODEBASE_INNER_SCRIPT:-${SCRIPT_DIR}/run_codebase_adaption_qwen3.5_4B.sh}"
HOST_APPTAINER_PREFIX="${CODEBASE_HOST_APPTAINER_PREFIX:-/home/qixinx/apps/apptainer}"

if [[ -f "${HOST_APPTAINER_PREFIX}/env.sh" ]]; then
  source "${HOST_APPTAINER_PREFIX}/env.sh"
fi

[[ -f "${MILES_SIF}" ]] || { echo "Missing Miles SIF: ${MILES_SIF}" >&2; exit 1; }
[[ -f "${INNER_SCRIPT}" ]] || { echo "Missing inner training script: ${INNER_SCRIPT}" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "Host apptainer is not available" >&2; exit 1; }

# Make the nested (in-container) apptainer calls resolve the same user-space
# install: prepend its bin to the container PATH, and keep proot's loader
# scratch on a writable node-local dir (cluster /tmp quirks break the default).
export APPTAINERENV_PREPEND_PATH="${HOST_APPTAINER_PREFIX}/bin"
export APPTAINERENV_PROOT_TMP_DIR="${PROOT_TMP_DIR:-${TMPDIR:-/tmp}/proot-${UID}}"

BIND_ARGS=(
  --bind /project/flame,/home/qixinx
)

# Opt-in sglang patch (ported from sglang PR #27140): recapture CUDA graphs
# after RL weight updates so replayed graphs never read relocated hybrid-model
# state buffers. File-level bind shadows the SIF copy; no image rebuild.
if [[ "${CODEBASE_SGLANG_RECAPTURE_PATCH:-0}" == "1" ]]; then
  PATCHED_MIXIN="${SCRIPT_DIR}/patches/sglang/scheduler_update_weights_mixin.py"
  [[ -f "${PATCHED_MIXIN}" ]] || { echo "Missing patched mixin: ${PATCHED_MIXIN}" >&2; exit 1; }
  BIND_ARGS+=(--bind "${PATCHED_MIXIN}:/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py")
  echo "[launch] sglang recapture patch ENABLED (PR #27140 port): ${PATCHED_MIXIN}"
fi

# Opt-in Megatron dist-ckpt patch (ported from radixark/Megatron-LM PR #66):
# dp_reshardable optimizer param_state resume — the rank-0-only common list is
# per-rank-length; adopt the loaded rank's structure and carry `step` forward.
# Without this, ANY full resume (with optimizer state) of ckpts saved under
# --optimizer-cpu-offload/--use-precision-aware-optimizer fails with
# "Cannot merge two lists with different lengths" (2026-07-15 incident).
if [[ "${CODEBASE_MEGATRON_CKPT_PATCH:-0}" == "1" ]]; then
  PATCHED_DICT_UTILS="${SCRIPT_DIR}/patches/megatron/dict_utils.py"
  [[ -f "${PATCHED_DICT_UTILS}" ]] || { echo "Missing patched dict_utils: ${PATCHED_DICT_UTILS}" >&2; exit 1; }
  BIND_ARGS+=(--bind "${PATCHED_DICT_UTILS}:/root/Megatron-LM/megatron/core/dist_checkpointing/dict_utils.py")
  echo "[launch] Megatron ckpt-resume patch ENABLED (PR #66 port): ${PATCHED_DICT_UTILS}"
fi

exec apptainer exec --nv "${BIND_ARGS[@]}" "${MILES_SIF}" bash "${INNER_SCRIPT}"
