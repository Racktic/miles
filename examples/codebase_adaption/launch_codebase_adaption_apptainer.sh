#!/usr/bin/env bash
set -euo pipefail

# Host-side launcher. Miles runs in the outer development SIF; clbench then
# uses the host Apptainer runtime exposed below to start each issue SIF.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MILES_SIF="${CODEBASE_MILES_SIF:-/data/user_data/qixinx/images/miles_dev-202606081341.sif}"
INNER_SCRIPT="${CODEBASE_INNER_SCRIPT:-${SCRIPT_DIR}/run_codebase_adaption_qwen3.5_4B.sh}"
RUNTIME_ROOT="${CODEBASE_APPTAINER_RUNTIME_ROOT:-/tmp/codebase-adaption-apptainer-${UID}}"
HOST_LIB_DIR="${RUNTIME_ROOT}/host-libs"
VARLIB_DIR="${RUNTIME_ROOT}/varlib"

[[ -f "${MILES_SIF}" ]] || { echo "Missing Miles SIF: ${MILES_SIF}" >&2; exit 1; }
[[ -f "${INNER_SCRIPT}" ]] || { echo "Missing inner training script: ${INNER_SCRIPT}" >&2; exit 1; }
command -v apptainer >/dev/null || { echo "Host apptainer is not available" >&2; exit 1; }

HOST_LIBS=(
  /lib64/libsubid.so.3
  /lib64/libcrypt.so.2
  /lib64/libseccomp.so.2
  /lib64/libaudit.so.1
  /lib64/libsemanage.so.2
  /lib64/libcap-ng.so.0
  /lib64/libsepol.so.2
  /lib64/libbz2.so.1
  /lib64/libfuse3.so.3
  /lib64/liblz4.so.1
  /lib64/liblzma.so.5
  /lib64/liblzo2.so.2
  /lib64/libz.so.1
  /lib64/libzstd.so.1
)

mkdir -p "${HOST_LIB_DIR}" "${VARLIB_DIR}/mnt/session"
for source in "${HOST_LIBS[@]}"; do
  [[ -e "${source}" ]] || { echo "Missing host Apptainer library: ${source}" >&2; exit 1; }
  cp -L -- "${source}" "${HOST_LIB_DIR}/"
done

BIND_ARGS=(
  --bind /data,/home/qixinx
  --bind "${VARLIB_DIR}:/var/lib/apptainer"
  --bind /usr/bin/apptainer:/usr/bin/apptainer
  --bind /usr/libexec/apptainer:/usr/libexec/apptainer
  --bind /etc/apptainer:/etc/apptainer
)
for source in "${HOST_LIBS[@]}"; do
  name="$(basename -- "${source}")"
  BIND_ARGS+=(--bind "${HOST_LIB_DIR}/${name}:/lib/x86_64-linux-gnu/${name}")
done

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
