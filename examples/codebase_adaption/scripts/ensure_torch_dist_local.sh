#!/usr/bin/env bash
# orchard: torch_dist 参考权重的"现场生成"(DEPLOY_NEW_CLUSTER.md §3.4 的 A7)。
# 因 /home 25G 硬配额 + /project/flame 满, torch_dist(7.9G)不落永久存储,
# 每个训练 job 启动时在计算节点本地 SSD(/tmp)按需转换一次(~10-20min, 1 GPU)。
# 已存在(同一节点重复跑)则跳过。flame 腾出空间后可改回持久化路径并删除本步骤。
set -euo pipefail
DST="${CODEBASE_TORCH_DIST:-/tmp/qixinx/models/Qwen3.5-4B_torch_dist}"
HF_SRC="${CODEBASE_HF_CKPT:-/project/flame/qixinx/models/Qwen3.5-4B}"
SIF="${CODEBASE_MILES_SIF:-/home/qixinx/images/miles_dev-202606081341.sif}"
if [ -f "${DST}/latest_checkpointed_iteration.txt" ]; then
  echo "[ensure_torch_dist] SKIP: ${DST} already present"
  exit 0
fi
source /home/qixinx/apps/apptainer/env.sh
mkdir -p "$(dirname "${DST}")"
echo "[ensure_torch_dist] converting ${HF_SRC} -> ${DST}"
apptainer exec --nv --bind /project/flame,/home/qixinx "${SIF}" bash -c '
  cd /home/qixinx/miles && source scripts/models/qwen3.5-4B.sh &&
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint '"${HF_SRC}"' \
    --save '"${DST}"
[ -f "${DST}/latest_checkpointed_iteration.txt" ] || { echo "[ensure_torch_dist] FAIL: no tracker file"; exit 1; }
echo "[ensure_torch_dist] OK: $(du -sh "${DST}" | cut -f1)"
