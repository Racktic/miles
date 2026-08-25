# orchard 集群路径集(2026-07-18)。wrapper 在设好 CODEBASE_RUN_ID 之后 source 本文件。
# babel 迁移对照:DEPLOY_NEW_CLUSTER.md §4.1;DEPLOY_ROOT=/project/flame/qixinx。
# 2026-07-18:/project/flame 被写满(余 <1G),SIF/torch_dist/pydeps 改放 /home;
# 只有下载时已落地的 Qwen HF 权重和 240 题 SIF 库仍在 flame。腾出空间后可 mv 回去再改这里。
export CODEBASE_MILES_SIF="${CODEBASE_MILES_SIF:-/project/flame/qixinx/images/miles_dev-202606081341.sif}"
export CLBENCH_ROOT="${CLBENCH_ROOT:-/home/qixinx/continual-learning-bench}"
export CLBENCH_SIF_DIR="${CLBENCH_SIF_DIR:-/project/flame/qixinx/clbench_sifs}"
export CODEBASE_PYDEPS="${CODEBASE_PYDEPS:-/home/qixinx/miles_pydeps/codebase_py312_clean}"
export CODEBASE_HF_CKPT="${CODEBASE_HF_CKPT:-/project/flame/qixinx/models/Qwen3.5-4B}"
export CODEBASE_TORCH_DIST="${CODEBASE_TORCH_DIST:-/home/qixinx/models/Qwen3.5-4B_torch_dist}"

# ckpt 落点(暂定):/project/flame 仅剩 ~240G(全组共享,写满会影响他人),放 /home(5T 池
# 剩 ~1.9T)。正式训练前需和用户确认最终位置(每个保存点含 optimizer state,几十 GB 级)。
if [ -n "${CODEBASE_RUN_ID:-}" ]; then
  export CODEBASE_SAVE_DIR="${CODEBASE_SAVE_DIR:-/home/qixinx/codebase_adaption_runs/${CODEBASE_RUN_ID}/ckpt}"
fi

# 节点本地盘:orchard 无 /scratch,计算节点本地 RAID 挂在 /tmp(5.9T)。
export CODEBASE_SCR="${CODEBASE_SCR:-/tmp/qixinx}"
