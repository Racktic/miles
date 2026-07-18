#!/bin/bash
# RTX_PRO_6000 节点上的 fakeroot/GLIBC 最小验证(纯 CPU, 不动任何训练):
#   ① 节点诊断(apptainer 模式/subuid/glibc)
#   ② 忠实复现: 按 clbench 原样参数 exec 老 glibc 沙箱 -> 预期 GLIBC_2.38 报错
#   ③ 修复验证: bind 覆盖 /.singularity.d/libs/{faked,libfakeroot.so}(donor=GLIBC_2.34)
#   ④ 备选方案: 完全不用 --fakeroot 时 git reset 是否可行
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --account=aviralku
#SBATCH --nodelist=babel-l5-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/test-fakeroot-glibc-%j.out

set -uo pipefail
echo "########## ① 节点诊断 ##########"
hostname; apptainer --version; ldd --version | head -1
grep -E "^allow setuid" /etc/apptainer/apptainer.conf 2>/dev/null || echo "(读不到 apptainer.conf 的 allow setuid)"
grep -E "^qixinx:|^6016127:" /etc/subuid 2>/dev/null || echo "subuid: 无 qixinx 条目"

SCR=/scratch/qixinx; mkdir -p "$SCR/tmp" "$SCR/apptainer_tmp" "$SCR/apptainer_cache"
export TMPDIR="$SCR/tmp" APPTAINER_TMPDIR="$SCR/apptainer_tmp" APPTAINER_CACHEDIR="$SCR/apptainer_cache"
SIF=$(ls /data/user_data/qixinx/clbench/sifs/swecl_sympy__sympy_*.sif | head -1)
SB="$SCR/tmp/glibc-test-sandbox-$SLURM_JOB_ID"
COMPAT=/data/user_data/qixinx/clbench/fakeroot_compat

echo "########## 沙箱构建(与 clbench 同款: build --sandbox) ##########"
time apptainer --quiet build --sandbox "$SB" "$SIF" && echo BUILD_OK || { echo BUILD_FAIL; exit 1; }
echo "沙箱 glibc: $(apptainer --quiet exec --contain "$SB" ldd --version 2>/dev/null | head -1)"

ARGS=(--contain --cleanenv --fakeroot --no-mount hostfs,bind-paths)
echo "########## ② 忠实复现(clbench 原样参数) ##########"
apptainer --quiet exec "${ARGS[@]}" --pwd /testbed --writable "$SB" /bin/bash -lc \
  'echo uid=$(id -u); git -C /testbed status >/dev/null && echo BASELINE_GIT_OK' 2>&1 | tail -5

echo "########## ③ 修复验证(bind 覆盖 faked + libfakeroot.so) ##########"
FIX=(--bind "$COMPAT/faked:/.singularity.d/libs/faked" --bind "$COMPAT/libfakeroot.so:/.singularity.d/libs/libfakeroot.so")
apptainer --quiet exec "${ARGS[@]}" "${FIX[@]}" --pwd /testbed --writable "$SB" /bin/bash -lc \
  'echo uid=$(id -u); git -C /testbed reset --hard >/dev/null && echo FIX_GIT_RESET_OK; touch /testbed/_wtest && rm /testbed/_wtest && echo FIX_WRITE_OK' 2>&1 | tail -6

echo "########## ④ 备选: 无 --fakeroot ##########"
apptainer --quiet exec --contain --cleanenv --no-mount hostfs,bind-paths --pwd /testbed --writable "$SB" /bin/bash -lc \
  'echo uid=$(id -u); git -C /testbed reset --hard >/dev/null && echo NOFAKEROOT_GIT_OK; touch /testbed/_wtest && rm /testbed/_wtest && echo NOFAKEROOT_WRITE_OK' 2>&1 | tail -6

echo "########## 清理 ##########"
rm -rf "$SB" "$SB.tmp" 2>/dev/null
echo TEST_DONE
