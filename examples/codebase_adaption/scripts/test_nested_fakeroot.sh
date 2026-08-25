#!/usr/bin/env bash
# 嵌套容器路径的 GLIBC/fakeroot 直测(不需要 GPU, 不打扰训练)——orchard 版:
# 完整复刻 launch_codebase_adaption_apptainer.sh 的进容器方式(用户态 apptainer,
# 无 HOST_LIBS bind),在 miles SIF 内部按 clbench 两套真实参数分别起题容器:
#   [TRAIN] swe_bench_cl 训练参数(无 --fakeroot)
#   [EVAL ] codebase_adaptation 默认参数(带 --fakeroot)—— babel 7/13 崩的路径
set -uo pipefail
MILES_SIF=/project/flame/qixinx/images/miles_dev-202606081341.sif
SIF=$(ls /project/flame/qixinx/swebench_sifs/sympy__sympy-*.sif | head -1)
SCR=/tmp/qixinx; mkdir -p "$SCR/tmp" "$SCR/apptainer_tmp" "$SCR/apptainer_cache" "$SCR/proot"
HOST_APPTAINER_PREFIX=/home/qixinx/apps/apptainer
source "$HOST_APPTAINER_PREFIX/env.sh"
export APPTAINERENV_PREPEND_PATH="$HOST_APPTAINER_PREFIX/bin"
export APPTAINERENV_PROOT_TMP_DIR="$SCR/proot"
BIND=(--bind /project/flame,/home/qixinx)

echo "###### 外层: $(hostname) glibc=$(ldd --version|head -1|grep -oE '[0-9.]+$') apptainer=$(apptainer --version|grep -oE '[0-9.]+.*')"
apptainer exec "${BIND[@]}" "$MILES_SIF" bash -s <<'INNER'
set -u
export TMPDIR=/tmp/qixinx/tmp APPTAINER_TMPDIR=/tmp/qixinx/apptainer_tmp APPTAINER_CACHEDIR=/tmp/qixinx/apptainer_cache
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
echo "###### 内层(miles容器): glibc=$(ldd --version|head -1|grep -oE '[0-9.]+$'); which apptainer: $(which apptainer 2>/dev/null||echo NONE); which fakeroot: $(which fakeroot 2>/dev/null||echo NONE)"
SIF=$(ls /project/flame/qixinx/swebench_sifs/sympy__sympy-*.sif | head -1)
SB="/tmp/qixinx/tmp/nested-sb-$$"
apptainer --quiet build --sandbox "$SB" "$SIF" >/dev/null 2>&1 && echo "BUILD_OK" || { echo "BUILD_FAIL"; exit 1; }
echo "--- [TRAIN] swe_bench_cl 参数(无 fakeroot) ---"
apptainer --quiet exec --contain --cleanenv --no-mount hostfs,bind-paths --env LANG=C.UTF-8 --pwd /testbed --writable "$SB" /bin/bash -lc \
  'echo uid=$(id -u); git -C /testbed reset --hard >/dev/null 2>&1 && echo TRAIN_GIT_OK || echo TRAIN_GIT_FAIL; touch /testbed/_t && rm /testbed/_t && echo TRAIN_WRITE_OK' 2>&1 | tail -4
echo "--- [EVAL] codebase 默认参数(带 fakeroot)—babel 7/13 崩的路径 ---"
apptainer --quiet exec --contain --cleanenv --fakeroot --no-mount hostfs,bind-paths --pwd /testbed --writable "$SB" /bin/bash -lc \
  'echo uid=$(id -u); git -C /testbed reset --hard >/dev/null 2>&1 && echo EVAL_GIT_OK || echo EVAL_GIT_FAIL' 2>&1 | tail -6
rm -rf "$SB" "$SB.tmp" 2>/dev/null
echo INNER_DONE
INNER
echo "###### NESTED_TEST_DONE on $(hostname)"
