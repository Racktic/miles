#!/usr/bin/env bash
# 嵌套容器路径的 GLIBC/fakeroot 直测(不需要 GPU, 不打扰训练):
# 完整复刻 launch_codebase_adaption_apptainer.sh 的 binds 进入 miles SIF,
# 在其内部按 clbench 两套真实参数分别起题容器:
#   [TRAIN] swe_bench_cl 训练参数(无 --fakeroot)
#   [EVAL ] codebase_adaptation 默认参数(带 --fakeroot)—— 7/13 崩的路径
set -uo pipefail
MILES_SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
SIF=$(ls /data/user_data/qixinx/clbench/sifs/swecl_sympy__sympy_*.sif | head -1)
SCR=/scratch/qixinx; mkdir -p "$SCR/tmp" "$SCR/apptainer_tmp" "$SCR/apptainer_cache"
RUNTIME_ROOT="/tmp/nested-fkr-test-$$"; HOST_LIB_DIR="$RUNTIME_ROOT/host-libs"; VARLIB="$RUNTIME_ROOT/varlib"
mkdir -p "$HOST_LIB_DIR" "$VARLIB/mnt/session"
HOST_LIBS=(/lib64/libsubid.so.3 /lib64/libcrypt.so.2 /lib64/libseccomp.so.2 /lib64/libaudit.so.1 /lib64/libsemanage.so.2 /lib64/libcap-ng.so.0 /lib64/libsepol.so.2 /lib64/libbz2.so.1 /lib64/libfuse3.so.3 /lib64/liblz4.so.1 /lib64/liblzma.so.5 /lib64/liblzo2.so.2 /lib64/libz.so.1 /lib64/libzstd.so.1)
BIND=(--bind /data,/home/qixinx,/scratch --bind "$VARLIB:/var/lib/apptainer" --bind /usr/bin/apptainer:/usr/bin/apptainer --bind /usr/libexec/apptainer:/usr/libexec/apptainer --bind /etc/apptainer:/etc/apptainer)
for s in "${HOST_LIBS[@]}"; do cp -L "$s" "$HOST_LIB_DIR/"; BIND+=(--bind "$HOST_LIB_DIR/$(basename $s):/lib/x86_64-linux-gnu/$(basename $s)"); done

echo "###### 外层: $(hostname) glibc=$(ldd --version|head -1|grep -oE '[0-9.]+$') apptainer=$(apptainer --version|grep -oE '[0-9.]+.*')"
apptainer exec "${BIND[@]}" "$MILES_SIF" bash -s <<'INNER'
set -u
export TMPDIR=/scratch/qixinx/tmp APPTAINER_TMPDIR=/scratch/qixinx/apptainer_tmp APPTAINER_CACHEDIR=/scratch/qixinx/apptainer_cache
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
echo "###### 内层(miles容器): glibc=$(ldd --version|head -1|grep -oE '[0-9.]+$'); which fakeroot: $(which fakeroot 2>/dev/null||echo NONE)"
SIF=$(ls /data/user_data/qixinx/clbench/sifs/swecl_sympy__sympy_*.sif | head -1)
SB="/scratch/qixinx/tmp/nested-sb-$$"
apptainer --quiet build --sandbox "$SB" "$SIF" >/dev/null 2>&1 && echo "BUILD_OK" || { echo "BUILD_FAIL"; exit 1; }
echo "--- [TRAIN] swe_bench_cl 参数(无 fakeroot) ---"
apptainer --quiet exec --contain --cleanenv --no-mount hostfs,bind-paths --env LANG=C.UTF-8 --pwd /testbed --writable "$SB" /bin/bash -lc \
  'echo uid=$(id -u); git -C /testbed reset --hard >/dev/null 2>&1 && echo TRAIN_GIT_OK || echo TRAIN_GIT_FAIL; touch /testbed/_t && rm /testbed/_t && echo TRAIN_WRITE_OK' 2>&1 | tail -4
echo "--- [EVAL] codebase 默认参数(带 fakeroot)—7/13 崩的路径 ---"
apptainer --quiet exec --contain --cleanenv --fakeroot --no-mount hostfs,bind-paths --pwd /testbed --writable "$SB" /bin/bash -lc \
  'echo uid=$(id -u); git -C /testbed reset --hard >/dev/null 2>&1 && echo EVAL_GIT_OK || echo EVAL_GIT_FAIL' 2>&1 | tail -6
rm -rf "$SB" "$SB.tmp" 2>/dev/null
echo INNER_DONE
INNER
rm -rf "$RUNTIME_ROOT"
echo "###### NESTED_TEST_DONE on $(hostname)"
