#!/bin/bash
# 把一个实验的 torch_dist ckpt 依次:转 HF → 校验 → 上传 HF Hub → 校验 → 删本地 HF → 删 torch_dist。
# 滚动处理(一个 iter 转完传完再删,峰值只多 ~8G),为省磁盘。纯 CPU,不占 GPU。
#
# ⚠️ 转 HF 会丢优化器状态:产物只能 eval/推理,不能 resume 训练。
# ⚠️ 默认保护 latest_checkpointed_iteration.txt 指向的 iter(正在训练 run 的 resume 点);
#    已完成的实验要连最终 ckpt 一起导出 → 加 --include-latest。
#
# 用法:
#   HF_TOKEN=hf_xxx bash export_ckpts_to_hf.sh <CKPT_DIR> <HF_REPO> [--include-latest] [--public] [--dry-run]
#     CKPT_DIR : .../alchemy_runs/<RUNID>/ckpt
#     HF_REPO  : <用户名或org>/<repo名>   (一个实验 = 一个 repo,各 iter 作为子目录 iter_xxxxxxx/)
#   先 --dry-run 看会处理哪些 iter,再正式跑。
set -euo pipefail

SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
ORIGIN_HF=/data/user_data/qixinx/Qwen3-4B-Instruct-2507   # 取 tokenizer/config.json
REPO_ROOT=/home/qixinx/miles

CKPT_DIR="${1:?需要 CKPT_DIR(.../alchemy_runs/<RUNID>/ckpt)}"
HF_REPO="${2:?需要 HF_REPO(用户名/repo名)}"
shift 2
INCLUDE_LATEST=0; DRY=0; PRIVATE=True
for a in "$@"; do
  case "$a" in
    --include-latest) INCLUDE_LATEST=1 ;;
    --public)         PRIVATE=False ;;
    --dry-run)        DRY=1 ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done
: "${HF_TOKEN:?需要 export HF_TOKEN=hf_xxx(写权限)}"

[ -d "$CKPT_DIR" ] || { echo "CKPT_DIR 不存在: $CKPT_DIR"; exit 1; }
latest=$(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
RUN_NAME=$(basename "$(dirname "$CKPT_DIR")")   # 实验名 = run 目录名 = repo 内子文件夹
WORK="$CKPT_DIR/hf_export"; mkdir -p "$WORK"
echo "CKPT_DIR=$CKPT_DIR  HF_REPO=$HF_REPO  subfolder=$RUN_NAME  latest=$latest  include_latest=$INCLUDE_LATEST  private=$PRIVATE  dry_run=$DRY"

for iterdir in "$CKPT_DIR"/iter_*/; do
  [ -d "$iterdir" ] || continue
  iter=$(basename "$iterdir")
  iternum=$((10#$(echo "$iter" | sed 's/iter_//')))
  if [ -n "${ALCHEMY_EXPORT_ONLY_OLDER_THAN_LATEST:-}" ] && [ -n "$latest" ] && [ "$iternum" -gt "$latest" ]; then
    echo "[skip] $iter > latest($latest), likely an in-progress/future checkpoint"
    continue
  fi
  if [ "$iternum" = "$latest" ] && [ "$INCLUDE_LATEST" = 0 ]; then
    echo "[skip] $iter = latest($latest) 是 resume 点,默认保护(用 --include-latest 强制导出)"; continue
  fi
  out="$WORK/$iter"
  echo "================= [$iter] ================="
  if [ "$DRY" = 1 ]; then echo "(dry-run) 会: convert $iterdir → $out → upload $HF_REPO :: $RUN_NAME/$iter → 删本地HF+torch_dist"; continue; fi

  # 1) convert (CPU)
  echo "[$iter] 1/4 convert → $out"
  apptainer exec --bind /data,/home/qixinx \
    --env PYTHONPATH="$REPO_ROOT" --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    "$SIF" python "$REPO_ROOT/tools/convert_torch_dist_to_hf.py" \
      --input-dir "$iterdir" --output-dir "$out" --origin-hf-dir "$ORIGIN_HF" --force

  # 2) 校验本地 HF 完整(config + safetensors + index)
  echo "[$iter] 2/4 verify local HF"
  [ -f "$out/config.json" ] && ls "$out"/*.safetensors >/dev/null 2>&1 || { echo "❌ 转换不完整,中止(未删任何东西)"; exit 1; }

  # 3) upload(token 走 env,不进命令行)+ 远端校验。path_in_repo = <run名>/<iter>
  dest="$RUN_NAME/$iter"
  echo "[$iter] 3/4 upload → $HF_REPO :: $dest"
  apptainer exec --bind /data,/home/qixinx --env HF_TOKEN="$HF_TOKEN" \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt "$SIF" python - "$HF_REPO" "$out" "$dest" "$PRIVATE" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, local, dest, private = sys.argv[1], sys.argv[2], sys.argv[3], (sys.argv[4]=="True")
tok = os.environ["HF_TOKEN"]
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True, private=private, token=tok)
api.upload_folder(folder_path=local, path_in_repo=dest, repo_id=repo, repo_type="model", token=tok,
                  commit_message=f"add {dest}")
remote = [f for f in api.list_repo_files(repo, token=tok) if f.startswith(dest + "/")]
assert any(f.endswith("config.json") for f in remote), f"verify failed: no config.json under {dest}"
assert any(f.endswith(".safetensors") for f in remote), f"verify failed: no safetensors under {dest}"
print(f"  uploaded & verified: {len(remote)} files under {dest}/")
PY

  # 4) 上传成功 → 删本地 HF + torch_dist
  echo "[$iter] 4/4 upload ok → rm local HF + torch_dist"
  rm -rf "$out"
  rm -rf "$iterdir"
  echo "[$iter] done. /data free: $(df -h /data/user_data/qixinx | awk 'NR==2{print $4}')"
done

rmdir "$WORK" 2>/dev/null || true
echo "===== 全部完成。HF repo: https://huggingface.co/$HF_REPO ====="
