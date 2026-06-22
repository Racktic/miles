#!/bin/bash
# Serve + eval 4 个 ckpt(write-only 59/79, act-only 99/119),train-aligned setting,逐个串行。
# eval = hard-20 + summary-replace + explore prompt + no-thinking(= 训练时的 setting)。
set -uo pipefail
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
REPO=/home/qixinx/miles
DL=/data/user_data/qixinx/eval_ckpts/_dl
EXPLORE=$REPO/examples/alchemy/eval/prompt_variants/explore_v2.txt
ORACLE=$REPO/examples/alchemy/eval/oracle_cache.json
HARD=$REPO/examples/alchemy/data/hard_set_20.json
cd "$REPO"

names=(wr59 wr79 act99 act119)
paths=(
  "qwen3-4b-curr950-writeonly-r120-e10-20260620-024202/iter_0000059"
  "qwen3-4b-curr950-writeonly-r120-e10-20260620-024202/iter_0000079"
  "qwen3-4b-curr950-actonly-r120-e10-20260620-014759/iter_0000099"
  "qwen3-4b-curr950-actonly-r120-e10-20260620-014759/iter_0000119"
)

echo "===== 1) 下载 4 个 ckpt ====="
for i in "${!names[@]}"; do
  apptainer exec --bind /data,/home/qixinx \
    --env HF_TOKEN=${HF_TOKEN:?export HF_TOKEN first} \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt --env SSL_CERT_DIR=/etc/ssl/certs \
    "$SIF" python -c "
from huggingface_hub import snapshot_download
snapshot_download('Racktic/alchemy-qwen3-ckpt', repo_type='model',
  allow_patterns='${paths[$i]}/*', local_dir='$DL')" 2>&1 | grep -viE "Fetching|%|it/s|s/it|warning" | tail -1
  echo "  downloaded ${names[$i]}"
done

echo "===== 2) 逐个 serve + eval + finalize ====="
pkill -9 -f sglang 2>/dev/null; sleep 5
for i in "${!names[@]}"; do
  name=${names[$i]}; mp=$DL/${paths[$i]}
  echo "----- [$name] serve ($mp) -----"
  nohup apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    "$SIF" python -m sglang.launch_server --model-path "$mp" \
    --served-model-name "$name" --tp 1 --dp-size 8 --host 127.0.0.1 --port 30000 \
    --mem-fraction-static 0.6 --trust-remote-code > examples/alchemy/logs/serve_$name.log 2>&1 &
  # wait ready
  ok=0
  for t in $(seq 1 40); do curl -s http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; sleep 20; done
  if [ "$ok" != 1 ]; then echo "  [$name] serve 未就绪,跳过"; pkill -9 -f sglang; sleep 8; continue; fi
  echo "  [$name] serve ready, eval ..."
  examples/alchemy/eval/run_eval.sh --provider openai --base-url http://127.0.0.1:30000/v1 \
    --model "$name" --episode-file examples/alchemy/data/hard_set_20.json --no-thinking \
    --summary --summary-mode replace --extra-system-file "$EXPLORE" --workers 8 \
    --out-dir examples/alchemy/logs/eval-ckpt-$name > examples/alchemy/logs/eval_$name.log 2>&1
  python3 examples/alchemy/eval/finalize.py --run examples/alchemy/logs/eval-ckpt-$name --cache "$ORACLE" >/dev/null 2>&1
  echo "  [$name] done, kill serve"
  pkill -9 -f sglang; sleep 10
done

echo "===== 3) 汇总(train-aligned hard-20)====="
for name in "${names[@]}"; do
  python3 -c "
import json
try:
    d=json.load(open('examples/alchemy/logs/eval-ckpt-$name/normalized.json'))
    print(f'  $name: norm_score={d[\"performance_mean\"]:.3f}±{d[\"performance_se\"]:.3f}  norm_improve={d[\"i_score_mean\"]:+.3f}')
except Exception as e: print(f'  $name: (无结果 {e})')
"
done
echo "ALL DONE"
