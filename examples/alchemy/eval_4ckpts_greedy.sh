#!/bin/bash
# Greedy(temperature=0)重测 4 个 ckpt(wr59/wr79/act99/act119)。HF 已在 _dl,跳过下载。
# 一体化归一化已融进 eval_alchemy(summary.json 直接有 normalized 分)。out-dir 加 -greedy 不覆盖 temp=1。
set -uo pipefail
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
REPO=/home/qixinx/miles
DL=/data/user_data/qixinx/eval_ckpts/_dl
EXPLORE=$REPO/examples/alchemy/eval/prompt_variants/explore_v2.txt
ORACLE=$REPO/examples/alchemy/eval/oracle_cache.json
cd "$REPO"

names=(wr59 wr79 act99 act119)
paths=(
  "qwen3-4b-curr950-writeonly-r120-e10-20260620-024202/iter_0000059"
  "qwen3-4b-curr950-writeonly-r120-e10-20260620-024202/iter_0000079"
  "qwen3-4b-curr950-actonly-r120-e10-20260620-014759/iter_0000099"
  "qwen3-4b-curr950-actonly-r120-e10-20260620-014759/iter_0000119"
)

pkill -9 -f sglang 2>/dev/null; sleep 5
for i in "${!names[@]}"; do
  name=${names[$i]}; mp=$DL/${paths[$i]}
  echo "----- [$name] serve ($mp) -----"
  nohup apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    "$SIF" python -m sglang.launch_server --model-path "$mp" \
    --served-model-name "$name" --tp 1 --dp-size 8 --host 127.0.0.1 --port 30000 \
    --mem-fraction-static 0.6 --trust-remote-code > examples/alchemy/logs/serve_${name}_greedy.log 2>&1 &
  ok=0
  for t in $(seq 1 40); do curl -s http://127.0.0.1:30000/health >/dev/null 2>&1 && { ok=1; break; }; sleep 20; done
  if [ "$ok" != 1 ]; then echo "  [$name] serve 未就绪,跳过"; pkill -9 -f sglang; sleep 8; continue; fi
  echo "  [$name] serve ready, eval (greedy) ..."
  examples/alchemy/eval/run_eval.sh --provider openai --base-url http://127.0.0.1:30000/v1 \
    --model "$name" --episode-file examples/alchemy/data/hard_set_20.json --no-thinking \
    --temperature 0 \
    --summary --summary-mode replace --extra-system-file "$EXPLORE" --workers 8 \
    --out-dir examples/alchemy/logs/eval-ckpt-${name}-greedy > examples/alchemy/logs/eval_${name}_greedy.log 2>&1
  echo "  [$name] done, kill serve"
  pkill -9 -f sglang; sleep 10
done

echo "===== 汇总(GREEDY temp=0, train-aligned hard-20)====="
for name in "${names[@]}"; do
  python3 -c "
import json
try:
    d=json.load(open('examples/alchemy/logs/eval-ckpt-${name}-greedy/summary.json'))
    print(f'  $name: norm_score={d[\"performance_mean\"]:.3f}±{d[\"performance_se\"]:.3f}  i_score={d[\"i_score_mean\"]:+.3f}')
except Exception as e: print(f'  $name: (无结果 {e})')
"
done
echo "ALL DONE GREEDY"
