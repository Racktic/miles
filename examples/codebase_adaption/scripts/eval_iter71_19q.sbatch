#!/bin/bash
# iter_71 的 19 题评测: stateless(19×4) + replace(19×5题序) 串行, Megatron 加载式 eval。
# 用法: sbatch scripts/eval_iter71_19q.sbatch
#SBATCH --job-name=eval-iter71-19q
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=aviralku
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=350G
#SBATCH --time=12:00:00
#SBATCH --output=/home/qixinx/miles/examples/codebase_adaption/logs/eval-iter71-19q-%j.out

set -uo pipefail
export PYTHONUNBUFFERED=1
SD=/home/qixinx/miles/examples/codebase_adaption
export CODEBASE_SGLANG_MEM=0.4   # 48GB 级卡: 引擎 ~19GB + Megatron actor ~13GB + 开销
CKPT=/data/group_data/rl/yuxiaoq/qixinx/codebase_adaption_runs/swecl-4b-actonly-6p6/eval_iter71_ckpt

run_mode() {
  local MODE=$1 EXPECT=$2 TIMEOUT_MIN=$3
  local RUN_ID="eval-iter71-${MODE}"
  local EVALDIR="$SD/logs/$RUN_ID/traj/eval/heldout/rollout_0"
  echo "=== [$(date)] launching $MODE (expect $EXPECT eval episode files) ==="
  EVAL_MODE=$MODE EVAL_CKPT_DIR=$CKPT EVAL_TAG=iter71 \
    bash "$SD/scripts/eval_ckpt_19q.sh" > "$SD/logs/${RUN_ID}.console.log" 2>&1 &
  local LPID=$!
  # 等 eval 轨迹齐 -> 缓冲 3 分钟 -> 杀掉(其后只剩无害废步)
  local waited=0
  while [ "$(ls "$EVALDIR"/ep_*.json 2>/dev/null | wc -l)" -lt "$EXPECT" ]; do
    if ! kill -0 $LPID 2>/dev/null; then echo "[$MODE] launcher exited early (可能 eval 已完成或崩溃)"; break; fi
    sleep 60; waited=$((waited+1))
    if [ $waited -ge $TIMEOUT_MIN ]; then echo "[$MODE] TIMEOUT after ${TIMEOUT_MIN}min"; break; fi
  done
  sleep 180
  echo "=== [$(date)] $MODE eval files: $(ls "$EVALDIR"/ep_*.json 2>/dev/null | wc -l)/$EXPECT — tearing down ==="
  kill $LPID 2>/dev/null || true
  # 杀干净容器内 ray/sglang(按 comm 精确匹配, 防 pkill 自杀)
  for pid in $(ps -u $USER -o pid,comm | awk '$2 ~ /^(sglang::|ray::|raylet|gcs_server)/ {print $1}'); do kill -9 $pid 2>/dev/null || true; done
  apptainer exec /data/user_data/qixinx/images/miles_dev-202606081341.sif ray stop --force 2>/dev/null || true
  sleep 30
}

run_mode stateless 76 180
run_mode replace 5 240

echo "=== [$(date)] all done; scoring ==="
python3 - <<'EOF'
import json, glob, statistics as st
SD="/home/qixinx/miles/examples/codebase_adaption"
B={o["instance_id"]:o["reward"] for o in json.load(open(f"{SD}/data/baseline_4b_textfmt.json"))["instance_outcomes"]}
for mode,agg in (("stateless","per_instance"),("replace","per_run")):
    fs=glob.glob(f"{SD}/logs/eval-iter71-{mode}/traj/eval/heldout/rollout_0/ep_*.json")
    print(f"\n== {mode}: {len(fs)} files ==")
    if not fs: continue
    if agg=="per_instance":
        acc={}
        for f in fs:
            d=json.load(open(f))
            for o in d["outcomes"]:
                if o.get("eval_status")=="env_error": continue
                acc.setdefault(o["instance_id"],[]).append(o["reward"])
        avg={i:st.mean(v) for i,v in acc.items()}
        print(f"instances={len(avg)} cum_reward(avg@n)={sum(avg.values()):.3f}")
        print({i.split('__')[1]:round(v,3) for i,v in sorted(avg.items()) if v>0})
    else:
        crs=[];cgs=[]
        for f in fs:
            d=json.load(open(f))
            outs=[o for o in d["outcomes"] if o.get("eval_status")!="env_error"]
            crs.append(sum(o["reward"] for o in outs))
            cgs.append(sum(o["reward"]-B.get(o["instance_id"],0.0) for o in outs))
        se=lambda v: st.stdev(v)/len(v)**0.5 if len(v)>1 else 0
        print(f"runs={len(crs)} cum_reward={st.mean(crs):.3f}±{se(crs):.3f} cum_gain(vs untrained bl)={st.mean(cgs):.3f}±{se(cgs):.3f}")
EOF
