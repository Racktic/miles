# Alchemy 自助运行手册(serve / eval / 训练)

> 所有命令在 **babel 计算节点**上、用 apptainer SIF 跑。先确保你在一个有 8×A100 的节点上(见 §0)。
> 关键事实:account=**aviralku**(你只关联这一个);SIF / 模型 / 数据路径见下。

## §0 环境变量(每次新开 shell 先 source 这几行)
```bash
export SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
export REPO=/home/qixinx/miles
export MASTER_ADDR=127.0.0.1; unset RAY_ADDRESS
cd "$REPO/examples/alchemy"
# wandb key(从 ~/.netrc 读,不写进命令行)
export WANDB_API_KEY=$(grep -A2 "machine api.wandb.ai" ~/.netrc | grep password | awk '{print $2}')
```
**申请节点(若没有)**:`salloc -A aviralku -p rl -q rl_qos --gres=gpu:8 -c 64 --mem=512G -t 24:00:00`(rl 分区,8 卡)。

**重要约束**:serve(eval用)和训练**都要占满 8 卡,不能同时跑**。顺序:要么 serve+eval,要么训练;切换前务必清理(§5)。

---

## §1 Serve Qwen3-4B(给 eval 用,推理服务)
```bash
nohup apptainer exec --nv \
  --env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server \
    --model-path /data/user_data/qixinx/Qwen3-4B-Instruct-2507 \
    --served-model-name qwen3-4b --tp 1 --dp-size 8 \
    --host 127.0.0.1 --port 30000 --mem-fraction-static 0.5 --trust-remote-code \
    > logs/serve_qwen3-4B.log 2>&1 &
# 等 ready(~2-4min):看到 "The server is fired up" 或 /health 通
curl -s http://127.0.0.1:30000/health && echo OK
```
(换 Qwen3.5:`--model-path /data/user_data/qixinx/Qwen3.5-4B --served-model-name qwen3.5-4b`,eval 时加 `--no-thinking`。)

---

## §2 Eval:hard-20 上跑 3 组(serve 必须先起好)
公共参数:`--provider openai --base-url http://127.0.0.1:30000/v1 --model qwen3-4b --episode-file examples/alchemy/data/hard_set_20.json --no-thinking --workers 8`

```bash
cd "$REPO"
# 1) baseline(无 prior、无 summary)
examples/alchemy/eval/run_eval.sh --provider openai --base-url http://127.0.0.1:30000/v1 \
  --model qwen3-4b --episode-file examples/alchemy/data/hard_set_20.json --no-thinking --workers 8 \
  --out-dir examples/alchemy/logs/eval-q34b-baseline

# 2) summary(replace 模式,对齐训练)
examples/alchemy/eval/run_eval.sh --provider openai --base-url http://127.0.0.1:30000/v1 \
  --model qwen3-4b --episode-file examples/alchemy/data/hard_set_20.json --no-thinking --workers 8 \
  --summary --summary-mode replace \
  --out-dir examples/alchemy/logs/eval-q34b-summary

# 3) prior(加 prior-info 块)
examples/alchemy/eval/run_eval.sh --provider openai --base-url http://127.0.0.1:30000/v1 \
  --model qwen3-4b --episode-file examples/alchemy/data/hard_set_20.json --no-thinking --workers 8 \
  --prior-info \
  --out-dir examples/alchemy/logs/eval-q34b-prior
```
结果落在各自 `--out-dir`;分数对照 `examples/alchemy/eval/BASELINE_RESULTS.md`(Qwen3.5/Claude 基线)。
eval 跑完 **务必清理 serve**(§5)再做别的。

---

## §3 训练 Qwen3-4B(alchemy 两条流 GRPO)
> 当前训练流已 bake `prompt_variants/explore_v2.txt` 的探索提示(`build_training_system`)。
> 目的不是直接提分,而是把 potion usage 拉起来,让 `F_k` 不再接近 0,从而恢复 WRITE 流信号。
> 冒烟时优先看 §4 的 `WRITE n` 是否从 0 起飞。

**冒烟(2 步,不存档,验证能跑):**
```bash
cd "$REPO/examples/alchemy"
RUNID=$(date +%Y%m%d-%H%M%S)
ALCHEMY_RUN_ID=$RUNID ALCHEMY_USE_WANDB=1 WANDB_GROUP=qwen3-4B-smoke \
ALCHEMY_TP=2 ALCHEMY_CP=1 ALCHEMY_SAVE_INTERVAL=999999 \
ALCHEMY_PROMPT_DATA="$REPO/examples/alchemy/data/alchemy_train_500_steps5.jsonl" \
ALCHEMY_N_SAMPLES=8 ALCHEMY_NUM_ROLLOUT=2 ALCHEMY_ROLLOUT_BATCH_SIZE=8 ALCHEMY_GLOBAL_BATCH_SIZE=64 \
apptainer exec --nv --bind /data,/home/qixinx "$SIF" bash run_alchemy_qwen3-4B.sh > logs/train_$RUNID.log 2>&1 &
echo "RUNID=$RUNID  LOG=logs/train_$RUNID.log"
```

**正式训练(全量 steps20、50 步、开存档):** 把上面改成
`ALCHEMY_PROMPT_DATA=.../alchemy_train_500.jsonl`(max_steps=20 全量)、`ALCHEMY_NUM_ROLLOUT=50`、**去掉 `ALCHEMY_SAVE_INTERVAL=999999`**(默认每 10 步存,**每次 ~66GB**,注意磁盘)。
- 配置说明:Qwen3-4B 标准注意力,**TP2/CP1/DP4**(8 卡)即高效;无需 CP/高 TP(那是 Qwen3.5-GDN 才有的坑)。
- 数据文件:`alchemy_train_500.jsonl`(steps20原生)/ `_steps5.jsonl`(5轮)/ `_steps1.jsonl`(单轮);减 trial 数用 `ALCHEMY_NUM_TRIALS_CAP=N`。

---

## §4 监控(训练 run)
训练脚本已接入 `examples.alchemy.alchemy_metrics.log_rollout_data`,wandb 每个 rollout 会额外记录:
- `alchemy_action/{potion,cauldron,end_trial}_frac`,`alchemy_action/invalid_frac`,`alchemy_action/zero_potion_episode_frac`
- `alchemy_score/act_n`,`alchemy_score/act_raw_mean`,`alchemy_score/act_norm_mean`
- `alchemy_score/norm_trial_0_mean`...`alchemy_score/norm_trial_9_mean`,`alchemy_score/norm_improve`
- `alchemy_write/{write_n,write_mean,kept_frac,fk_mean,fk_zero_frac,fk_ge3_frac,acc_mean}`
- `alchemy_grpo/{act_group_zero_std_frac,write_group_zero_std_frac}`
- `alchemy_response/{act_len_mean,write_len_mean,act_truncated_frac,write_truncated_frac}`

训练脚本也可用 Miles 内置 periodic eval。设置 `ALCHEMY_EVAL_INTERVAL=20` 后,每 20 个 rollout 会在
`data/hard_set_20_eval.jsonl`(由 `hard_set_20.json` 派生) 上跑一次轻量 eval,wandb 记录:
- `eval/hard20/norm_score`
- `eval/hard20/norm_improve`
- `eval/hard20/norm_trial_0_mean`...`eval/hard20/norm_trial_9_mean`

```bash
LOG=logs/train_<RUNID>.log
# 阶段计时(rollout/ref_logp/log_probs/train)
grep -aE "Timer (train_wait|ref_log_probs|log_probs|train) end" "$LOG" | tail
# 两条流样本数
grep -aoE "ACT n=[0-9]+ mean_r=[0-9.]+ . WRITE n=[0-9]+" "$LOG" | tail
# 报错 / softOOM / 看门狗
grep -anE "Watchdog caught|CUDA out of memory|Traceback|softOOM|return OOM" "$LOG" | tail
# GPU
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## §5 清理(⚠️ 切换 serve↔训练 / 跑完 必做)
**`ray job stop` 不够**(进程残留、占卡、ckpt 文件 .nfs busy 删不掉)。**必须:**
```bash
apptainer exec --nv --bind /data,/home/qixinx "$SIF" bash -lc "ray stop --force"
pkill -9 -f "sglang"; pkill -9 -f "python3 train.py"; sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 应全 0
# debug 训练 run 跑完删 ckpt(每个 ~66GB):
rm -rf /data/user_data/qixinx/alchemy_runs/<RUNID>
```

---

## §6 关键事实 / 坑
- account=**aviralku**(你只关联这一个;没有名为 qixinx 的 account)。
- Qwen3-4B torch_dist 已转好:`/data/user_data/qixinx/Qwen3-4B-Instruct-2507_torch_dist`(--ref-load 用)。要重转:`bash examples/alchemy/convert_ckpt_qwen3-4B.sh`(占 1 卡,~2min)。
- **Qwen3.5(GDN)很难搞**(慢/OOM,见 `TRAINING_BUGS_LOG.md`);**Qwen3-4B 计算上碾压(快 7-15×、零 OOM)**,但 WRITE 流为 0(上面 §3 ⚠️)。
- reload-timeout bug 修复已在代码里(`miles/utils/reloadable_process_group.py`);可提 PR 给 radixark/miles(slime 上游同有此 bug)。
- 训练日志直接看 `logs/train_<RUNID>.log`(alchemy 脚本会把 actor 日志写进去);若某次只看到提交信息,用 `ray job logs <jobid>`。
