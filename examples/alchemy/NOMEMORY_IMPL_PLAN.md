# Plan: no-memory (full-history) GRPO baseline for Symbolic Alchemy

## 目标 & 定位
no-memory = **full-history** ablation,对照现有 memory 版(ACT 用 memory / WRITE 写 memory)。
- 整个 episode(num_trials=10、多步对话)= **一条连续序列 = 一个训练 sample**;后面的 trial 直接看到前面所有 trial 的完整历史(**不写 / 不读 memory**)。
- 只有一条 ACT 流(没有 WRITE 流),reward = **每个 trial 段的真实 trial score**(环境 ground-truth,**dense / per-trial process reward,不是 learned PRM**)。
- advantage = GRPO 按 `(group_index, trial_pos)` 跨 sibling 标准化,**产出 per-token(segment-level)advantage**:同一条序列内,trial-k 段的所有 token 赋 adv_k。
- 用途:证明"是否需要 memory"——no-memory(全历史)vs memory(act-only/双训)在 hard-20 上的对比。

## 已确认的关键事实(本次调查,带依据)
1. **token 长度不需要 CP**。用真实 full-history eval(`logs/eval-q34b-v2-baseline`,explore-v2 + no-prior + **no-summary**,num_trials=10,edges=7=hard 档)统计:
   - mean 10.4k / p90 14.7k / **max 15.4k**,0% 超 20480、0% 超 32768。
   - (昨天用 act-only summary-replace traj 重建估的 23.7k/42k 作废——数据源错。)
2. **长度由 packing + `max-tokens-per-gpu` 控制,不是 seq-length**(slime varlen/thd 模式;官方 QA)。我们已开 `--use-dynamic-batch-size`(`run_alchemy_qwen3-4B.sh:193`,日志已生效),所以 `--max-tokens-per-gpu` 真生效。
3. **显存配置**:`ALCHEMY_MAX_TOK_PER_GPU=25600`(覆盖 max 15.4k + 留余量给训练中轨迹变长;避开 32768 打包 2 条长样本的 OOM——脚本注释 `run_alchemy_qwen3-4B.sh:201-203` 记录过 32768 OOM)。**不开 CP**。监控:若 RL 让探索变多、单序列 >25.6k,再调高或开 CP。
4. **advantage 落地必须改核心**(custom-loss 绕不过,见下):`compute_advantages`(`miles/backends/training_utils/loss.py:65`)在 loss 函数前就把 reward 在 `advantages.py` broadcast 成 per-token。loss 层本身已是 per-token 逐元素相乘(`loss_hub/losses.py:93` `torch.cat(batch["advantages"])`、`loss_hub/math_utils.py:240` `-ratio*advantages`),**天然支持 per-token**,瓶颈只在 advantage 计算那一步的 scalar 假设。
5. **必须用 `grpo`,不要 `gspo`**:gspo 是 sequence-level 重要性比率,与 segment-level advantage 语义冲突;grpo 的 per-token ratio 才对。

## 文件改动(全部在 worktree;核心只动 1 处、向后兼容)

### 1. `examples/alchemy/alchemy_rollout.py` — 新增 no-memory 分支(memory 路径不动)
- `generate()` 顶部:`if (seed.metadata or {}).get("no_memory"): return await _generate_nomemory(input)`。现有 per-trial / per-step 路径**原样保留**。
- 新 `_generate_nomemory()`:
  - 整个 episode 用**一条不断增长的 message 列表**跑(full-history):system(= `build_training_system`,含 explore)+ 每步 `render_game_state` + 模型 ACT 输出,逐步追加,**不做 summary-replace**。
  - **记录每个 trial 的 token 区间**:在 response-space(loss_mask 坐标系)里,标出 trial-k 的 ACT token 段 `[start_k, end_k)`。这是 segment-advantage 的关键元数据,塞进 sample.metadata(如 `trial_spans=[(s0,e0),...]`)。
  - 产出**一个 Sample**(整 episode),`tokens` = 完整序列,`loss_mask` 只在模型 ACT 输出的 token 上为 1(user/state token 为 0),`metadata={no_memory:True, group_index, trial_spans, per_trial_scores}`。
  - episode 结束后取 `env.per_trial_scores`(raw;归一化在 group 内抵消)。

### 2. `examples/alchemy/alchemy_advantage.py` — no-memory 分支(memory 分支不动)
- 在现有 `reward_post_process(args, samples)` 里加 `if sample.metadata.get("no_memory")` 分支:
  - 对每个 no-memory sample:按 `(group_index, trial_pos)` 跨 siblings 收集 trial-k 的 raw score → 组内 `(v-mean)/(std+eps)` 得 **adv_k(scalar per trial)**。
  - 用 `trial_spans` 把 adv_k 填进一个**长度 = response_length 的 per-token list**(trial-k 段填 adv_k;非 ACT / 段间 token 填 0,反正 loss_mask=0 不参与)。
  - 返回的 `advantages` 中,该 sample 的元素是这个 **per-token list**(而非 scalar)。
- memory 版样本仍返回 scalar(原逻辑),两种混用安全(下面核心改动按元素类型分流)。

### 3. `miles/backends/training_utils/loss_hub/advantages.py` — 核心,唯一改动,向后兼容(~5 行)
- `grpo` 分支(`:31-35`)加判断:
  ```python
  if args.advantage_estimator in ["grpo", "gspo"]:
      if rewards and isinstance(rewards[0], (list, tuple)):        # per-token (no-memory)
          returns = [torch.tensor(r, dtype=torch.float32, device=kl[i].device) for i, r in enumerate(rewards)]
          assert all(len(returns[i]) == len(kl[i]) for i in range(len(returns)))
      else:                                                         # scalar (现有路径,一行不变)
          rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
          returns = get_grpo_returns(rewards, kl)
      advantages = [r for r in returns]
  ```
- **scalar 路径完全不变** → 所有现有实验零行为影响。`get_grpo_returns` 不动。
- 注:`train_data_conversion.py:29` 的 `assert len(rewards)==len(samples)` 仍成立(外层还是 per-sample list,只是元素从 float 变 list),不用改。`normalize_advantages`(whitening)对 cat 后的 per-token 向量天然兼容,不用改。

### 4. `examples/alchemy/run_alchemy_nomemory_qwen3-4B.sh` — 新 run 脚本(复制 run_alchemy_qwen3-4B.sh 改)
- env 开关:`ALCHEMY_NO_MEMORY=1`(rollout 据此设 metadata `no_memory`;通过 RUNTIME_ENV_JSON 传)。
- `--max-tokens-per-gpu 25600`(`ALCHEMY_MAX_TOK_PER_GPU`)、`--context-parallel-size 1`、**确认 `--advantage-estimator grpo`**(不是 gspo)。
- 不需要 WRITE 相关(min_fk / gprime / wm_* 在 no-memory 分支不用)。
- 其余对齐:explore prompt、curriculum(950, shuffle=0)、num_trials=10、TP/CP、num_rollout、eval-interval=10(periodic eval 用同一 no-memory full-history setting)。

### 不改的:miles 核心除 advantages.py 外一律不动;loss.py / losses.py / math_utils.py / train_data_conversion.py 全部沿用。

## 数据流(一条 no-memory rollout,带 7 个 sibling)
- rollout:整 episode 一条序列 → 1 个 Sample(`trial_spans` + `per_trial_scores` 在 metadata)。
- hook:按 `(group_index, trial_pos)` 跨 siblings 标准化每个 trial_pos 的 score → adv_k;用 trial_spans 铺成 per-token list。
- `compute_advantages`:走新分支,per-token list → per-sample 的 per-token 张量(跳过 broadcast)。
- loss:`torch.cat(advantages)` → `-ratio*advantages` 逐 token 生效。

## 风险 / 监控
- **单序列变长**:25600 基于当前 3.7 turns/trial(→15.4k)。RL 若让探索增多 → 监控 train 侧实际单序列 token / truncation,真触顶再调高 max-tokens-per-gpu 或开 CP。
- **OOM**:25600 介于"放得下 1 条 15.4k"与"打不下 2 条"之间;若 train fwd/bwd OOM,先降 max-tokens-per-gpu。
- **segment 边界对齐**:trial_spans 必须和 loss_mask / response token 严格对齐(off-by-one 会把 advantage 贴错段)——dry-run 必须逐 token 核对。

## 验证(分阶段)
1. **rollout dry-run**(不训练):跑 sglang :30000,1 episode 3-4 trials,打印每个 trial 的 token 区间 + per_trial_scores。断言:trial_spans 覆盖且不重叠、∪ = 所有 ACT token、loss_mask 在这些段=1。
2. **hook 单测**(纯 python):合成 2 siblings × trial_spans → 断言每个 `(group,trial_pos)` 的 per-token advantage 在对应段=组内标准化值、段外=0、长度=response_length。
3. **advantages.py 单测**:喂 per-token list → 断言 returns[i] 原样保留 per-token(不被 broadcast 成常数);喂 scalar → 断言走原 get_grpo_returns(回归)。
4. **one-step train**(`--num-rollout 1`):grpo、ref 载入、KL 有限、序列不被 trim、step 完成、log 单序列长度分布;确认无 OOM。

## 关键文件引用
- 改:`examples/alchemy/alchemy_rollout.py`(新分支)·`examples/alchemy/alchemy_advantage.py`(新分支)·`miles/backends/training_utils/loss_hub/advantages.py:31-35`(核心,~5 行,向后兼容)·`examples/alchemy/run_alchemy_nomemory_qwen3-4B.sh`(新)
- 参考:`miles/backends/training_utils/loss.py:65`(advantage 在 loss 前算)·`loss_hub/losses.py:93` / `math_utils.py:240`(loss 层 per-token)·`loss_hub/math_utils.py:414-421`(get_grpo_returns scalar broadcast)·现有 memory 版 `alchemy_advantage.py` / `alchemy_rollout.py`(ACT 流分组逻辑可复用)
- token 依据:`logs/eval-q34b-v2-baseline`(full-history 实测分布)
