# ACT Reward Design Report: Memory-Delta Exploration Signal

## 1. Core Correction

我们衡量 ACT exploration 的目标不是判断一段轨迹“看起来有没有探索”，而是判断：

```text
这次 trial 之后，memory 相比上一次是否新增了有用 knowledge？
```

所以更合理的 LLM-as-judge 输入不是 raw trajectory，而是：

```text
M_{k-1}: trial k 之前的 memory
M_k:     trial k 之后更新出来的 memory
```

judge 评估的是 `M_{k-1} -> M_k` 的 semantic delta。

这比直接喂 trajectory 更干净，因为我们的 project 目标本来就是：

```text
experience -> memory -> future decision
```

如果一次 ACT exploration 真有价值，它应该在 WRITE 之后表现为 memory 中新增了可复用知识。反过来，如果 memory 没变、只是改写措辞、或者删掉了重要条件，那说明这次 experience 对 knowledge state 没有贡献，甚至有害。

## 2. Why Memory Delta Is Better Than Trajectory Judge

直接 judge trajectory 有几个问题：

- 容易被模型 verbose reasoning 迷惑。
- 不同 trial 长度差异很大，长轨迹看起来更“努力探索”。
- judge 需要理解 action validity、state transition、reward scale，负担太重。
- 它衡量的是 behavior appearance，不一定衡量 knowledge gain。

memory-delta judge 更贴近我们真正关心的量：

```text
Knowledge gain from this trial.
```

它回答的问题是：

- `M_k` 相比 `M_{k-1}` 是否新增了 potion effect？
- 新增内容是不是 action-relevant？
- 是否记录了 no-effect / negative effect 这类可避免未来错误的知识？
- 是否保留了之前 memory 里的重要事实？
- 是否只是 rephrasing，没有新增信息？
- 是否 hallucinate 或 overgeneralize？

## 3. Proposed ACT Exploration Reward

对 trial `k`，我们已经在线生成：

```text
M_{k-1}: previous memory
T_k: trial k experience
M_k: updated memory
```

ACT exploration reward 可以定义为：

```text
explore_reward_k = Judge(M_{k-1}, M_k)
```

然后 ACT reward 是：

```text
act_reward_k = task_reward_k + beta * explore_reward_k
```

其中 `task_reward_k` 可以先保持当前 raw trial reward；后续更合理的是 oracle-normalized trial reward。

## 4. Judge Dimensions

Judge 不应该奖励 memory 变长本身。它应该奖励**非冗余、可执行、保真**的 knowledge update。

建议四个维度，每个 `0-2`：

### 4.1 New Actionable Knowledge

`M_k` 是否新增了可用于未来 ACT 的规则？

```text
0 = 没有新增可用知识，或只是重复旧 memory
1 = 新增了少量局部事实，但不完整
2 = 新增了清晰、可执行的 potion/stone/reward 规则
```

例子：

```text
Pink: blue large round -> blue small round, reward +1 -> +15
```

这种是高价值新增知识。

### 4.2 Evidence Specificity

新增内容是否具体到 stone feature、potion color、方向和 reward change？

```text
0 = vague，例如 “pink may be useful”
1 = 有部分条件，但缺 feature / reward change
2 = 明确写出作用对象、变化方向、reward 后果
```

### 4.3 Non-Redundancy

`M_k` 是否真的新增信息，而不是换一种说法重复 `M_{k-1}`？

```text
0 = 基本只是 rephrase
1 = 有少量新内容，但大部分重复
2 = 主要更新都是新事实或新限制条件
```

### 4.4 Retention / No Harmful Forgetting

`M_k` 是否保留了 `M_{k-1}` 中仍然重要的事实？

```text
0 = 删除或改坏了重要已确认事实
1 = 有轻微丢失或过度简化
2 = 保留旧知识，同时加入新知识
```

这个维度很重要，因为我们已经看到 sig3/sig4 raw 有 memory 变短甚至 lossy 的趋势。我们不希望 reward 鼓励“短但忘东西”的 memory。

## 5. Reward Formula

第一版可以用：

```text
explore_reward =
  0.35 * new_actionable_knowledge
+ 0.25 * evidence_specificity
+ 0.20 * non_redundancy
+ 0.20 * retention
```

再除以 2，归一化到 `[0, 1]`。

也可以加一个 penalty：

```text
if retention == 0:
    explore_reward -= 0.3
```

避免模型通过 aggressive compression 获得假高分。

## 6. Judge Input

第一版 judge 输入只给 memory pair：

```text
Previous memory M_{k-1}:
...

Updated memory M_k:
...
```

不喂完整 trajectory。

理由：我们希望 judge 评估 memory delta，而不是判断行为过程。

可选增强：给 judge 一个机器提取的 diff hint，例如：

```text
length_change: +120 chars
new_potion_mentions: [pink, red]
removed_potion_mentions: [green]
```

但第一版可以不加，先保持简单。

## 7. Judge Prompt Draft

```text
You are judging whether the agent's latest trial led to meaningful exploration progress.

You will be given:
1. The previous memory before one trial.
2. The updated memory after that trial.

Do NOT judge writing style or verbosity.
Do NOT reward a longer memory unless it adds substantive exploration progress.
Do NOT reward restating the previous memory in different words.
Do NOT reward generic advice such as "try more potions" unless it names a concrete hypothesis or target.
Do NOT require the update to be correct with respect to hidden ground truth; judge only whether the updated memory shows useful exploration progress compared with the previous memory.

Reward updates that:
- add new discoveries about potion effects, stone transformations, reward-relevant patterns, or useful strategies;
- correct previous wrong, uncertain, or overconfident beliefs;
- create concrete hypotheses or verification targets for future trials;
- differ from the previous memory in a meaningful, non-redundant way.

Score the update on four dimensions, each from 0 to 2:

1. new_discoveries:
0 = no new discovery
1 = minor or tentative new discovery
2 = clear useful new discovery

2. error_correction:
0 = no correction
1 = clarifies or weakly revises a prior belief
2 = clearly corrects a previous mistake or resolves important uncertainty

3. verification_targets:
0 = no new exploration target
1 = vague or partial hypothesis to test
2 = concrete useful hypothesis or action target for future exploration

4. non_redundant_change:
0 = mostly redundant or cosmetic
1 = some meaningful change
2 = substantially different in a useful way

Return only valid JSON:
{
  "brief_reason": "one short sentence",
  "new_discoveries": int,
  "error_correction": int,
  "verification_targets": int,
  "non_redundant_change": int
}
```

## 8. How This Becomes ACT Reward

这个 reward 虽然由 memory delta 计算，但 credit 应该给 trial `k` 的 ACT sample，因为：

```text
ACT_k produced experience T_k
WRITE converted M_{k-1} + T_k into M_k
M_k - M_{k-1} reflects knowledge gained from ACT_k
```

所以训练时可以：

```text
ACT sample for trial k gets:
task_reward_k + beta * memory_delta_reward_k
```

注意：这不是给 WRITE reward。WRITE reward 仍然可以保留 downstream / transition_acc / norm_improve。这里的 memory delta reward 是把 “产生有用 experience” 这件事反馈给 ACT。

## 9. Important Caveat

memory delta 同时受 ACT 和 WRITE 影响。

如果 ACT 探索得很好，但 WRITE 写坏了，memory-delta reward 会低。这看起来不公平，但在我们的 pipeline 里也合理：最终我们关心的是 experience 是否真的进入 memory。

不过为了诊断清楚，建议同时记录：

```text
memory_delta_reward
unique_transition_count
new_transition_count
write_acc
trial_raw_reward
trial_norm_reward
```

如果 transition count 很高但 memory_delta_reward 很低，说明瓶颈在 WRITE。

如果 transition count 低且 memory_delta_reward 低，说明瓶颈在 ACT exploration。

## 10. Offline Validation Plan

第一步不要直接在线训练，先离线验证。

从已有轨迹里抽取：

```text
(M_{k-1}, M_k)
```

覆盖这些 runs：

- no-memory 不适用，因为没有 memory
- freeform
- sig3down
- sig4raw
- sig4norm
- act-only summary runs if available

统计：

```text
memory_delta/new_discoveries
memory_delta/error_correction
memory_delta/verification_targets
memory_delta/non_redundant_change
memory_delta/explore_score
```

再和这些指标相关：

```text
trial unique transition count
next trial norm score
future norm score
write acc
memory length delta
```

关键验证问题：

```text
memory_delta_reward 是否能区分 freeform verbose update、sig3 lossy update、sig4 compact useful update？
```

如果能区分，它就适合进入 online reward shaping。

## 11. Online Cost Control

Online 全量 judge 太贵：

```text
rollout_batch_size=8
n_samples=8
num_trials=10
=> 640 memory updates / rollout
```

所以如果进入 online，建议：

```text
judge_sample_rate = 0.1
```

或者只 judge：

```text
trial k in {0, 1, 2, 5, 8}
```

并且缓存：

```text
hash(M_{k-1}, M_k)
```

## 12. Relation to Proxy Reward

LLM memory-delta judge 更适合做第一阶段验证。最终我们可能用 proxy reward 替代 API。

可能的 proxy：

```text
new_potion_effect_mentions
new_stone_feature_mentions
new_reward_change_mentions
removed_fact_count
summary_semantic_delta
unique_transition_count
positive_transition_discovery
no_effect_discovery
```

LLM judge 可以帮助我们校准这些 proxy。

## 13. Recommendation

下一步我建议做：

1. 实现 offline memory-delta judge。
2. 对已有 runs 的 `(M_{k-1}, M_k)` 打分。
3. 先不要在线训练。
4. 看它是否能解释：

```text
freeform 很长但不一定高质量
sig3 很短但可能 lossy
sig4raw 更 compact/actionable
sig4norm 是否保留更多 evidence
```

如果这个信号成立，再把它作为 ACT exploration reward 加进训练。

---

## 14. 离线有效性验证（已完成 2026-06-27）

### 14.1 我们做了什么

用 `validate_memory_delta_judge.py` 在已有 eval 轨迹上抽取相邻 memory pair `(M_{k-1}, M_k)`，
用第 7 节的 judge prompt 调 **deepseek-chat** 打分，`explore_score = (new_discoveries +
error_correction + verification_targets + non_redundant_change) / 8 ∈ [0,1]`。

- 先 `--dry-run` 跑一遍估 API cost（只拼 prompt 不调用），再真跑。
- **120 个 pair = 3 个 run × 5 episodes × 8 trials**（`skip_first_pair`，k=1..8，40 pair/run）。
- 三个 run：`act119`（act-only）、`co99`（co-train）、`sig4norm119_rep1`（纵向 norm-improve）。
- judge 只喂 memory pair，不喂 trajectory（第 6 节决定）。
- 结果目录：`logs/act_judge_validation/judge_act119_co99_sig4norm119_first5/`
  （`pairs.jsonl` 每行含 `judge` 字段 + `brief_reason`；`summary.json` 是按 run 的聚合）。

### 14.2 正确的看法：within-cell，不是跨 run 均值

**跨 run 均值是错误视角**：三组 explore_score 均值 ≈ 0.44 / 0.41 / 0.44，看起来"分不开"——
但这没有意义，因为差异本来就在 cell 内部、求均值会被抵消。我们真正要看的是
**同一个 episode、同一个 trial（=GRPO sibling 实际比较的粒度）三组的差异**。

把 40 个 (episode, trial) cell 做方差分解（三组齐全）：

| 指标 | 值 |
|---|---|
| within-cell 方差（同 ep+trial、不同 policy） | **占总方差 53%** |
| between-cell 方差（题目/trial 难度） | 47% |
| within-cell 平均极差 (max−min over 3 runs) | **0.525**（满量程 1.0） |
| 三组完全相同的 cell | 1/40 |
| 各 run 在 cell 内最高的次数 | act 12 / sig4 12 / co99 10 / 平手 6 |

→ 一半以上方差来自"同题同 trial、不同探索"，且没有哪个 run 恒定占优。
**信号在 GRPO 实际用到的粒度上把探索差异拉得开。**

### 14.3 与下游 reward 的相关性 ≈ 0（以及为什么这不是反驳）

per-pair 把 `summaries[k]`=M_k 驱动的 trial k+1 分数 join 回来（来自 `results.jsonl` 的
`agent_per_trial`），相关性全部 |r| < 0.09：

| 目标 | corr(explore_score, ·) |
|---|---|
| 下一 trial raw | −0.008 |
| raw improvement (r_{k+1}−r_k) | +0.011 |
| norm improvement | −0.029 |
| 剩余 episode 平均 raw | −0.088 |
| 剩余 episode 平均 norm | +0.053 |

**这不构成对信号的反驳**，原因有二：
1. 下游 reward 本身就是公认的噪声/差 credit 信号——用它验证 judge 在逻辑上循环
   （若下游 reward 干净就不需要这个 judge）。
2. 与核心假设自洽：如果瓶颈是"ACT 用不好 memory"，那么即使 M_k 真新增了知识，
   也不会转成下一 trial 的 reward——零相关恰恰**符合**"下游使用是瓶颈"的判断。
   所以下游 reward 是验证这个 judge 的**错误标尺**；正确标尺是"相对自身 prior 的知识增益"，
   由下面的 case study 直接核对。

### 14.4 Case study（人工逐 cell 核对）

**正向**（act 占优的高极差 cell）：`logs/act_judge_validation/judge_act119_co99_sig4norm119_first5/case_analysis_top4.txt`
- 高分对应**真新发现/真纠错**：ep9 t4 act 新增 `Orange: blue large pointy→blue large round +1`；
  ep9 t7 act 把"+15 来自 red"**改正为 green**（本轮实测支撑）。
- co99 的 +900~+1200 字符全是"New Evidence"段、复述已知 → 0 分。**judge 没被长度骗。**

**反向**（act 最低 / 三组全低）：`logs/act_judge_validation/judge_act119_co99_sig4norm119_first5/reverse_cases.txt`
- act=0 恰好对应：M_k 逐字未变（ep3 t4）、复述已知路径（ep6 t8）、对自身已知效果再确认（ep8 t7）。
- **最强证据 ep8 t7**：同一条 `Yellow: purple large pointy→purple large round +1`，对 act 是冗余(0)、
  对 sig4 是新发现(0.625)——judge 衡量的是**相对各自 prior memory 的 novelty**，
  不是绝对事实、更不是认 run 身份。这正是当 ACT-credit 用时要的性质。
- 全低 cell（ep1 t8 / ep3 t8，都在 episode 末尾）：三组都只在复述已知 +15 路径，探索枯竭，全 0 正确。

### 14.5 稳定性 & 小瑕疵

- **test-retest 稳定**：同配置重跑两次，run 均值漂移 < 0.03（act 0.475→0.444、co99 0.422→0.409、
  sig4 0.450→0.438）。要更稳可加 2–3 vote。
- **小瑕疵**：judge 偶尔把"stone index / 总分记账订正"误判成 `error_correction` 给分
  （ep3 t4 sig4 拿了 0.375）。量级小，可在 prompt 里加一句"不奖励纯记账/索引修正"或靠 multi-vote 压掉。

### 14.6 三个设计顾虑及结论

| 顾虑 | 结论 |
|---|---|
| 没有 retention 维度，会鼓励遗忘 | 不该由这个信号管：遗忘是 WRITE 的失误，会被 WRITE stream 与 raw task reward 两头惩罚；硬塞进来反而耦合"探索"与"保真" |
| ACT/WRITE 混在一起 | 线上 sibling **共用同一 WRITE policy**，是共享 baseline，GRPO group whitening 一减即抵消；离线用三个不同 checkpoint（writer 不同）是更难设定，信号仍成立，线上只会更干净 |
| 假 discovery 也会拿高分 | 假发现也是 experience，被它误导的决策会被 task reward 压回；唯一约束是 **β 的相对量级**（explore 项幅度应明显小于 task 项，避免纠偏前先带偏） |

**结论：信号通过正/反向离线验证，可以进入在线训练 pipeline。**

---

## 15. 接入训练 pipeline 的方案（下一步，非侵入式）

### 15.1 设计原则

- **加法 + 开关**：`act_reward_k = task_reward_k + β · explore_reward_k`，β 默认 0 = 完全保持现行为。
  开关走 env/config（如 `ALCHEMY_ACT_EXPLORE_BETA`、`act_explore_reward_beta`），rollout 里 `getattr` 读一次。
- **复用现有机制**：explore_reward 加在 **ACT sample** 上（不是 WRITE）；ACT 已按
  `(group_index, trial_pos)` 分组 whitening，shared baseline 自动抵消，advantage/metrics hook 无需改。
- **每 (episode, trial) 一次 judge**：trial k 的所有 ACT step 共享同一个 `explore_reward_k`
  （M_{k-1}→M_k 是 trial-level 的量），不是每个 ACT sample 各调一次。

### 15.2 高效调用 judge（成本控制）

朴素成本：`rollout_batch(8) × n_samples(8) × (num_trials−1)≈9 ≈ 576 次 judge / rollout`，太贵。可叠加：

1. **本地 sglang judge（首选，零 API 成本）**：rollout 已经在用本地 sglang url 做 `_score_memory_accuracy`
   的异步生成。judge 完全可以走同一条路——对本地 endpoint 异步批量请求、greedy、解析 JSON，
   无外部 API、无 rate limit、可 `asyncio.gather`。代价是 judge 模型可能弱于 deepseek（可单独 serve 一个更强的 judge 模型）。
2. **dedup / cache**：按 `hash(M_{k-1}, M_k)` 缓存——episode 早期 siblings 常产出相同 memory，可大幅去重。
3. **sub-sample trials**：只 judge 一部分 trial（如 config 指定 `{0,1,2,5,8}` 或按比例），其余 explore_reward=0。
4. **multi-vote 仅在需要时**：默认单次；要降噪再开 2–3 vote。

> 若坚持用 deepseek API：必须叠加 cache + sub-sample + 限并发，并先用 dry-run 估每 rollout 成本。
> 推荐先用**本地 sglang judge** 跑通 + 小规模 smoke，再决定是否换 API 提质。

### 15.3 验证步骤（沿用既有规范）

1. smoke（`NUM_ROLLOUT` 小 + `DRYFAST`）：确认 explore_reward 正确加到 ACT、β=0 时与现行为逐位一致、
   原有监控指标全绿。
2. 看 ACT advantage 是否因 explore 项产生合理 within-group spread；judge 调用数 / cache 命中率 / 耗时。
3. 全集对照：`act-only`（β=0）vs `act+explore`（β>0），同 window、同 ckpt 冷启。
