# Memory 演化对比:act-only vs 双训(co-train)

对比两组实验中,模型写出的 memory(summary)随训练的前后变化。**先原文,后分析。**

## Setting(先讲清楚)

- **act-only**:`TRAIN_ACT=1, TRAIN_WRITE=0`。memory 被 ACT 读来决策,但**写 memory 这件事没有任何 reward 信号** → memory 只是 ACT 的副产品。
  - run: `qwen3-4b-curr950-actonly-r120-e10-20260620-014759`,traj 范围 **rollout 0~119(全程)**
- **双训(co-train)**:`TRAIN_ACT=1, TRAIN_WRITE=1`。memory 写入被 **WRITE reward** 直接优化(reward = memory 对未来 `(stone,potion)→(result,reward)` 转移的预测准确度)。
  - run: `qwen3-4b-curr950-r120-e10-20260618-174648`,traj 范围 **rollout 80~112**(早期 0-79 traj 未保留)
- **数据**:每个 episode 的 `summaries[-1]`(最后一个 trial 后写的最终 memory)。每组在早/中/晚 rollout 各取样。
- **指标**:char(字符数)与 token(Qwen3-4B tokenizer),char/token ≈ 4.4~4.7。

## 1. 长度演化

| | 早期 | 中期 | 晚期 |
|---|---|---|---|
| **act-only** | r5: 1769 char / **384 tok** | r60: 2723 / 583 | r115: 2591 char / **551 tok** |
| **双训** | r80: 4583 char / **1032 tok** | r100: 5547 / 1304 | r112: 5454 char / **1251 tok** |

- act-only:从 ~1800 缓慢长到 ~2600 char 就稳了(~550 token)
- 双训:稳定在 ~5500 char(~1250 token),**是 act-only 的 ~2.3 倍**

---

## 2. 完整 memory 原文

### 2.1 ACT-ONLY 早期 rollout_5 (episode_index=288)
`per_trial_scores=[1, 2, 16, 2, 2, 1, 15, 1, 2, 3]` — memory 1759 char

```
### Potion Effects
- Orange: Changes blue small pointy → blue small round; increases reward from -1 to +1. Confirmed effect: increases reward for small pointy stones. Has no effect on large pointy stones (e.g., stone 2 → blue large pointy with reward -3, unchanged).
- Green: No observed effect. Tested on blue small pointy stone (stone 1), reward remained at -1. No effect on other stones or features.
- Turquoise: No observed effect. Tested on purple small pointy stone (stone 2), reward remained at +1. No change in stone feature or reward.
- Yellow: No observed effect. Tested on purple small pointy stone (stone 2), reward remained at +1. No change in stone feature or reward.
- Pink: No observed effect. Tested on purple small pointy stone (stone 2), reward remained at +1. No change in stone feature or reward.

### Highest Reward Combination
- Blue small round with reward +1: Achieved by transforming blue small pointy stone using orange potion. This is the highest reward observed so far. ... Current evidence shows no higher reward possible.
- Purple small round with reward +15: Observed as a standalone high-reward stone with no transformation or potion effect affecting it. ... represents a non-interactive, intrinsic high-value state.
```

### 2.2 ACT-ONLY 晚期 rollout_115 (episode_index=249)
`per_trial_scores=[16, 30, 2, 15, 2, 16, 2, 2, 3, 15]` — memory 2591 char

```
### Potion Effects
- Turquoise: Increases reward significantly for small round stones when transformed into small pointy stones. This transformation yields a reward jump from +1 to +15. Effective only on purple small round → purple small pointy. No effect on large pointy stones or blue small pointy stones. Also produces a positive reward increase (from -1 to +1) when applied to purple large round → purple large pointy...
- Red: When applied to purple large pointy stones, causes a color change to blue and a reward drop from +1 to -1. This indicates red is not neutral and has unintended side effects...
- Pink: Causes a negative reward change when applied to blue small pointy stones (transformed to blue small round, reward drops from +1 to -1). Harmful in this configuration...
- Orange: No effect on reward in tested cases. ... Neutral overall.
- Yellow: No effect on reward... Neutral across stone types tested.
- Green: When applied to blue large pointy stones, transforms them into purple large pointy stones and increases reward from -1 to +1. ... Previously thought neutral, but this trial proves it is effective in transforming blue to purple and improving reward value.

### Highest Reward Combination
Purple small pointy stone with reward +15, achieved by transforming a purple small round stone into purple small pointy using turquoise. ... No higher reward was achieved in this episode...
```

### 2.3 双训 早期 rollout_80 (episode_index=681)
`per_trial_scores=[2, 3, 2, 3, 16, 3, 16, -3, 2, 3]` — memory 4556 char

```
### Potion Effects
- Green:
  - When applied to purple small round (stone 2, reward -3), produces blue small round with reward -1 (a decrease in magnitude, i.e., less negative).
  - When applied to blue small round (stone 1 or 2, reward -1), produces blue small pointy with reward +1 (a positive transformation from negative to positive), regardless of stone index.
  - Has no effect when applied to stone 0 (purple large pointy, +1) or stone 1 (purple large round, -1).
- Yellow: ... No observed effect on any stone's reward or form.
- Turquoise:
  - When applied to purple small round (stone 2, reward -3), transforms it into blue small round with reward -1 (a net improvement, less negative).
  - When applied to purple large round (stone 0, reward -1), transforms it into blue large round with reward +1 (positive transformation...).
  - When applied to purple small pointy (stone 0, reward -1), transforms it into blue small pointy with reward +1 (... new finding).
  - When applied to blue small round (stone 2, reward -1), results in no change...
- Pink:
  - When applied to blue small round (stone 2, reward -1), transforms it into purple small round with reward -3 (a degradation...).
  - When applied to blue small pointy (stone 1, reward +1), transforms it into purple small pointy (reward -1) (new finding: enables creation of negative reward stones from positive ones).
- Orange: No observed effect on any stone's form or reward (tested in multiple instances...).
- Red: No observed effect on any stone's form or reward (tested in multiple instances...).

### Highest Reward Combination
- The highest reward achieved so far is +15, achieved by placing stone 0 (blue large pointy, reward +15) in the cauldron.
- ... New finding: Pink transforms blue small pointy (+1) into purple small pointy (-1)...
- New confirmation: Turquoise can transform purple small pointy (reward -1) into blue small pointy (reward +1), confirmed in this trial.
- New observation: Pink transforms blue small round (reward -1) into purple small round (reward -3)...
- No transformation path exists to convert blue small round to blue small pointy using available potions in this trial — green is not available...
```

### 2.4 双训 晚期 rollout_112 (episode_index=72)
`per_trial_scores=[30, 30, 15, 30, 30, 0, 15, 30, -3, 13]` — memory 5459 char

```
### Potion Effects
- Turquoise (potion 1/2/8): Transforms purple small pointy → blue small pointy; increases reward from +1 to +15. Strong positive effect; confirmed capable of producing a high reward. (Tested on stones 1 and 2; effective in all cases observed.) No effect on purple large round (→ remains purple large round, reward -3), purple small round, or blue stones...
- Green (potion 2/5/9/10): Transforms blue large pointy → blue small pointy; increases reward from +1 to +15. ... (Tested on stone 2; confirmed transformation and +15 reward.) No effect on blue small round...
- Yellow (potion 0/3/7/11): No observed transformation effect on any stone...
- Orange (potion 4/8/11): No observable effect...
- Pink (potion 0/5/8/9): Transforms blue large round → purple large round (reward +1 → -3); negative effect... Also transforms blue small round → purple small round (reward +1 → -1)...
- Red (potion 3/4/6/9/10): No observable effect on any stone...

### Highest Reward Combination
+15 reward achieved via:
- Blue small pointy stone (originally from purple small pointy → transformed by turquoise)
- Blue small pointy stone (originally from blue large pointy → transformed by green)
...
- Total score reached 30 via placement of two +15 stones in the cauldron.
- ✅ Confirmed: Green and turquoise are the only effective transformation potions.
- ❌ No evidence of transformation from blue small round, purple large pointy, or blue large round to any pointy or small shape.
- ✅ Purple small pointy and blue large pointy remain the only viable starting stones for achieving +15.
- ❌ All other stones ... remain at +1 or -1 and are unresponsive to all available potions.
```

---

## 3. 分析

### 3.1 ACT-ONLY 的前后变化(r5 → r115)
- **早期病征——"测一次就判死刑"**:r5 里 5 个 potion 有 4 个被判 `No observed effect`,而且每个只 `Tested on` **一颗石头**就下结论 → 探索不足 + 过早武断。
- **晚期出现自我修正**:r115 里 Green 写道 *"Previously thought neutral, but this trial proves it is effective"*,覆盖到 6 个 potion。
- **但本质没变**:仍是自然语言主观判断("strong positive effect"、"Harmful in this configuration"),长度只从 1759→2591 char。memory 没奖励 → 不进化出更可用的结构。

### 3.2 双训的前后变化(r80 → r112)
- **早期(r80)就已是精确转移记录 + 累积式发现**:分 stone index 记 `(状态)→(potion)→(新状态, reward)`,用 `New finding / New confirmation / New observation` 标注知识累积,慎下"无效"死结论。
- **晚期(r112)进化成"决策表"**:potion 标了编号(potion 1/2/8)、用 ✅/❌ 做决策性总结("✅ Confirmed: Green and turquoise are the only effective"、"❌ No evidence of...")、分层更新("Updated from current trial")。

### 3.3 核心差异

| 维度 | act-only | 双训 |
|---|---|---|
| 下结论方式 | "No observed effect"(测 1 次就武断)| "New finding/confirmation"(累积精化,慎下死结论)|
| 记录形式 | 主观描述("strong"/"harmful")| 精确转移 `状态→potion→状态+reward` |
| 长度 | 1759→2591 char(~550 tok,小增)| 4556→5459 char(~1250 tok,2x+)|
| 决策辅助 | 无 | ✅/❌ 总结 + potion 编号(可直接查表决策)|

### 3.4 根因
双训的 **WRITE reward 直接考"memory 能否预测未来转移"**,逼出了精确转移表 + 累积式发现;act-only 的 memory 没有任何奖励信号,停在"测一次就主观判断"的水平,前后基本不进化(只是堆更多观察)。

## 4. 诚实的限制
1. **早晚是不同 episode**(shuffle=0 + curriculum,r5=ep288、r115=ep249、r80=ep681、r112=ep72),不是同一任务的学习轨迹 → 看的是 **memory 质量/风格的整体演化**,不是同题进步。
2. **双训没有 0-79 的 traj**,r80 已是中后期 → 看不到双训从 act-only 那种水平起步的过程,"早期"其实是中期。
3. 长度/得分跨 episode 不可直接比绝对值(curriculum 难度不同);per_trial_scores 仅作参考。
