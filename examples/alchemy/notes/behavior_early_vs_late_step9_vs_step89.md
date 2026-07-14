# EARLY(step 9)vs LATE(step 89)行为对比分析

**Run**: `qwen3-4b-curr950-sig4norm-w3-expl03-budgetv3-r120-e10-20260630b`(co-train ACT+WRITE,explore β0.3,budget+v3)
**Eval 集**: hard20(20 个 episode,每 episode 10 个 trial,共 200 个 trial;两个 checkpoint 用完全相同的 episode)
**数据源**:
- EARLY(step 9):`.../traj/eval/hard20/rollout_9/`(20 个 JSON)
- LATE (step 89):`.../traj/eval/hard20/rollout_89/`(20 个 JSON)

## 环境规则(读数前先明确 setting)

Symbolic Alchemy:每个 **trial** 最多 20 个 turn。agent 把 potion 作用于石头(每个 potion 会秘密抬高或压低某类石头的 reward),trial 得分 = 存入 cauldron 的石头的 reward 之和。10 个 trial 共享同一套隐藏规则:早期 trial 应探索学规则,晚期应利用规则拿高分。

关键机制:
- 得分只来自 `put stoneX into cauldron`(把石头当前 reward 存入)。
- potion 是**一次性**的,用过即消耗。
- 石头一旦存入 cauldron 就从盘面消失,不能再操作。
- 无效动作(操作已存入的石头、复用已消耗的 potion、引用不存在的石头)**不扣已经拿到的分**,只是白白浪费一个 turn。

---

## 一、量化行为对照表(20 ep / 200 trial 的均值)

| 指标 | STEP 9 | STEP 89 |
|---|---|---|
| **平均 episode 得分** | 77.6 | **120.5**(+55%) |
| 每 trial 平均分,trial0 → trial9 | 6.4 → 9.2 | 10.2 → 13.2 |
| 每 trial 用的 turn 数,早期(trial 0–2) | 7.18 | **12.97** |
| 每 trial 用的 turn 数,晚期(trial 7–9) | 4.57 | **10.57** |
| 每 trial 试的不同 potion 数,早期 | 4.43 | **8.47** |
| 每 trial 试的不同 potion 数,晚期 | 1.95 | **6.07** |
| 撞 20-turn 上限的 trial 比例,早 / 晚 | 0% / 0% | 3.3% / 5.0% |
| **无效动作比例** | 1.4% | **12.5%**(9×) |
| 同一 trial 内完全重复的(石头,potion) | 0 | **64** |
| 退化循环(单 trial 内 ≥5 连续无效) | 0/200 | **3/200** |
| 每 episode 有效存石(bank)次数 | 16.0 | **21.2** |
| 推理中引用记忆的比例,早 / 晚 | 0.33 / 0.44 | 0.39 / 0.39 |
| WRITE 侧 summary 长度(字符,早期) | 564 | **1589** |

无效动作分类(总数):

| 无效类型 | STEP 9 | STEP 89 |
|---|---|---|
| 复用已消耗 potion | 13 | **128** |
| 对已存入石头再操作 | 3 | **135** |
| 空 / 幻觉动作(引用不存在的石头) | 0 | **18** |

---

## 二、策略变化要点 + 逐条例子作证

### 要点 1:晚期(step 89)在每个阶段都探索得更猛,但并没有更快转入 exploit

step 89 每 trial 用的 turn 和试的 potion 数几乎是 step 9 的两倍;即使到 trial 7–9,它仍要烧 ~10.6 个 turn / ~6 个 potion,而 step 9 只用 4.6 个 turn / 2 个 potion。经典的 explore→exploit 收敛曲线在 **step 9 反而更干净**(7.18 → 4.57 turn);step 89 到晚期还在做实验。+55% 得分来自**更彻底的实验 + 把改良过的石头都存起来**,而不是更快 exploit。

**作证例子**(同一个 episode `ep_0_episode_1.json` 的 trial 0 早期 vs trial 9 晚期):

STEP 9,trial 9(仅 **4 步**,教科书式 exploit):
```
put stone1 into potion1
put stone0 into potion2
put stone2 into cauldron   [+15]
end the trial
```
→ 试 2 个 potion,确认好石头后立刻存 +15,结束。

STEP 89,trial 9(**12 步**,已经拿到 +15 却继续过度实验):
```
put stone2 into cauldron   [+15]   ← 一上来就把已知好石头存了
put stone0 into potion5
put stone0 into potion1
put stone2 into cauldron   [INVALID]   ← 对已存石头再操作
put stone0 into potion2
put stone1 into potion7
put stone1 into potion3
put stone0 into potion0
put stone1 into cauldron   [+1]
put stone0 into potion4
put stone0 into cauldron   [-1]    ← 过度折腾反而存进一个 -1
end the trial
```
→ 已经拿到 +15 后不收手,又折腾 8 个 potion,最后甚至把一个石头搞成 -1 还存了进去。这直接说明 step 89 晚期没有"见好就收",而是继续探索/实验。

同 episode 的 `per_trial_scores`:
- STEP 9:`[1, 15, 1, 2, 0, -1, 15, 15, 15, 15]`
- STEP 89:`[1, 15, 30, 2, 15, -2, 15, 15, 16, 15]`

step 89 靠 trial 2 的 30、trial 4 的 15 等把整体拉高(存多颗改良石头),但也出现 trial 5 的 -2(过度实验的代价)。

---

### 要点 2:得分上去了,但动作合法性反而下降(奖励没有惩罚无效动作)

无效动作从 1.4% 涨到 12.5%(9×)。这些多出来的无效动作是**格式合法但操作非法**:对已存入的石头再用 potion、复用已消耗的 potion(64 次 trial 内完全重复)。因为无效动作不扣已拿到的分,RL 策略学会了容忍它们——刷分优先,合法性靠后。

**作证例子 A(对已存入石头再操作)** `rollout_89/ep_1_episode_3.json` trial 1:
```
put stone1 into potion4
put stone2 into potion1
put stone2 into cauldron            ← stone2 已存入,离开盘面
put stone1 into potion5
put stone1 into cauldron            ← stone1 已存入
put stone2 into potion2   [INVALID] ← 又想操作已经不在盘面的 stone2
put stone0 into cauldron
put stone1 into potion7   [INVALID] ← 又想操作已经不在盘面的 stone1
end the trial
```

**作证例子 B(量级对比)**:同类无效在 step 9 几乎不发生:
- 复用已消耗 potion:step 9 = 13 → step 89 = **128**
- 对已存入石头再操作:step 9 = 3 → step 89 = **135**

step 9 全程 200 trial 里"对已存石头再操作"仅 3 次(如 `ep_10_episode_29.json` trial 0 的一次 `put stone2 into cauldron` 重复),而 step 89 高达 135 次——这是新学出来的坏习惯,不是随机噪声。

---

### 要点 3:出现了 step 9 完全没有的退化模式(reasoning 崩溃 + 幻觉石头 + 烧到上限)

在 3/200 个 trial 里,存完所有石头后模型**不发 `end the trial`**,而是不断输出没有 REASONING、纯裸命令的动作,并**幻觉出根本不存在的石头编号**(盘面只有 0–2 号,它却引用 5–9 号),一直循环到 20-turn 上限。共 18 个这类空/幻觉动作,全部出现在 step 89。

**作证例子** `rollout_89/ep_14_episode_51.json` trial 8(20 步里 13 步无效,撞满 20-turn 上限):
```
st0  put stone1 into potion4
st1  put stone1 into potion3
st2  put stone1 into cauldron   [+15]
st3  put stone2 into cauldron   [+15]
st4  put stone1 into potion1    [INVALID]
st5  put stone0 into potion1
st6  (空动作)                    [INVALID]
st7  put stone0 into potion9
st8  put stone0 into cauldron   [+15]   ← 3 颗石头全存完,盘面已空
st9  (空动作)                    [INVALID]
st10 (空动作)                    [INVALID]
...
st19 (空动作)                    [INVALID]   ← 一路空动作烧到 20-turn 上限
```

第 12 步的完整 `raw_act`(全长仅 43 字符,没有 OBSERVATION、没有 REASONING,直接幻觉 stone 5):
```
ACTION: Place stone 5 in potion 4<|im_end|>
```

对比 step 9:全程 200 trial **0 次**这种 ≥5 连续无效的退化循环,且**从不撞 20-turn 上限**。

---

### 要点 4:WRITE / 记忆侧显著成熟(内容更长、更结构化、能推出多步变换链)

step 89 的 summary 比 step 9 长(早期 1589 vs 564 字符),而且从"单步 potion 效果"进化到能推理**多步变换链**。两侧在晚期 trial 的 ACT 推理都会引用记忆(都能引用 "+15" 规则),所以记忆的*使用*水平相近,但记忆的*内容质量*在 step 89 明显更高。

**作证例子**(同 episode `ep_0_episode_1.json`,trial 4 之后的 summary):

STEP 9 summary(单步效果罗列):
> "**Pink**: Increases reward when applied to blue large round stones (from -1 to +1); also dramatically on blue large pointy stones (from +1 to +15)... **Highest Reward Combination**: Blue large pointy stone transformed by pink → blue small pointy with reward +15."

STEP 89 summary(推出**两步组合链** + 稳定性判断):
> "The transformation path (**purple → blue via orange, then +15 via pink**) is now confirmed and can be reliably replicated... This outcome is stable and repeatable, regardless of whether the stone is initially small or large."

step 89 把 "orange 先把 purple 变 blue,再用 pink 拿 +15" 这条两步链推理并固化进了记忆,这是 step 9 没有的。

**作证例子(ACT 推理引用记忆)** step 89 `ep_0_episode_1.json` trial 9 开局:
> "I have identified that the highest reward (+15) is achieved by applying pink to a blue large pointy stone... applying it to a blue large round stone yields a reward increase to +1, **as noted previously**."

对比 step 9 早期 trial 0 的泛泛探索(无记忆可用):
> "So far, every potion tested has had no effect on the stones. This suggests that either the effects are rare or require specific combinations... I'll try stone 2 (purple large pointy) in potion 7 (red)."

---

### 要点 5:两侧正常情况下都会提前结束 trial(不故意烧满 20 turn)

step 9 撞 20-turn 上限的比例为 **0%**(早/晚都是 0)。step 89 早 3.3% / 晚 5.0%,而且**唯一撞上限的情形就是要点 3 的退化循环**(存完石头后停不下来),不是有意的 20-turn 深度探索。也就是说,step 89 撞上限不是"更勤奋地探索",而是"该结束时结束不了"的病态。

---

## 三、最需要注意的两点(结论)

1. **+55% 的得分是真的,靠的是更多实验 + 激进地把整套改良石头全存进去**(21 vs 16 次 bank/ep),而**不是**靠更清晰的 explore→exploit 切换——step 89 晚期探索得比 step 9 早期还猛。

2. **出现了新的"过度实验 / 停不下来"病态**:无效率 9×、trial 内 potion 复用、以及一旦没有合法动作就幻觉出不存在的石头、reasoning 崩溃、烧 turn 到上限的退化循环。根因是**无效动作不扣已存的分,奖励信号没有惩罚它们**——策略学到了刷分,代价是牺牲动作合法性和"该结束就结束"的纪律。若要修,建议在 reward 里对无效动作 / 幻觉动作 / 撞上限加轻微惩罚,或在存完石头后引导及时 `end the trial`。

---

## 附:复现方式

所有数字由 Python 直接遍历上述两个 rollout 目录的 20+20 个 JSON 得到(字段:`turns[*].{trial,step,action,valid,reward,new_trial,raw_act}`、`per_trial_scores`、`summaries`)。关键判定逻辑:
- turn/trial、distinct potion:按 `trial` 分组统计。
- 撞上限:trial 内 `max(step) >= 19`。
- 无效分类:维护每 trial 的 `banked` 石头集合与 `usedpot` potion 集合,对 invalid 动作判定属于"复用 potion / 操作已存石头 / 空幻觉"。
- 退化循环:trial 内最长连续 invalid 段 ≥ 5。
