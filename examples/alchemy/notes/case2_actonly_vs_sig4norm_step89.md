# 案例研究 2 — actonly vs sig4norm(都 β0.3+budgetv3,step 89)

> ## ⚠️ 修订(2026-07-03,后续实验推翻了本文的主结论)
>
> 本文写作时只有 step-89 的**单次**训练中评测(单次评测在 hard20 上天然有 ±0.05~0.08 的抖动),当时的主结论"co-train 更差(0.45 vs 0.38)、memory 被训坏拖累分数"**已被两组后续实验推翻**:
>
> **1. memory-swap 2×2(step-99 ckpt,每格独立评测 3 次)** —— 把两个模型的"玩游戏"(ACT)和"写记忆"(WRITE)两个角色拆开交叉组合:
>
> | 分数 | 记忆用 actonly 写 | 记忆用 sig4norm 写 |
> |---|---|---|
> | 动作用 actonly | 0.415 ± 0.052 | 0.444 ± 0.077 |
> | 动作用 sig4norm | 0.408 ± 0.045 | 0.420 ± 0.025 |
>
> 四格全部落在 0.408~0.444,**没有任何一对差异超过噪声**;而且"sig4norm 写的记忆"那一列反而**略高**(+0.02)——"co-train 把 memory 训坏了"不成立。
>
> **2. 终点对比(iter_119,offline 多次评测)**:actonly **0.454±0.034**(6次) vs sig4norm **0.439±0.022**(3次)——**分数打平**;局内进步幅度(norm_improve,即 WRITE 奖励直接优化的目标)sig4norm 方向性更高(+0.213 vs +0.151,~1.8 个标准误,未过显著门槛)。
>
> **回头看,当时错在哪**:step-89 恰好踩在 sig4norm 训练曲线的低谷尾部(它的单次评测 89 步 0.376 → 99 步 0.473),单点+单次评测把噪声当成了真差距。
>
> **本文仍然成立的部分**:①行为差异是真的(sig4norm 探索更多、晚期不收敛)——但其归因后来查清是 **co-train 本身**而非记忆问题(见 case4),且这个行为差异**基本不影响分数**;②记忆文本层面的观察(把石头颜色错当 potion、逐字固化等)在文本上是真的——但 memory-swap 证明这些瑕疵的**净代价约等于零**,"更长更自信但更不准"的记忆照样好用(甚至略好)。
>
> 下文为原始分析,按上述修订阅读。

**对比两个 run(同一 step 89、同一设定,唯一差别 = 是否 RL 训练 WRITE 流):**
- **ACTONLY**: `qwen3-4b-curr950-actonly-w3-expl03-budgetv3-r120-e10-20260630`(train ACT only,**WRITE 写但不训练**)
- **SIG4NORM**: `qwen3-4b-curr950-sig4norm-w3-expl03-budgetv3-r120-e10-20260630b`(**co-train ACT+WRITE**,WRITE 信号 downstream_norm_improve)

其余全一致:Qwen3-4B · explore β=0.3 · budget+explore_v3 · window 3 · summary replace · eval hard20(**同一批 20 个 episode**,按 episode_index 逐题对拍)。
**方法**: 3 个 read-only 分析 agent(统计 / 记忆 / 行为)+ 逐字例子。

---

## TL;DR

**原始分几乎打平(ACT 125.75 vs SIG 120.5,逐题 10-10),但 oracle 归一化后 ACT ≈ 0.45 > SIG ≈ 0.38**——差距来自 sig4norm 在**高天花板难题上崩**(归一化惩罚"把 ceiling 留在桌上")。两个层面解释了 co-train 为什么反而更差:

1. **行为**:sig4norm **晚期不收敛、还在广撒探索**,而且**跟自己的 memory 对着干**(重测被 memory 标记无效的 potion、耗掉造 +15 链所需的单次性 potion);效率更低、无效更多、更少主动结束 trial。
2. **记忆**:co-train 让 memory **更长更自信,但更不准**——出现 reward-gaming 式退化(把石头属性当成 potion、逐字固化错误 bullet、"immune"过度断言),归入的真实 potion 反而**更少**。

高天花板难题上 sig4norm 的崩盘:ep16 ACT **245** vs SIG **101**、ep53 145 vs 56、ep11 146 vs 94、ep8 103 vs 57。

---

## 一、统计(stats)

| 指标(20 ep 均值) | ACTONLY | SIG4NORM |
|---|---:|---:|
| 平均 episode 得分(Σ10 trial) | 125.75 | 120.50 |
| 逐题对拍 | 10 胜 | 10 胜(0 平) |
| trial 均分 早期→晚期 | 9.58 → 14.15 | 10.15 → **13.65** |
| turns/trial 早期→晚期 | 12.23 → **8.10** | 12.97 → **10.57** |
| **distinct potions/trial 早期→晚期** | 8.08 → **4.43** | 8.33 → **6.00** |
| reward / 有效动作 | **1.39** | 1.22 |
| 无效动作率 | 8.8% | **12.5%** |
| 不显式 end、拖到 20-turn cap 的 trial | 1/200 | **5/200** |
| 无法解析的 "None" 动作 | **0** | **18** |

**关键**:ACT 随 trial 推进把 potion 搜索**收窄**(8.08→4.43)去 exploit 已学到的配方;**SIG 到最后一 trial 还在广撒 ~6 个 potion**(晚期 trial9:ACT 4.40 vs SIG 6.55),不 exploit——记忆本该让它停止重探,但 co-train 版没有。归一化分数被这种"不收敛 + 高浪费"拉低。

---

## 二、写入记忆(memory)—— co-train 让 memory 更差,不是更好

| 指标(180 条/侧) | ACTONLY(WRITE 冻结) | SIG4NORM(WRITE co-train) |
|---|---:|---:|
| 平均词数 / 条 | 250 | **292**(+17%) |
| hedging / 条 | 0.71 | 1.36(≈2×) |
| 断言词 / 条 | 1.79 | 2.97 |
| **把石头属性错当成 potion 的 bullet** | **0** | **26** |
| "immune/不受任何 potion 影响" 过度断言(总) | 14 | **48**(≈3.4×) |
| **最终 memory 归入的真实 potion 效果(共6个)** | **4.4** | 3.95 |
| 连续 summary 近逐字重复(>0.9 相似) | 0.50 | 0.11 |

**发现**:co-train 的 WRITE 让 memory **更长、更自信、更多 reward-delta 提及,但归入的真实 potion 反而更少(4.4→3.95)——是灌水不是信号**。标志性退化:
- **因果错位**:SIG 造出 26 个把"结果石头颜色"当成"起作用 potion"的 bullet(ACT=0),例如写 `Blue: increases reward`(其实是 orange 把石头变蓝、reward 才涨)。
- **逐字固化错误**:ep37 那条错误 `Blue:` bullet **连续 6 条 memory(M3–M8)逐字不变**。
- **讨好评估的过度断言**:3.4× 的"该石头 immune/stable"话术,并非 trial 支持。

**逐字例子(同一题 ep37,描述同一观察:purple small round → blue small round,−1→+1):**

ACTONLY(**正确归因给 orange potion**):
> "- Orange: Converts purple small round stones to blue small round; preserves size but changes color; reward remains +1."

SIG4NORM(**错归因给"蓝色"**,且 M3–M8 逐字重复):
> "- Blue: Increases reward when present in small round form (e.g., purple small round → blue small round, -1 → +1); positive effect in round shape; **reward increases with blue color**."

SIG4NORM 的 "immune" 过度断言(ep53,又把颜色当 potion):
> "- Blue: No direct potion effect observed; only stone type with confirmed high-reward configuration … **stable and unaffected by any potion**."

---

## 三、行为(behavior)

| 指标 | ACTONLY | SIG4NORM |
|---|---:|---:|
| reward / 有效动作 | **1.39** | 1.22 |
| 无效 cauldron 尝试(spam) | 54 | **92** |
| 重复相同连续动作 | 23 | 43 |
| 每 turn 推理词数 | 190 | 167 |
| 无法解析 "None" 动作 | 0 | 18 |

**要点:**
- **ACT 晚期 exploit,SIG 晚期还在探**:晚期 ACT 收窄到 4.43 potion / 8.1 turn;SIG 仍散在 6.0 potion / 10.57 turn 却拿到更低的晚期分(13.65 vs 14.15)。
- **SIG 会违背自己的 memory**(核心证据):ep16 memory 明说"red 对 blue small pointy 无效",SIG 却把 red 用在 blue small pointy 上、reasoning 自己都写"这与之前观察矛盾",把造 +15 链所需的**唯一 red potion 耗掉** → 15 分/11 turn;而 ACT 按 memory 搭全链 → 30 分/7 turn。
- **共享模型被带偏**:18 个无法解析动作 + 更短推理(167 vs 190 词)= 优化 memory 目标时把 ACT 策略的输出 crispness 也拖坏了(collateral policy drift)。
- **规则宽松时 SIG 的多探索偶尔能赢**(ep30 trial9:SIG 31 vs ACT 2),这也是逐题 10-10 的来源;但平均 + 归一化后,ACT 的"晚期收敛 + 低浪费"占优,SIG 的灾难性低分题把它拖下去。

**逐字例子(ep16,晚期 trial 8,同一起始局面):**

ACTONLY —— 按记忆搭配方,7 turn → 30 分:
> "The purple large pointy stone was successfully transformed into purple large round via orange … Now, I have a purple large round stone, which is the critical precursor for the red potion transformation. Applying red to a purple large round stone converts it into a purple small round stone wi[th +15]…"

SIG4NORM —— 重测 memory 已标记无效的 potion,浪费后才承认矛盾,11 turn → 15 分:
> "The red potion did not produce a positive transformation … when applied to blue small pointy, which **contradicts earlier observations** … This suggests that red's effectiveness is highly specific to purple large round stones…"
> (它当时手上的 memory 已写明:"Red: 对 large round 高效 … 对 small pointy 无效(尤其 blue small pointy)")

---

## 跨案例的关键洞见:绝对进步 vs 相对退化

两份 case study 看似矛盾,其实**参照系不同**:
- **案例1**(sig4norm 早 vs 晚):memory **变好了**(多步链↑、null↓、无灌水)——**绝对进步**。
- **案例2**(sig4norm vs actonly):sig4norm 的 memory **更 gaming**(因果错位、逐字固化、immune 过度断言),归入真实 potion 更少——**相对退化**。

合起来的结论:

> **RL 训练 WRITE 流(downstream_norm_improve)既让 memory 更精致(链式/结构化),又引入了 reward-gaming 退化(把石头当 potion、逐字固化错误、灌水式自信),并把共享模型的 ACT 策略也带偏。因为这个 WRITE 目标 grounding 不足,writer 学会了"写得 effect-dense、自信"而不是"写对因果归因";再叠加行为上的"晚期不收敛 + 违背自己 memory + 浪费单次性 potion",净效果相对"只训 ACT"是负的——尤其在高天花板难题上崩。**

这正面回答了 **"为什么 co-train WRITE 打不过 ACT-only"**。

---

## 附:改进方向(供参考,非结论)
- WRITE 目标需要更强 grounding(如对因果归因正确性直接给信号,而非只看下游分),抑制"把石头当 potion / immune 灌水"这类 gaming。
- 行为侧:对无效动作 / 不结束 trial 给惩罚,治"停不下来"和"违背 memory 重探"。

**数据路径**: ACTONLY `.../20260630/traj/eval/hard20/rollout_89/`,SIG4NORM `.../20260630b/traj/eval/hard20/rollout_89/`(各 20 个 `ep_*_episode_*.json`,含 `turns[].raw_act`、`summary_in`、`per_trial_scores`、`summaries`)。
