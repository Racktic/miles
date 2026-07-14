# 案例研究 1 — sig4norm β0.3 训练演化(step 9 → step 89)

**Run**: `qwen3-4b-curr950-sig4norm-w3-expl03-budgetv3-r120-e10-20260630b`
(Qwen3-4B-Instruct-2507 · **co-train ACT+WRITE**(WRITE 信号 downstream_norm_improve) · explore reward β=0.3 · budget 提醒 + explore_v3 prompt · memory window 3)
**对比**: 训练早期 checkpoint **rollout_9(step 9)** vs 晚期 **rollout_89(step 89)**,eval on hard20(20 个 episode,每 episode 10 trial,两侧同一批 episode)
**数据**: `.../traj/eval/hard20/rollout_9/` 与 `.../rollout_89/`(各 20 个 JSON)
**方法**: 3 个 read-only 分析 agent(统计 / 写入记忆 / 行为)+ 逐字例子。

> 环境规则(读数前先明确):每 trial 最多 20 turn;agent 把 potion 作用于石头(每个 potion 秘密抬高或压低某类石头 reward),得分只来自 `put stone into cauldron`(存入,拿该石头当前 reward);potion **一次性**、石头存入即消失;**无效动作(操作已存石头 / 复用已耗 potion / 引用不存在石头)不扣已挣的分**,只白费一个 turn。10 个 trial 共享同一隐藏规则:早探索、晚利用。

---

## TL;DR

训练把 eval 分数拉高 **+55%(episode 77.6 → 120.5)**,三个角度一致地指向同一机制:**探索大幅加强 + 写入记忆(被 RL 训练)真的变好**。但代价是晚期涌现了 **"过度实验 / 停不下来"的病态**(无效动作 9×、幻觉不存在的石头、循环烧到 turn 上限)——因为无效动作不扣分,reward 信号不惩罚它。

| 维度 | step 9 | step 89 | 变化 |
|---|---:|---:|---|
| episode 得分 | 77.6 | **120.5** | **+55%**,paired t=3.44 显著,10 个 trial 全涨 |
| num_turns / episode | 55.8 | **112.7** | 翻倍(t=16.5) |
| distinct potions / trial | 2.9 | **6.9** | +138% |
| within-ep 学习(last5−first5) | +3.53 | +3.08 | **没变陡** |
| 无效动作率 | 1.4% | **12.5%** | 9× |
| memory 多步转换链占比 | 0.45 | **0.65** | +44% |
| memory "no-effect" 空记录 / 条 | 6.42 | 4.53 | −30% |
| 每 episode bank 次数 | 16.0 | **21.2** | +33% |

---

## 一、统计(stats)

per-trial 得分(idx 0→9):
- EARLY: `6.4, 4.5, 6.1, 6.7, 6.4, 9.7, 10.5, 11.5, 6.9, 9.2`
- LATE : `10.2, 7.7, 12.6, 11.6, 10.6, 12.7, 14.3, 15.6, 12.3, 13.2`(**每个 trial idx 上 LATE ≥ EARLY**)

turns per trial(idx 0→9):
- EARLY: `9.9, 6.1, 5.6, 5.5, 5.4, 4.9, 4.7, 4.6, 4.5, 4.6`(前重后轻,很快 coast)
- LATE : `14.3, 13.0, 11.7, 10.6, 11.1, 10.5, 10.0, 9.3, 11.9, 10.6`(**每个 trial 都在重探索**)

- 得分 +4.30/trial,sd=5.59,**paired t=3.44(n=20)→ 显著**;turns 增长 **t=16.5 → 极强**。
- **within-episode 学习斜率没变陡**(+3.53→+3.08):是整条轨迹**整体抬升**(基线能力更强),不是 in-context 适应更快。
- **cash-in 质量也升**(cauldron reward/action 4.83→5.68):探索更多、但存进去的石头更好。

**结论**:训练让模型**普遍更强 + 更爱探**,分数在每个 trial 位置都涨;增益来自"更彻底实验 + 把改良后的石头都存起来",不是更快 exploit。

---

## 二、写入记忆(memory,WRITE 流被 RL 训练)

| 指标(180 条 summary/侧) | step 9 | step 89 |
|---|---:|---:|
| 平均字符 / 条 | 1613 | 1820(+13%) |
| hedging(unknown/unclear)/ 条 | 0.09 | **0.21**(更诚实校准) |
| "no-effect/unchanged" 声称 / 条 | 6.42 | 4.53(−30%) |
| 含多步转换链(A→B→C)的最终 memory 占比 | 0.45 | **0.65** |
| 归入真实 potion 的效果数(最终 memory) | 1.55 | 1.70 |

**发现:memory 从"泛泛单-potion + 一堆 no-effect"演化成"编号列出通往 +15 的多步转换路线",而且是真进步、无 gaming**——长度只 +13%(没灌水)、hedging 反而升(更诚实)、reward-delta 声称更密而非虚高。memory 变好 → 下游 ACT 动作变好 → reward +55%。

**逐字例子(同一题 ep_idx 5):**

EARLY(step 9)—— 断言 +15 不可复现:
> "+15 reward … Achieved only in the initial conditions before any potion interaction. **No potion increases, preserves, or reproduces this value** … No combination yields higher than +1."
> (该 episode 总分 7.0)

LATE(step 89)—— 找到**三条**重构 +15 的路线:
> "Three viable routes have been confirmed: 1. Purple small pointy → blue small pointy (via yellow) → blue small round (via turquoise) 2. Blue large pointy → blue small pointy (via red) → blue small round (via turquoise) 3. Purple small round → blue small round (via yellow)."
> (同一 episode,总分跳到 ~24)

LATE 还会记录**约束条件**(ep_idx 0):
> "The transformation path (purple → blue via orange, then +15 via pink) is now confirmed and can be reliably replicated … In current trials where **pink is unavailable, the highest possible reward remains unattainable**."

---

## 三、行为(behavior)

| 指标 | step 9 | step 89 |
|---|---:|---:|
| turns/trial 早期(0-2) | 7.18 | **12.97** |
| turns/trial 晚期(7-9) | 4.57 | **10.57** |
| distinct potions/trial 早期 | 4.43 | 8.47 |
| distinct potions/trial 晚期 | 1.95 | **6.07** |
| 无效动作率 | 1.4% | **12.5%(9×)** |
| 同 trial 内完全重复(石头,potion) | 0 | **64** |
| 退化循环(单 trial ≥5 连续无效) | 0/200 | **3/200** |

**要点:**
- **晚期探索更猛,但没有更快 exploit**:step 89 到 trial 7-9 仍烧 ~10.6 turn / ~6 potion,而 step 9 只用 4.6 turn / 2 potion。经典 explore→exploit 收敛在 step 9 反而更干净。
- **得分升、动作合法性降**:多出来的无效动作是"格式合法但操作非法"——对已存石头再操作、复用已耗 potion(64 次)。因为不扣分,策略容忍了。
- **⚠️ 新退化模式(step 9 没有)**:3/200 trial 里,存完所有石头后不 end,反而吐无 reasoning 的裸命令、**幻觉不存在的 stone 5-9**,循环到 20-turn 上限。

**逐字例子:**

EARLY 泛泛探索(step 9, trial 0):
> "So far, every potion tested has had no effect on the stones … I'll try stone 2 (purple large pointy, reward -1) in potion 7 (red)"

LATE 记忆驱动、有针对性(step 89, trial 9 开局——先存再基于记忆推理):
> "I have identified that the highest reward (+15) is achieved by applying pink to a blue large pointy stone … The pink potion is available in slots 1, 2, 3, 4 …"

LATE 退化崩溃(step 89, `ep_14_episode_51.json` trial 8 第 12 步——完整 43 字符 raw_act、无 REASONING、幻觉 stone 5、循环到上限):
> `ACTION: Place stone 5 in potion 4<|im_end|>`

---

## 结论

1. **+55% 是真的、显著的、每个 trial 都涨**——训练提升的是整体能力 + 探索彻底度,不是 in-context 学习变快。
2. **WRITE 流的 RL 训练确实让 memory 变好**(多步链、更诚实、无灌水),且因果性地反映在下游 reward 上。
3. **但晚期涌现"停不下来"病态**(无效 9×、幻觉石头、循环烧 turn)——根因是 **invalid 不扣已挣分,reward 不惩罚它**;这条正呼应 β0.3"过度探索"的担忧,且在 co-train 版晚期才显现。

> ⚠️ 注意参照系:本报告只比 **sig4norm 自己的早 vs 晚**,看到的是**绝对进步**。若拿它和 **actonly**(不训练 WRITE)比,memory 会显出相对退化——见案例研究 2。

**路径**: EARLY `.../rollout_9/`,LATE `.../rollout_89/`;最典型退化例子 `rollout_89/ep_14_episode_51.json` trial 8。
