# 案例研究 3 — exploration reward 的 β-ablation(ACT-only · budgetv3 · w3)

**对比两个 run(唯一差别 = ACT exploration reward 的 β):**
- **B03(β=0.3)**: `qwen3-4b-curr950-actonly-w3-expl03-budgetv3-r120-e10-20260630`(已训完,0-119)
- **B0(β=0)**: `qwen3-4b-curr950-actonly-w3-b0-budgetv3-r120-e10-20260701`(分析时训至 ~100,**仍在训**;step100 数据为部分快照)

其余全同:Qwen3-4B · ACT-only(WRITE 不训练)· budget 提醒 + explore_v3 prompt · window 3 · 同 task reward。
explore reward 机制:deepseek judge 给 memory delta(M_{k-1}→M_k)打 new_discoveries/error_correction/verification_targets/non_redundant_change 分,组内标准化后 `act_adv += β·explore_adv`。**注意:新发现只能来自 valid 的新颖动作**——invalid/重测产生不了新观察,在 sibling 排名中垫底。

**出发假说(用户)**:β0.3 给探索方向做了清晰界定;β0 组"过度探索导致失控"(turns 高、invalid 高)。
**裁决:核心成立、形态需修正**——β0 不是"探索过头",而是**"不会停"+"丢状态"**;explore reward 实际扮演**行为正则器**。

---

## 一、训练期演化(匹配步数,64 ep/步)

| step | B03 turns/ep | B0 turns/ep | B03 inv% | B0 inv% | B03 撞cap% | B0 撞cap% | B03 train奖励 | B0 train奖励 |
|---|---|---|---|---|---|---|---|---|
| 0 | 55.0 | 57.4 | 1.8 | 1.6 | 0.0 | 0.0 | 12.05 | 13.00 |
| 20 | 53.3 | 62.0 | 1.3 | 2.4 | 0.0 | 0.0 | 9.21 | 11.57 |
| 30 | 53.1 | **75.2** | 1.6 | **6.6** | 0.2 | 1.9 | 10.55 | 11.12 |
| 40 | 55.9 | 93.3 | 1.8 | 10.9 | 0.0 | 4.4 | 11.51 | 13.41 |
| 60 | 80.0 | **131.5** | 5.1 | **21.2** | 1.4 | **17.0** | 14.20 | 14.80 |
| 80 | 93.9 | 135.3 | 7.1 | 22.0 | 2.0 | 18.3 | 16.89 | 16.73 |
| 90 | 97.1 | 145.1 | 8.3 | 23.3 | 1.6 | 22.2 | 15.35 | 16.16 |

- 0-20 步两组不可区分;**step ~30 分叉**,60-70 加速,之后 B0 停驻在"高膨胀吸引子"(~140 turns/ep、~23% invalid、~20% trial 撞 20-turn 上限)。
- **train reward 全程分不出两组**(B0 甚至略高)——漂移在训练信号里不可见。

**eval(hard20,oracle 归一化,单 rep in-training):**

| ckpt | 9 | 29 | 39 | 49 | 69 | 79 | 89 | 99 |
|---|---|---|---|---|---|---|---|---|
| B03 | 0.307 | 0.333 | 0.353 | 0.404 | 0.415 | 0.419 | 0.448 | **0.495** |
| B0 | 0.312 | 0.362 | 0.254 | 0.276 | 0.295 | 0.321 | 0.302 | 0.350 |

ckpt 69-99 均值:**B03 0.444 vs B0 0.317(+0.13)**。B0 从 step 29 后再无上升趋势——漂移的代价全部体现在泛化上。

---

## 二、B0 多出来的 turns 是什么(step 60,+51.5 turns/ep 的记账)

| 成分 | 贡献 |
|---|---|
| **invalid 动作** | **+23.9** |
| 重复 (stone,potion) 的 valid 重测 | +17.0 |
| 真正新探测(新 pair) | 仅 +7.3 |
| 多余入锅等 | +4.5 |

**invalid 解剖**(B0@60,1786 个):对**已入锅石头**继续操作 64% > 重用已耗 potion 28% > 幻觉编号 8%;无格式崩坏(unparseable≈0)。→ **状态跟踪失败,不是大胆试错**。

**头号填充物 = "dead-trial 空转"**:3 颗石头全部入锅后不发 `end the trial`,吐无 reasoning 的裸 ACTION 打已消失的石头,循环到 20-turn 上限:
- 空转 ~**15.7 turns/ep**(B03 仅 1.5);38/64 episode 含 ≥10 连 invalid(B03 7/64);主动结束率 86% vs **99%**。

逐字(B0 rollout_100/ep_6407,trial 8 末五手,全 invalid、无 reasoning):
> `ACTION: Place stone 1 in potion 11` → `ACTION: Place stone 0 in potion 10` → `ACTION: Place stone 1 in potion 11` → `ACTION: Place stone 0 in potion 0` → `ACTION: Place stone 1 in potion 11`

更扎心的对照(B0 rollout_60/ep_3852,invalid turn 上清醒地写):
> "The trial score is now 45… **No further exploration is needed or beneficial**… The remaining turns a[re]…" ——**下一步仍然操作已入锅的石头,而不是 end**。

**reasoning dropout 机制**:两组"越深越裸奔"曲线一致(trial 的 16-20 turn 处 ~70% 无 reasoning);差别是 **B0 常驻深水区**(24.5% 的 turn 零 reasoning vs 8.1%)。退化模式 = reasoning 缺席,不是 reasoning 混乱。

---

## 三、假说的机制检验(信息产出率 + 组内 reward 相关)

| step | B03 yield(novel/turn) | B0 yield | B03 breadth/waste | B0 breadth/waste |
|---|---|---|---|---|
| 20 | 29.2% | 29.0% | 2.37 | 2.09 |
| 60 | 28.4% | 22.9% | 1.75 | **1.03** |
| 80 | 26.7% | 21.9% | 1.51 | **1.00** |

- **B03 的每 turn 信息产出稳定在 27-29%**;B0 跌 ~25%,breadth/waste 跌到 1.00(每多探 1 个 potion 搭进 1 个废 turn)。
- **task reward 对烧 turn 全盲**:step50+ 全部 GRPO 组 pool(B0 376 组),组内 Spearman(score, waste) = **−0.087**,44% 的组里更浪费的一半反而分更高 → 漂移"无人看管",不是被 reward 选中。
- **B0 的绝对探索量反而更高**(novel pairs/ep 30 vs 26、train score 相当)——丢的是**性价比**(1.6× turn 成本买同样信息),不是探索本身;所以训练看不见、eval 塌掉。
- redundant 占比两组几乎相同(~30%)——**分化全部在 invalid**;"漫无目的重测"不是主因。

---

## 结论

1. **explore reward 在这套 setting 里是"行为正则器"**:它是唯一能区分"有产出的 turn"和"垃圾 turn"的梯度来源(judge 只为 valid 新观察付费,空转/invalid 的 sibling 组内垫底)。task reward 对 turn 使用零信号(invalid 免费),budget+v3 prompt 又鼓励长轨迹——拿掉 β,策略必然漂向"不会停"的退化态。
2. **退化形态**:dead-trial 空转(bank 完不 end)+ 深轨迹 reasoning dropout + 状态跟踪崩坏;不是"探索欲过强"。
3. **代价隐蔽而巨大**:train reward 无感,eval hard20 差 **0.13**(0.444 vs 0.317,ckpt69-99)。
4. **β=0.3 是刹车不是墙**:B03 自身也在慢漂(turns 53→94、invalid 1.3→8.6%),且同款退化模式低频存在(7/64)。根治要靠环境侧信号(对 invalid/不 end 直接惩罚)。
5. **与 case1 的联结**:sig4norm(β0.3 co-train)case1 里的"幻觉石头空转"是同一吸引子的低频显形——它是 budgetv3 setting 的固有失稳方向,explore reward 把它压稀,并未消灭。

**注意事项**:B0 仍在训练,step 100+ 数据为部分快照;eval 为单 rep in-training(±0.05-0.08 噪声),但 69-99 五个点方向一致,结论稳。B0 训完后可补终点 offline eval 复核。

**数据路径**:B03 `logs/qwen3-4b-curr950-actonly-w3-expl03-budgetv3-r120-e10-20260630/traj/`;B0 `logs/qwen3-4b-curr950-actonly-w3-b0-budgetv3-r120-e10-20260701/traj/`。分析:3 个只读 subagent(量化演化 / 轨迹取证 / 假说检验)。
