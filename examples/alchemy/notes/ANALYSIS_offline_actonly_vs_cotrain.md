# 离线轨迹分析:act-only vs 双训(co-train) 为什么双训分数更低

## 背景与疑问
预期:如果双训(ACT + WRITE 两路 reward 同时学)能 co-learn 互相促进,上限应**高于**只学 ACT。
实际:**act-only 离线分数显著高于双训**,与预期相反。本分析用同一套 hard-20 测试集的离线推理轨迹,定位差异来源。

## Setting
- **act-only**:`TRAIN_ACT=1, TRAIN_WRITE=0`,评测 ckpt = act119(act-only 最好)
- **双训**:`TRAIN_ACT=1, TRAIN_WRITE=1`,评测 ckpt = co99(双训最好)
- eval:hard-20 + summary-replace + explore + no-thinking,**temperature=0.7**
- 数据:`logs/offline_evals/trajectories/eval-ckpt-{act119,co99,co99-rep2,co99-rep3}/`
- 注:act119 单遍、co99 取 3 遍均值;act119(step119) vs co99(step99) step 不同,但 act99(0.431) 也高于 co99(0.373),结论不受影响。

## 1. 总分 + 逐题对比
- **总均:act119 = 0.473,co99 = 0.373**(双训低 0.10)
- 逐题(per-episode normalized,20 题):
  - **双训明显输的 7 题**(diff>0.15):ep 64/29/6/26/1/11/59(act 0.6+,双训 0.2~0.5)
  - 双训赢的 2 题:ep 9/37
  - 其余 11 题接近
- 差距集中在这 7 题。

## 2. 机制一:双训"单轮上限低"(不是某轮崩,是全程)
每轮 raw 得分:**act119 经常稳定拿 15~30,co99 大量卡在 +2/+3**。例:
```
ep6 (oracle上限/轮 [17,31,30,17,17,1,17,2,16,17]):
  act119: [2, 30, 30, 15, 3, 0, 3, 2, 15, 15]   ← 多轮转出 +15/+30
  co99  : [3, 15, -3, 2, 2, 1, 3, 2, 3, 3]       ← 几乎全卡 +1~+3
```
act-only 学会"用 potion 把石头转成高价值(+15)再投",双训常停在初始低价值石头没做转化。

## 3. 机制二:投出石头的质量差(全 20 题统计)
| 指标 | act119 | co99 | 解读 |
|---|---|---|---|
| 每 trial potion 操作数 | 4.65 | 4.91 | 双训略多(差异不大)|
| 投入石头平均 reward | 6.88 | 5.16 | 双训低 |
| **高价值率(投出 ≥15)** | **43%** | **32%** | 双训更少投出 +15 |
| **负值率(投出 <0)** | **4%** | **12%** | **双训投坏石头是 act 的 3 倍** |

→ 主要差异不在"涂多少 potion",而在**投出石头的质量**:双训更少投高价值、更常投负值(把石头转坏/没转好就投)。

## 4. 典型实例:ep6 trial2(act 得 30 / co 得 -3)
| | act119 | co99 |
|---|---|---|
| 动作 | 2 步 potion 精准转化 → 投 2 石各 **+15** | **6 步 potion 乱涂**(stone0 被反复涂 4 种 potion)→ 投 3 石全 **-1** |
| 该 trial 的 memory | 1916 char | **2452 char(更长更详细)** |

双训 memory 写得更详细,但 ACT 执行时把石头涂坏成 -1;act-only memory 简略却精准 2 步转出 +15。

## 5. 结论
**双训没有 co-learn 互相促进,反而 WRITE 拖累了 ACT 的执行精度:**
- 双训 memory 更详细精确(~1250 vs ~550 token,见 [MEMORY_EVOLUTION_actonly_vs_cotrain.md]),但**用 memory 去操作时反而更差**(高价值率↓、负值率×3)。
- 一句话:**memory 写得好 ≠ 用得好**。在固定模型容量下,WRITE 这条 reward 流摊薄/带偏了 ACT,而非促进它。

## 6. 待解决:WRITE 到底怎么拖累 ACT?
双训分数低,还分不清两种可能:
- **(A) memory 记错了** → 误导决策(WRITE 质量问题)
- **(B) memory 是对的,但 ACT 没按它执行/执行能力退化**(ACT 问题)

## 7. 下一步实验:解耦 ACT 与 WRITE(待跑)
**思路**(用户提出):固定 ACT 执行者,只替换 WRITE 写手,看双训的 write 能力相对 act-only 如何。
- **配置 baseline**:act-only 模型全程(act-only 执行 + act-only 写 memory)
- **配置 解耦**:act-only 模型负责每个 trial 的 ACT 执行,**双训模型负责在 trial 之间写 memory**,下一 trial 仍由 act-only 模型基于这份 memory 去 ACT
- **对比**:解耦 vs baseline 的 hard-20 分数
  - 若 **解耦 > baseline** → 双训的 memory 确实更好(其低分主因是 ACT 退化,即 (B))
  - 若 **解耦 ≈ 或 < baseline** → 双训 memory 也没更优(WRITE 没学到有用东西,即 (A))

这能直接定性"WRITE 拖累 ACT"是 ACT 退化还是 memory 没用。实现要点:eval 框架需支持 ACT/WRITE 由两个不同 served model 分别承担(待确认 eval_alchemy.py 改造量)。

## 8. 解耦实验结果(已跑,2×2 各 3 遍 @0.7, hard-20)
实现:`eval_alchemy.py` 加 `--write-*`(不侵入,默认 WRITE=ACT 同模型),两个 sglang server 分别承担 ACT/WRITE。

| | memory=act119 | memory=co99 |
|---|---|---|
| **执行=act119** | A 0.464±0.010 (3遍) | B 0.448±0.017 (3遍) |
| **执行=co99** | C 0.400±0.010 (4遍,去 outlier) | D 0.373±0.013 (3遍) |

(C 跑了 5 遍 [0.487,0.393,0.395,0.414,0.398],去掉高端 outlier rep1=0.487,取其余 4 遍 = 0.400±0.010。)

**拆解(A 0.464 → D 0.373,共降 0.091):**
- 换执行者 act119→co99(=**ACT 退化**):A→C −0.064、B→D −0.075 → 均 **~0.070**
- 换 memory act119→co99(=**memory 退化**):A→B −0.016、C→D −0.027 → 均 **~0.022**
- **结论:双训低分主因是 ACT 退化(~0.070)≫ memory 退化(~0.022)。co99 的 memory 基本不拖后腿(B≈A),问题主要在执行端。**
- **B≈A**(方差边缘)→ co99 memory 在 act119 执行下基本不拖后腿(假设 (B) 成立:memory 不是主因)。
- **C>D**(+0.052)→ 给 co99 执行换好 memory 能救回一部分,但执行本身是硬伤。

## 9. case 机制(真读 transcript,5 个 episode)
act119 是**忠实的 memory-follower**——同一开局、同一 ACT 模型,决策几乎全由 memory 编码的"世界模型"驱动。两种 memory 写手各有系统性偏差:
- **co99 拖差**:ep6 平铺转移字典→盲试;ep29 探索不足→**错误封顶**(把"+1 是上限"写死,act119 不再追 +15);ep1 真实 +15 锚点→**贪婪**(工具缺席仍追 +15,毁掉到手的 +1)。
- **co99 帮好**:ep37 准确的否定性知识(no effect on purple)→正确收手避自残;ep15 准确转化路径+正确因果归因(potion 造高分)→主动升级。
- 共同根:co99(WRITE 优化)的 memory 偏"全量记录+高成就锚点",但对"本轮实际能拿多少"判断不稳(低估=封顶 / 高估=贪婪);act119 自写 memory 保守贴合眼前,但因果归因粗糙(把 potion 功劳记成"石头自带")。

## 10. 改进假设:scale F_k 的覆盖(不是大小)
**ep29 的"错误封顶"直指 WRITE 信号缺陷**:WRITE reward = memory 对 F_k(该 rollout 未来观察到的转移集)的预测准确率。若那个 rollout 没探到某条转移(如 large round--yellow-->+15),它**不在 F_k 里 → memory 漏掉也不被惩罚 → "封顶 +1" 的 memory 在 WRITE 指标上反而满分**。这是危险的:WRITE 只奖励"记准我看到的",不奖励"世界模型完整"。

**改进方向(待验证):scale F_k 的覆盖来源,而非长度。** 单纯调大 `wm_fk_cap` 无用(没探到的转移再多 cap 也不会出现)。应换/加来源:
- **group 内 sibling rollouts 的转移并集**当 F_k(8 个 sibling 大概率有某个探到了那条转移)——近乎零额外成本
- oracle/环境采样的转移(最彻底,但可能贵/在线难拿)
- 更长 horizon / 跨 trial 的 future

这样"封顶型" memory 拿不到高 WRITE reward,被迫学更完整的世界模型。**注意:此假设只解 ep29 类(覆盖型封顶);ep1 类(贪婪锚点)是 ACT 没结合"当前工具可用性"做条件化决策,属 ACT 侧问题,scale F_k 无效。**

## 11. WRITE 训练效率改进:zero-std group 过滤 + 与 scale F_k 的关系
**观察:`alchemy_grpo/write_group_zero_std_frac > 0.5`** —— 超过一半的 WRITE group 组内 G'=4 个候选 memory 的 reward 完全相同(zero std)。GRPO 里 `(r−mean)/(√var+eps)`,zero-std → advantage=0 → **这些 group 不产生任何梯度**。而 `alchemy_advantage.py` 当前没剔除它们,它们仍占 `max-tokens-per-gpu` 的 batch 名额。

**根因(和 scale F_k 同源):** WRITE reward = G'=4 候选对 F_k 的 generation accuracy。F_k 只 3-12 个转移、acc 离散,4 候选极易撞同值(都对→都满分,或都错→都0)→ zero-std。F_k 越小/越同质,zero-std 越多。

**两个互补改进:**
- **scale F_k 覆盖(治本)**:转移更多更难更多样 → 候选区分度↑ → std>0 → 把"无梯度"变"有梯度"。同时修 §10 的封顶问题。
- **过滤 zero-std group(治标)**:在 `alchemy_advantage.py` 给 zero-std group 标 `remove_sample`(loss_mask=0)→ 释放被 0 优势样本占掉的 batch 名额、避免梯度稀释。**注意:过滤本身不增加学习信号(这些 group 本就 adv=0),只是不浪费 batch;真正"增梯度"靠 scale F_k。**
