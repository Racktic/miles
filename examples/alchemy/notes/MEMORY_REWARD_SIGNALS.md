# Memory Reward Signals — 候选信号与研究计划

> **目标**:为 memory writing(WRITE 流)找到比当前 transition-acc 更好的 reward 信号,把 memory 从
> "mapping 字典"推向"actionable knowledge"。本文汇总候选信号 + 各自衡量 memory 的哪种**性质**,作为
> 接下来一段时间的研究主线。**只讨论"有哪些信号 / 各衡量什么",不含具体估计/实现细节**(那是选定信号之后的事)。

相关:`../ANALYSIS_offline_actonly_vs_cotrain.md`(act-only vs 双训,"写得好≠用得好")、
`../IMPL_PLAN_scale_fk_and_filter_zerostd.md`(transition-acc 侧的改进)。

---

## 动机

- 当前 WRITE reward = memory 对 `F_k`(未来转移集)的 exact-match 预测准确率(= 下面的**信号 1**)。
- 两个已观察到的问题:
  1. **它是 intrinsic proxy**:"记得准" ≠ "对决策有用"。co99 的 memory 在 transition 层面不差,但 ACT 用了反而差(详见 ANALYSIS 文档)。
  2. **它在结构上诱导 mapping**:要把转移预测准,最省事就是把 `(stone, potion)→result` 背成一张表,根本不需要抽象成 knowledge。**co-train memory 长得像 mapping,是这个 reward 的 artifact,不是模型的本意**。
- 老师的方向:memory 好不好,最终要看 **agent 用了之后做得怎么样** → 应把 policy 表现纳入 memory writing 的信号。

## 核心洞察:credit assignment(为什么 policy 信号能"白拿")

trial k 的得分 `r_k = f(M_{k-1}, ACT_k, 局面_k)`。当前把 `r_k` 的 advantage **全记给 ACT_k**。但在
ACT group = `(episode g, trial_pos k)` 里:同 episode 同 trial 的**初始局面相同**、ACT **同一套参数**,
所以 sibling 之间 `r_k` 的差异**主要来自各自持有的 `M_{k-1}` 不同**。

→ 这个 advantage 本来就(部分)**属于 memory**,现在全给 ACT 是 credit 漏分。
→ 应该(也)**回传给写 `M_{k-1}` 的那次 WRITE**。
→ **双重 credit**:同一个 `r_k` 既教"怎么做"(ACT)又教"怎么记"(WRITE),且**几乎零成本**(复用现成 ACT advantage)。

信号 3 / 4 都是这个洞察的具体化(区别只在"减什么 baseline")。

---

## 候选信号

### 信号 1 — transition-acc(现状)
- **定义**:memory 对 `F_k` 逐条转移的生成预测准确率。
- **衡量**:memory 记得**对不对**。
- **性质**:intrinsic;dense;相对便宜。**但诱导 mapping**,且"记得准"与"用得好"脱节。
- **角色**:可能保留作"正确性"正则项,而非主信号。

### 信号 2 — counterfactual vs 原始历史
- **定义**:`reward(M_k) = score(M_k, k+1) − score(M_{k-1} + trial_k 原始历史, k+1)`。
- **衡量**:把 trial k 压缩成 memory 这个 **writing 动作的净损益**。baseline 含 trial k 的**全部原始信息**,M_k 只是它的压缩 → **信息量被控制住,唯一变量是"压缩/抽象得好不好"**。
- **性质**:instrumental;有**绝对符号**(>0 = 压缩得比原始历史还好用,提炼出 actionable knowledge;≈0 = 无损;<0 = 丢信息/写坏)。本质是 replace vs no-summary 的 per-step 版,**目标 = 把 writing 训到"压缩 ≥ 原始"**。
- **注意**:这个公共 baseline 在 G' 候选**组内白化下会被吸收**(advantage 里消掉)→ 要起作用得当**绝对锚**(换 group 来源,或拿来 filter 掉 reward<0 的坏 memory)。需额外跑 baseline rollout(G' 候选共享,+1/点)。

### 信号 3 — 横向 advantage(group baseline)
- **定义**:`reward(M_{k-1}) = r_k − mean_i r_k^i` —— 把 ACT group(同 episode 同 trial_pos)白化后的 advantage 回传给写 `M_{k-1}` 的 WRITE。
- **衡量**:这份 memory **比其他 sibling 的 memory 好多少**(横向相对)。
- **性质**:instrumental;**几乎免费**(复用现成 ACT advantage,无额外 rollout / 无 G' 候选);group 天然控制了 trial 难度 + ACT 执行能力 → 隔离出 memory 的横向相对贡献。
- **注意**:`M_{k-1}` 的质量纠缠"前面探索到什么 × 怎么写"(advantage 会把探索的功劳也算进 writing);只用 `r_k` 是近端,完整 credit 含后续 trial。

### 信号 4 — 纵向 improvement(temporal baseline)★最新、最贴目标
- **定义**:`reward(M_{k-1}) = r_k − r_{k-1}` —— 用 rollout **自己上一轮**当 baseline。
- **衡量**:这份 memory 让 agent **比上一轮进步多少 = 从历史中学到并用上了多少**。
- **为什么需要**:绝对 `r_k` 把"rollout 本来的水平"算进 memory 功劳;用自己的 `r_{k-1}` 去基线才隔离出"进步"。
  例:`A: 0.1→0.5 (Δ+0.4)` 的 memory 其实优于 `B: 0.9→0.8 (Δ−0.1)`,尽管 B 绝对分更高。
  **而且这直接 align 我们 meta-learning 的真目标 I_score**(本来就是 cross-trial improvement)。
- **性质边界**(是"信号衡量什么"的语义,不是估计问题):
  - **telescoping**:`r_{k-1}` 又被 `M_{k-2}` 影响 → Δ 实为"M_{k-1} − M_{k-2}",相邻 memory 的 credit 互相抵消。
  - **天花板**:`r_{k-1}` 接近 oracle 上限时 Δ 必 ≈0/负 → 系统性**惩罚"在好轨迹上写的 memory"**。
  - **维持型被低估**:作用是"维持高分 / 防退步"的好 memory(Δ≈0)会被判没用。
- **可与信号 3 组合**:`(r_k − r_{k-1})` 再做 group 白化 = "这份 memory 带来的进步 vs 其他 sibling 的进步"。

### 信号 6 — 信息论 / 决策确定性
- **定义**:memory 让 policy 对好动作(oracle / 高分动作)的把握提升多少(如 `log π(a*|s, M)` 的增益,或动作分布熵的下降)。
- **衡量**:memory 让决策**多确定 / 多对**。
- **性质**:dense + 便宜(一次 forward,不 rollout);**但是 proxy**(likelihood ≠ 实际 rollout 表现),整体不如 1-4 直接 → 作为备选。

---

## 已否决

- **信号 5(vs no-memory 反事实)**:no-memory 裸跑无任何先验,分数必然很低 → `r_k(memory) − r_k(no-memory)` 几乎恒为大正数,**不区分 memory 好坏**,无意义。

---

## 信号一览(选型总览)

| # | 信号 | memory 的价值 = | 抓的性质 | 成本 | 关键弱点 |
|---|---|---|---|---|---|
| 1 | transition-acc(现状) | 预测转移准 | **记得对** | 低 | 诱导 mapping;与"用得好"脱节 |
| 2 | counterfactual vs 原始历史 | 压缩净损益 | **压缩有没有损** | 中(+baseline rollout) | 组内白化会吸收 baseline |
| 3 | 横向 advantage | 比同侪 memory 好 | **横向相对改变** | 极低(复用) | 纠缠探索 + temporal |
| 4 | 纵向 improvement ★ | 比自己上轮进步 | **从历史学到多少** | 低(复用 r) | telescoping / 天花板 / 维持低估 |
| 6 | 决策确定性 | 让决策更确定/对 | **决策确定性** | 低(一次 forward) | proxy,不如 1-4 直接 |

一句话:**1 衡量"对不对",2 衡量"压缩有没有损",3/4 衡量"带来多少改变(横向/纵向)",6 衡量"让决策多确定"**。
它们不互斥 —— 真正的问题是"**我们想让 memory 具备哪几种性质**",这决定用哪几个、怎么组合。

---

## 开放问题 / 下一步

1. **我们想让 memory 具备哪几种性质?**(对不对 / 有没有用 / 带来多少进步 / 多确定)→ 决定信号选择与组合。
2. **transition-acc 彻底放弃,还是保留作"正确性"正则**,与 instrumental 信号并存(一个保证"记得对",一个保证"用得好")?
3. **credit 的 horizon**:近端单 trial vs 后续折扣累积(这是估计层面,选定信号后再谈)。
4. **是否走双重 credit**(同一 advantage 同时喂 ACT + WRITE)。

_(讨论进行中;1–4、6 为当前保留的候选,5 已否决。)_
