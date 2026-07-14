# Memory Evolution Analysis: Freeform vs Downstream Signals

这份分析关注一个问题：

**不同 WRITE reward signal 训练出来的 memory，到底是越来越短、越来越长，还是保持稳定？这些长度变化对应的内容质量是什么？**

分析对象：

- `freeform`: `qwen3-4b-curr950-freeform-r120-e10-20260621-022402`
- `sig3_downstream`: `qwen3-4b-curr950-sig3down-fixed-r120-e10-20260625`
- `sig4_raw_improve`: `qwen3-4b-curr950-sig4improve-fixed-r120-e10-20260625`
- `sig4_norm_improve`: `qwen3-4b-curr950-sig4normimprove-userdata-r120-e10-20260625-232325`

统计文件：

- `examples/alchemy/logs/memory_evolution_analysis.json`
- `examples/alchemy/logs/memory_evolution_analysis_auto.md`

我额外手动读了若干代表 case，不只看长度统计。

## 1. 结论先讲

你的观察基本成立，但需要加一个 caveat：

1. **freeform memory 一直保持很长，而且略微变长。**  
   final memory 平均长度从 early 的 `10036 chars` 到 late 的 `10444 chars`，基本稳定在 `10k+`。

2. **sig3_downstream 的 memory 长度显著坍缩。**  
   final memory 平均长度从 `1732 chars` 降到 `306 chars`，只剩 early 的 `17.7%`。这不是轻微压缩，而是非常明显的 brevity collapse。

3. **sig4_raw_improve 也明显变短，但没有 sig3 那么极端。**  
   final memory 平均长度从 `1703 chars` 降到 `801 chars`，约为 early 的 `47.0%`。它仍然保留若干有用 transition，但细节明显减少。

4. **sig4_norm_improve 目前没有明显变短。**  
   这组目前只有 rollout `0-73` 的轨迹，final memory 平均长度从 `1787 chars` 到 `1722 chars`，基本稳定。不能直接和跑满 119 的 sig3/sig4raw 做最终趋势对比，但至少说明 norm-improve 没有出现 sig3 那种快速坍缩。

## 2. Length Trend

| signal | episodes | last rollout | early final length | late final length | late / early | change |
|---|---:|---:|---:|---:|---:|---:|
| freeform | 7680 | 119 | 10035.7 | 10443.9 | 1.041 | +408.1 |
| sig3_downstream | 7680 | 119 | 1731.5 | 306.0 | 0.177 | -1425.5 |
| sig4_raw_improve | 7680 | 119 | 1702.8 | 800.9 | 0.470 | -902.0 |
| sig4_norm_improve | 4681 | 73 | 1786.7 | 1722.4 | 0.964 | -64.4 |

更细一点看几个 rollout：

| signal | rollout 0 | rollout 39/40 | rollout 79/80 | last |
|---|---:|---:|---:|---:|
| freeform | 9961 | 10187 | 10408 | 10441 |
| sig3_downstream | 1806 | 723 | 288 | 335 |
| sig4_raw_improve | 1832 | 989 | 760 | 887 |
| sig4_norm_improve | 1804 | 2075 | N/A | 1782 at rollout 72 |

这里的 `length` 是每个 episode 最后一个 memory 的字符数平均值。

## 3. Freeform: 一直很长，而且经常过度叙述

freeform 的 memory 非常稳定地维持在 `10k` 字符左右。它不会因为训练推进而变短，反而 late stage 略微增长。

### Case: rollout 0, episode 203

路径：

`examples/alchemy/logs/qwen3-4b-curr950-freeform-r120-e10-20260621-022402/traj/train/rollout_0/ep_41_episode_203.json`

final memory 长度：`10045`

内容片段：

```text
✅ Episode Summary: Hidden Chemistry (Final Updated After This Trial — Full Validation)

### Confirmed Effects of Potions

| Potion | Observed Effect | Evidence | Notes |
| Yellow (potion 4, 5, 6) | Increases reward — but only on blue large pointy stones | ... |
| Pink (potion 1, 2) | No effect | ... |
| Orange (potion 2, 7, 8, 10) | No effect | ... |
...
```

这个 memory 有两个特点：

- 它试图维护一张很完整的表，包含 potion、证据、notes、限制条件。
- 它会反复写 “confirmed / direct evidence / no effect / validation” 这类话，信息密度不高。

### Case: rollout 119, episode 999

路径：

`examples/alchemy/logs/qwen3-4b-curr950-freeform-r120-e10-20260621-022402/traj/train/rollout_119/ep_7634_episode_999.json`

final memory 长度：`10629`

内容片段：

```text
Episode Hidden Chemistry Summary
Version: 1.8 — Updated from Final Trial

### Confirmed Elemental Affinities & Transformations

1. Orange Potion — Master Transformer of Shape & Reward
- Effect on blue large round stone:
  - Converts low-reward blue large round (+1) -> high-reward blue large pointy (+15).
  - Transformation is direct, reversible, and highly efficient.
  - Strictly requires: blue large round substrate and presence of orange.
...
```

这个 late case 仍然很长，而且出现了更强的叙事风格，例如 “Master Transformer”、“activation energy”、“Version 1.8”。它不是没有信息，但会把少量 transition 包装成非常长的 explanation。

**结论：freeform 的问题不是不会写 memory，而是太愿意写，且没有长度/稀疏性压力。**  
它像一个 verbose scratchpad，而不是一个紧凑的 knowledge state。

## 4. sig3_downstream: 长度显著坍缩，最后接近极简规则表

sig3 的 reward 是 `r_{k+1}`，也就是只看下一步 downstream reward。训练后 memory 变得非常短。

### Case: rollout 0, episode 18

路径：

`examples/alchemy/logs/qwen3-4b-curr950-sig3down-fixed-r120-e10-20260625/traj/train/rollout_0/ep_0_episode_18.json`

final memory 长度：`1740`

内容片段：

```text
### Potion Effects
- Red: Increases reward for small pointy stones ...
- Green: No effect on any stone tested ...
- Yellow: Decreases reward for small pointy stones ...
- Orange: No effect observed ...
- Turquoise: No effect observed ...
- Pink: No observed effect ...

### Highest Reward Combination
+1 reward achieved with a blue large round stone ...
```

early sig3 memory 还是比较完整的：它覆盖多个 potion，也解释了 no-effect 和 harmful effect。

### Case: rollout 80, episode 343

路径：

`examples/alchemy/logs/qwen3-4b-curr950-sig3down-fixed-r120-e10-20260625/traj/train/rollout_80/ep_5120_episode_343.json`

final memory 长度：`268`

内容：

```text
### Potion Effects
- Pink: Increases reward from -1 to +15 in blue large round stone (transformed to blue small round)
- Orange: Transforms purple large round into blue large round with reward +1

### Highest Reward Combination
Blue small round with reward +15
```

到 rollout 80，memory 已经只剩两条 potion effect 和一个最高组合。它很干净，但明显丢掉了很多 uncertainty 和 counter-evidence。

### Case: rollout 119, episode 203

路径：

`examples/alchemy/logs/qwen3-4b-curr950-sig3down-fixed-r120-e10-20260625/traj/train/rollout_119/ep_7646_episode_203.json`

final memory 长度：`323`

内容：

```text
### Potion Effects
- Yellow: preserves +1 reward (no change in stone feature, neutral effect)
- Red: transforms purple large pointy to blue large round, +1 reward (effective transformation on large pointy to round)

### Highest Reward Combination
Blue small round with reward +15 (achieved via stone 1 in potion 10)
```

这个 case 很能说明 sig3 的问题：它学会了“短”，但不一定学会了“充分”。`Highest Reward Combination` 里提到 `stone 1 in potion 10`，但 potion 10 的颜色、可复用规则、适用条件都没有保留下来。对后续 ACT 来说，这种 memory 可能不够 actionable。

**结论：sig3_downstream 可能把 WRITE policy 推向了极度压缩。**  
这可能是因为 `r_{k+1}` 对 memory 长度没有显式奖励，短 memory 更容易稳定生成、少犯格式错；而只要下一题偶然表现不错，短 memory 也会被强化。

## 5. sig4_raw_improve: 也变短，但比 sig3 保留更多 transition 结构

sig4 raw improve 的 reward 是 `r_{k+1} - r_k`。它也让 memory 变短，但没有 sig3 那么极端。

### Case: rollout 0, episode 180

路径：

`examples/alchemy/logs/qwen3-4b-curr950-sig4improve-fixed-r120-e10-20260625/traj/train/rollout_0/ep_16_episode_180.json`

final memory 长度：`1782`

内容片段：

```text
### Potion Effects
- Orange: Increases reward of purple small round stones from -1 to +1 ...
- Turquoise: Has no effect on any stone when tested ...
- Pink: No effect ...
- Green: No effect ...
- Yellow: No effect ...
- Red: Not tested ...

### Highest Reward Combination
- The highest reward seen is +15, achieved by the blue large round stone ...
```

early sig4 raw 和 early sig3 类似，也比较完整。

### Case: rollout 80, episode 326

路径：

`examples/alchemy/logs/qwen3-4b-curr950-sig4improve-fixed-r120-e10-20260625/traj/train/rollout_80/ep_5153_episode_326.json`

final memory 长度：`704`

内容：

```text
### Potion Effects
- Orange: Transforms blue large pointy to blue small pointy (reward +1 -> +15)
  and purple large pointy to purple small pointy (reward -1 -> +1);
  strongly increases reward in pointy stones, particularly when transforming large to small
- Red: No effect on reward ...
- Turquoise: No effect on reward ...

### Highest Reward Combination
+15 reward from blue small pointy stone ...
```

相比 sig3，sig4 raw 的 late memory 仍然保留了：

- transformation source and target
- reward change
- some applicability condition
- no-effect controls

### Case: rollout 119, episode 203

路径：

`examples/alchemy/logs/qwen3-4b-curr950-sig4improve-fixed-r120-e10-20260625/traj/train/rollout_119/ep_7641_episode_203.json`

final memory 长度：`876`

内容：

```text
### Potion Effects
- Pink: Converts a blue large round stone into a blue small round stone, increasing reward from +1 to +15.
- Pink: Converts blue large pointy stone into blue small pointy stone, increasing reward from -1 to +1.
- Pink: Has no effect on purple large pointy stone ...
- Red: Converts purple small round stone into blue small round stone, increasing reward to +15.
- Red: Converts purple small pointy stone into blue small pointy stone ...
- Yellow: Converts blue small pointy stone into blue small round stone, increasing reward to +15.

### Highest Reward Combination
Blue small round stone with a reward of +15 ...
```

这个 memory 比 sig3 late case 更有用：它保留了多个可执行 transition。缺点是它可能把一些 “transforms but reward still bad” 写成 no-effect 或者过度泛化，但整体比 sig3 更像 compact knowledge。

**结论：sig4 raw improve 也带来压缩压力，但压缩后仍保留比较多 action-relevant transition。**  
不过因为 raw improve 本身很 noisy，这种压缩是否稳定提升 policy 还需要 offline eval 支持。

## 6. sig4_norm_improve: 长度目前稳定，内容更像 early-stage structured memory

sig4 norm improve 的 reward 是：

```text
r_{k+1}/oracle_{k+1} - r_k/oracle_k
```

这组目前只到 rollout 73，所以不能直接判断最终是否会变短。但目前看，它没有出现 sig3/sig4raw 那样的长度坍缩。

### Case: rollout 0, episode 716

路径：

`examples/alchemy/logs/qwen3-4b-curr950-sig4normimprove-userdata-r120-e10-20260625-232325/traj/train/rollout_0/ep_13_episode_716.json`

final memory 长度：`1696`

内容片段：

```text
### Potion Effects
- Red: No effect on reward value ...
- Pink: No effect on reward value ...
- Green: Improves reward for low-reward stones ...
- Turquoise: Increases reward for negative-reward stones ...
- Orange: No observed effect yet ...

### Highest Reward Combination
+15 reward is the highest achieved ...
```

### Case: rollout 73, episode 843

路径：

`examples/alchemy/logs/qwen3-4b-curr950-sig4normimprove-userdata-r120-e10-20260625-232325/traj/train/rollout_73/ep_4734_episode_843.json`

final memory 长度：`1540`

内容：

```text
### Potion Effects
- Green: When applied to blue small round, changes it to blue small pointy with reward -1;
  when applied to purple large round, changes it to purple large pointy with reward +15
- Turquoise: When applied to purple small pointy, transforms it to purple large pointy with reward +15;
  when applied to blue small pointy, transforms it to blue large pointy with reward +1
- Red: When applied to blue small pointy, changes it back to blue small round with reward -3
- Yellow: no change on tested blue pointy stones
- Pink: no change on tested blue stones

### Highest Reward Combination
Purple large pointy stone (reward +15) achieved via green ...
Purple large pointy stone (reward +15) achieved via turquoise ...
```

这个 memory 比 sig3 late case 长很多，也更像“evidence-preserving summary”：有 positive transitions，也有 negative/no-effect controls。

**结论：sig4_norm_improve 目前没有强烈压缩 memory。**  
这可能是好事，也可能意味着它还没学到简洁表达。需要等它跑到 119 后再比较。

## 7. Interpretation

### 7.1 Memory length 不是越短越好

sig3 的 memory 变短非常明显，但 case 里能看到：它有时短到只剩两三条规则，缺少适用条件、颜色、potion identity 或 counter-evidence。这种 memory 看起来简洁，但可能不够支撑后续 ACT。

### 7.2 Freeform 长不是因为信息多，而是没有压缩压力

freeform 的 late memory 仍然很长，且包含大量 meta language：

- “confirmed”
- “validation”
- “master transformer”
- “activation energy”
- “version”
- long tables

这些会占 context，但不一定增加 decision-relevant information。

### 7.3 sig4 raw 的压缩形态目前最像 usable compact memory

sig4 raw 的 late memory 通常在 `700-900 chars`，比 sig3 更完整，比 freeform 更紧凑。它会保留多个 transformations，例如：

```text
Pink: blue large round -> blue small round, +1 -> +15
Red: purple small round -> blue small round, +15
Yellow: blue small pointy -> blue small round, +15
```

这类 memory 对 ACT 是 actionable 的。

### 7.4 sig4 norm 的长度稳定说明 normalization 可能改变了 WRITE 的压缩压力

raw improve 可能会奖励“短、直接、容易带来下一步提升”的 memory；norm improve 由于按 oracle 归一化，可能不会那么强地鼓励极端压缩。但目前只跑到 73，不能下最终结论。

## 8. Takeaways for Next Experiments

1. **需要把 memory length/structure 加成正式监控指标。**

建议 wandb 加：

```text
write/memory_final_len_mean
write/memory_final_len_p10
write/memory_final_len_p90
write/memory_bullets_mean
write/memory_lines_mean
```

2. **sig3 如果跑 discounted future reward，要同时监控 memory length。**

因为 sig3 当前已经有明显 brevity collapse。future reward 可能缓解 myopic credit，但也可能继续奖励短 memory。

3. **sig4 raw 值得离线测，因为它的 memory 形态看起来不错。**

它不是 performance 一定好，但从 case 上看，它生成的 memory 比 freeform 更紧凑，比 sig3 late memory 更可用。

4. **sig4 norm 需要等完整 119 后再做同样分析。**

目前它的 memory 没有明显缩短，而且内容保留证据更充分；这可能是一个优点。

## 9. Bottom Line

这组 memory analysis 支持一个比较清楚的故事：

- **freeform**: memory 很长，信息密度低，像 verbose scratchpad。
- **sig3_downstream**: memory 极度变短，但可能短到丢失关键条件。
- **sig4_raw_improve**: memory 变短但仍保留 actionable transitions，是目前最像 compact knowledge 的形态。
- **sig4_norm_improve**: 目前长度稳定，证据保留更多，但需要完整训练后再判断。

所以后续不应该只看 downstream norm score，也应该看 memory 是否在变成一个真正可用的 knowledge state。
