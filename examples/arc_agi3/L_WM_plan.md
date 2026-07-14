# L_WM 接入规划（世界模型损失 → 当前 object 两阶段 RL）

> 目标：把 `world_model_memory_rl.md` 的
> **L_WM = E[ w_t · ( −log P_θ(d_t | x_t, M_t, a_t) ) ]** 接进我们现在的
> **object-only 两阶段（ACT→REWRITE）** 管线。本文只定方案，不写代码。

---

## 1. 现状 ↔ 设计 doc 的映射

| doc | 我们的实现 |
|---|---|
| ① 写 memory `M_t = digest(M_{t-1}, x_{t-1}, a_{t-1}, x_t)`（digest「到达 x_t 的那步」） | **REWRITE**（digest a_t 的结果 x_t→x_{t+1}）→ 产出的 `M_t` 正是下一轮 ACT 用的 `M_{t-1}` |
| ② 选动作 + 预测 `(a_t, d̂_t) = act(x_t, M_t)` | **ACT**（用 `M_{t-1}` + `x_t` 的 object 列表选 `a_t`） |
| `d_t = Δ(x_t, x_{t+1})` 真实 diff | **`objects_diff_text(before, after)`**（对象级 diff，已实现、已验证） |

**关键对齐**：doc 的 `M_t`（act 时所用 memory）= 我们 ACT 时的 `M_{t-1}`（都已 digest 了「到达 `x_t` 的那一步」）。所以 L_WM 的条件 memory 就是 **ACT 序列里的 `M_{t-1}`**，索引无需改，只是命名差一位。

---

## 2. 核心设计决策

### 2.1 d_t 接在哪个序列 —— **ACT 序列**（不是 REWRITE）
- L_WM 预测 `d_t` 时**绝不能看到 `x_{t+1}`**，否则就是抄答案（d_t = after − before）。
- **REWRITE 序列含 after（x_{t+1}）→ 泄露，不能用。**
- **ACT 在 env.step 之前、只有 `x_t`** → 正确的、无泄露的预测点。与 doc ②（act 时预测 d̂，在 env 之前）一致。
- 方案：训练时把 ACT 序列**扩展**为
  ```
  [ prompt: SYSTEM_ACT + M_{t-1} + x_t(objects) ]  +  [ reasoning + action ]  +  [ d_t ]
                                                        └── L_RL(PPO) 段 ──┘    └─ L_WM(NLL) 段 ─┘
  ```
  - `reasoning+action` 段 → 走 GRPO/PPO（L_RL，用 advantage）。
  - `d_t` 段 → teacher-force 真实 `d_t` 的 token，算 NLL × w_t（L_WM）。
- `d̂_t` 是 dead variable（doc 也说），**rollout 不必生成预测**；训练时直接把真实 `d_t`（`objects_diff_text`）拼到 action 之后做 teacher forcing。

### 2.2 门控 w_t
- `w_t = 1 − 1/N(k_t)`，键 `k_t = a_t`（**动作级，先用**，doc §6 倾向）。
- 含义：某动作**第一次**用 → w≈0（首见不可约误差，不罚）；**重复**用还预测不准 → w→1（罚「见过却没记住」= memory erosion）。
- `N` 在 **rollout** 里维护，**per-episode 重置**（最简；跨 episode 全局计数要在并行 rollout 间共享状态，留作后续）。
- 每 turn：`w_t = 1 − 1/N[a_t]`，然后 `N[a_t] += 1`；把 `w_t` 写进该 ACT sample 的 `metadata`。

### 2.3 intrinsic reward（M3 的第二半，建议分步上）
- `r_int = α / √N(s,a)`，count-based（doc §6 先用这个，零额外网络）。
- 合进逐步奖励：`r_t = r_ext + α·r_int`，`r_ext = 1[levels↑]`。
- ⚠ **代价**：这要求 **per-step reward + GAE**，而我们现在是 **episode-sparse reward 广播**（`arc_advantage` 按 episode 分组给同一 advantage）。上 r_int = 要改 advantage 估计。
- **建议**：第一步**只上 L_WM**（reward 维持 episode-sparse），第二步再上 r_int（连带改 advantage）。不要一次改两处。

### 2.4 总损失
```
L_total = L_RL(GRPO, action 段)  +  β · L_WM(d_t 段 NLL × w_t)
```

---

## 3. miles 接入点（已勘探）

- **loss dispatcher**：`loss_hub/losses.py:474 get_loss_function` → `loss_type=custom_loss` 时 `load_function(args.custom_loss_function_path)`。
- **参照实现**：
  - `policy_loss_function`（losses.py:62）= per-token PPO，用 `batch["loss_masks"]` 选 response token、`advantages` 广播、`compute_policy_loss(ppo_kl, advantages)`、`sum_of_sample_mean` 归约。
  - `sft_loss_function`（losses.py:~455）= `loss = -sum_of_sample_mean(log_probs)` —— **就是纯 NLL，L_WM 的现成骨架**。
- **我们要写的** `wm_policy_loss(args, batch, logits, sum_of_sample_mean) -> (loss, metrics)`：
  1. 复用 `policy_loss_function` 的逻辑，但 PPO 只在 **action 段**的 mask 上算 → `L_RL`。
  2. 在 **d_t 段**的 mask 上算 `-log_probs`（teacher-forced NLL），按 sample 的 `w_t` 加权、求和 → `L_WM`。
  3. `return (L_RL + β·L_WM, {pg_loss, wm_loss, w_mean, ...})`。
- **batch 需要携带的新东西**（都从 `Sample` 流进 `RolloutBatch`）：
  - `tokens`：ACT sample 末尾**追加 d_t 的 token**。
  - 一个**分段 mask**：区分「PPO 段（reasoning+action）」与「WM 段（d_t）」——见下方 Open 问题①。
  - `metadata`：`w_wm`（门控）、`phase`（act/rewrite）。REWRITE sample 无 WM 段。

---

## 4. 实现步骤（高层，落地顺序）

1. **（可先做）物体匹配 refine**：Hungarian + 允许 reshape，提升 `objects_diff_text` 的 d_t 质量——它是 L_WM 的监督 target，之前发现 `dy=-9 误配` / `vline64→63 报成 GONE+APPEARED` 的瑕疵会污染 target。
2. **`arc_rollout_objects.py`**：
   - 维护 per-episode `N`（动作级），每 turn 算 `w_t`（先不上 r_int）。
   - ACT sample：把真实 `d_t` tokenize 后追加到 `tokens`（action 之后）；`loss_mask` 标出 PPO 段 / WM 段；`metadata["w_wm"]=w_t`。
3. **`arc_wm_loss.py`（新）**：`wm_policy_loss = L_RL(action) + β·Σ w_t·NLL(d_t)`。
4. **config / run**：`loss_type=custom_loss`、`custom_loss_function_path=examples.arc_agi3.arc_wm_loss.wm_policy_loss`、`β`。
5. **（第二步）** intrinsic reward + per-step GAE（改 `arc_advantage`）。

---

## 5. Open 问题 / 待拍板

1. **分段 loss_mask（最大未知）**：miles 的 `loss_masks` 是单一 0/1（1=算 loss）。我们要在一条 response 里区分**两种 loss**（action→PPO、d_t→NLL）。需确认 `Sample.loss_mask` → `RolloutBatch["loss_masks"]` 的传递能否携带「第二个 mask」或 token-range；否则的方案：用 `metadata` 存 d_t 的 token 区间，custom loss 里自行切分。**这是动手前必须先验证的点。**
2. **两次调用 vs 一次调用的梯度差异**：L_WM 接在 ACT 序列时，`M_{t-1}` 是 ACT 的 **prompt**（上一轮 REWRITE 的输出），不在本序列算 loss → L_WM 的梯度**不直接**塑造 memory 的「书写」，memory 质量靠 RL + 下一轮的预测能力**间接**学。doc 的「一次调用」变体（`M_t` 和 `d̂_t` 同一次生成）梯度更直接。**先用两次调用观察；若 memory 不被 L_WM 有效塑造，再考虑把 ACT 的预测段并进 REWRITE 之后的一次生成**（但要解决「REWRITE 看得到 after」的泄露——可能需要拆成「先 digest 写 M_t、再在不给 after 的前提下预测 d_t」两段同序列）。
3. **N 的范围**：per-episode（简单，先用）vs 全局跨 episode（更贴 meta-RL，但并行 rollout 间要共享计数，需 Ray actor 或文件锁）。
4. **门控键粒度**：动作级（先）→ (动作, state-bucket) → 规则级。state-bucket 可用 object 列表的哈希。
5. **d_t 段的 token 量**：object diff 很短（~50-150 token），NLL 集中在这几十个 token，开销可忽略；但要确认追加后整条 ACT 序列仍在 4096 内。
6. **β、α、γ 调参**：未定。
7. **是否值得先做 §7 的「不上 RL 的证伪实验」**（Opus vs Qwen 轨迹，memory 在场/不在场对 d_t 的 NLL 差）：能在动 RL 前先验证「memory 是 load-bearing 的、L_WM 的 target 形态成立」，性价比高。

---

## 6. 一句话总结

L_WM 最干净的落点是 **ACT 序列尾部 teacher-force 真实 object 级 d_t**（看不到 after、无泄露），用 `custom_loss_function_path` 写 `L_RL(action)+β·Σw_t·NLL(d_t)`，门控 `w_t=1−1/N(a_t)` 在 rollout 里按 per-episode 动作计数算。**先只上 L_WM**，把 intrinsic reward + per-step GAE 留作第二步。动手前**必须先确认 miles 的 loss_mask 能否在一条序列里区分 PPO 段与 NLL 段**——这是唯一的实现级未知。
