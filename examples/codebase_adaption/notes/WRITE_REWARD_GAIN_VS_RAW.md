# WRITE reward:用 gain 还是 raw reward——对 GRPO 梯度等价

## 疑问
alchemy 里 memory-write 的 downstream reward 是 **`reward[k+1] − reward[k]`**(raw,后一个 instance 的 reward 减前一个)。
但 miles 这边 `codebase_advantage.py` 用的是 **`gain[k+1] − gain[k]`**,其中 `gain = reward − baseline`。
问题:这算不算改错了信号?

## 结论(一句话)
**不改梯度。** 因为 WRITE 样本按 `(episode, 位置 k)` 分组做 GRPO,每道题的 `baseline` 在组内是常数,
一减组均值、一除组内 std 就被**精确抵消**。gain 版和 raw 版训练出的 advantage 完全一致(实测差到 1e-16 浮点噪声)。
唯一变的是诊断打印 `WRITE mean_r`。

## 代码位置(ground truth)
- `codebase_rollout.py:684` — `"gain": reward - baseline`
- `codebase_rollout.py:772` — `gains = [o["gain"] for o in outcomes]` → 喂给 `downstream_improve_rewards`
- `codebase_advantage.py:30` — `R(M_k) = mean(gain[k+1..k+K]) − mean(gain[k-K+1..k])`,window=1 即 `gain[k+1]−gain[k]`
- `codebase_advantage.py:11-14` — WRITE 分组键 `("write", group_index, downstream_trial_pos=k+1)`
- `codebase_rollout.py:518-521` — 训练 episode 的 issue 顺序直接取 metadata 里的 `instance_ids`,**n_samples 共享同一顺序**(所以组内位置 k、k+1 是同两道题 → baseline 常数)

## 推导
window=1:
```
R(M_k)^i = gain[k+1]^i − gain[k]^i
         = (r[k+1]^i − b[k+1]) − (r[k]^i − b[k])
         = (r[k+1]^i − r[k]^i) − (b[k+1] − b[k])      ← 记 C = b[k+1]−b[k]
```
组内(同 episode 同位置 k)`C` 是常数。GRPO:`A^i = (R^i − mean_i R)/std_i R`
- 减组均值:`R^i − mean(R) = (r[k+1]^i−r[k]^i) − mean_i(r[k+1]−r[k])` → C 抵消
- 除 std:常数平移不改 std
→ `A^i` 与 baseline 无关。window>1、k0_mode 各分支同理(固定位置 gain 的线性组合,组内常数)。

## 实证(`scripts` 里的 verify_write_reward.py,用真实 codebase_advantage 代码)
构造 1 episode × 5 题 × 8 samples,baseline=[0.30,0.55,0.12,0.62,0.40],per-sample raw reward 随机:

| 检验 | 结果 |
|---|---|
| 组内 Δreward(raw−gain)恒为常数 | ✅ 每组恒等 |
| 该常数 = `b[k+1]−b[k]` | ✅ 完全吻合(0.25/−0.43/0.50/−0.22)|
| **advantage(gain) vs advantage(raw)** | ✅ **max\|Δ\|=4.4e-16** |
| 诊断 mean_r | 不同(−0.0451 vs −0.0201)|

## 影响
- 正在跑的 4B / 9B 正式 run 用 gain,**信号正确,无需重启**。
- 若想让 diagnostic `WRITE mean_r` 与 alchemy 可比,把 `codebase_rollout.py:772` 的 `gain` 换成 raw `reward` 即可——只改打印,不改训练。
