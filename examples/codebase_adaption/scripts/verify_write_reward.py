"""实证: WRITE reward 用 gain vs raw reward, GRPO advantage 是否一致。
用真实 codebase_advantage.py 的 downstream_improve_rewards + reward_post_process。"""
import sys, random
sys.path.insert(0, "/home/qixinx/miles")
from examples.codebase_adaption.codebase_advantage import downstream_improve_rewards, reward_post_process

# —— 构造一个真实场景 ——
random.seed(42)
K = 5            # episode 内 5 道 issue (5+5 里取一段示意)
N = 8            # n_samples_per_prompt
# 每道 issue 的 baseline(组内常数, 同一 episode 固定顺序 -> 所有 sample 共享)
baselines = [0.30, 0.55, 0.12, 0.62, 0.40]
# 每个 sample 各 issue 的 raw reward(不同轨迹不同)
R = [[round(random.uniform(0, 1), 3) for _ in range(K)] for _ in range(N)]

# 每个 sample 的 gain 曲线 & raw 曲线, 各自算 write rewards
wr_gain = []   # gain 版: gain=reward-baseline
wr_raw  = []   # raw 版: 相当于 baseline=0
for i in range(N):
    gains = [R[i][k] - baselines[k] for k in range(K)]
    raws  = [R[i][k] for k in range(K)]
    wr_gain.append(downstream_improve_rewards(gains, window=1, k0_mode="improve"))
    wr_raw.append(downstream_improve_rewards(raws,  window=1, k0_mode="improve"))

# —— 构造 Sample 对象, 走真实 reward_post_process ——
class S:
    def __init__(self, reward, group_index, k):
        self.reward = reward
        self.group_index = group_index
        self.metadata = {"phase": "write", "downstream_trial_pos": k + 1, "rewrite_idx": k}
class Args:
    grpo_std_normalization = True

# 每个 write 位置 k 是一个 GRPO 组(group_index 同=同 episode, downstream_trial_pos=k+1)
# 把所有 sample、所有位置的 write 样本平铺, reward_post_process 内部按 _group_key 分组
def build(wr):
    samples = []
    for k in range(K - 1):        # 最后一个位置无 downstream, 被省略
        for i in range(N):
            if k in wr[i]:
                samples.append(S(wr[i][k], group_index=0, k=k))
    return samples

sg = build(wr_gain); sr = build(wr_raw)
_, adv_g = reward_post_process(Args(), sg)
_, adv_r = reward_post_process(Args(), sr)

print("\n===== 逐 write 位置 k, 逐 sample 对比 =====")
print(f"{'k':>2} {'i':>2} | {'reward_gain':>11} {'reward_raw':>10} {'Δreward(=C常数?)':>16} | {'adv_gain':>9} {'adv_raw':>9} {'Δadv':>10}")
idx = 0
maxdiff = 0.0
for k in range(K - 1):
    Cs = []
    for i in range(N):
        if k not in wr_gain[i]: continue
        rg, rr = sg[idx].reward, sr[idx].reward
        ag, ar = adv_g[idx], adv_r[idx]
        Cs.append(rr - rg)
        d = abs(ag - ar); maxdiff = max(maxdiff, d)
        print(f"{k:>2} {i:>2} | {rg:>11.4f} {rr:>10.4f} {rr-rg:>16.4f} | {ag:>9.5f} {ar:>9.5f} {d:>10.2e}")
        idx += 1
    # 组内 Δreward 应全等 = C = b[k+1]-b[k]
    C_expected = baselines[k+1] - baselines[k]
    print(f"   -> 组 k={k}: Δreward 组内是否恒等={max(Cs)-min(Cs) < 1e-9}, C={Cs[0]:.4f}, 理论 b[k+1]-b[k]={C_expected:.4f}")

print(f"\n===== 结论 =====")
print(f"最大 |adv_gain - adv_raw| = {maxdiff:.3e}")
print("→ advantage 完全一致" if maxdiff < 1e-9 else "→ !! 有差异, 推导错了")
