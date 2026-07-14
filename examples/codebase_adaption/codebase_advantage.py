"""Two-stream GRPO advantage for codebase-adaptation memory co-training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _group_key(sample):
    md = sample.metadata or {}
    if md.get("phase") == "write":
        if md.get("downstream_trial_pos") is not None:
            return ("write", sample.group_index, md.get("downstream_trial_pos"))
        return ("write", md.get("episode_id"), md.get("rewrite_idx"))
    return ("act", sample.group_index, md.get("trial_pos"))


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def downstream_improve_rewards(
    gains: list[float],
    *,
    window: int = 1,
    k0_mode: str = "improve",
) -> dict[int, float]:
    """Return WRITE rewards for memory M_k from a per-instance gain curve.

    R(M_k) = mean(gain[k+1..k+K]) - mean(gain[k-K+1..k]).
    Last memories without a downstream issue are omitted.
    """
    if k0_mode not in {"improve", "downstream", "skip"}:
        raise ValueError(f"k0_mode must be improve|downstream|skip, got {k0_mode!r}")
    k_win = max(1, int(window))
    out: dict[int, float] = {}
    for k in range(len(gains) - 1):
        nxt = gains[k + 1 : min(len(gains), k + 1 + k_win)]
        if not nxt:
            continue
        if k == 0 and k0_mode == "skip":
            continue
        if k == 0 and k0_mode == "downstream":
            out[k] = _mean(nxt)
            continue
        prv = gains[max(0, k - k_win + 1) : k + 1]
        if not prv:
            continue
        out[k] = _mean(nxt) - _mean(prv)
    return out


def reward_post_process(args, samples):
    """args, flat list[Sample] -> (raw_rewards, advantages)."""
    raw_rewards = [float(s.reward or 0.0) for s in samples]
    std_norm = getattr(args, "grpo_std_normalization", True)

    groups: dict[Any, list[int]] = defaultdict(list)
    for i, sample in enumerate(samples):
        groups[_group_key(sample)].append(i)

    advantages = [0.0] * len(samples)
    for _key, idxs in groups.items():
        vals = [raw_rewards[i] for i in idxs]
        n = len(vals)
        mean = sum(vals) / n
        denom = 1.0
        if std_norm and n > 1:
            var = sum((v - mean) ** 2 for v in vals) / n
            denom = (var**0.5) + 1e-6
        for i in idxs:
            advantages[i] = (raw_rewards[i] - mean) / denom

    act_r = [raw_rewards[i] for i, s in enumerate(samples) if (s.metadata or {}).get("phase") != "write"]
    wr_r = [raw_rewards[i] for i, s in enumerate(samples) if (s.metadata or {}).get("phase") == "write"]
    if act_r or wr_r:
        print(
            f"[codebase_advantage] ACT n={len(act_r)} mean_r={_mean(act_r):.4f} | "
            f"WRITE n={len(wr_r)} mean_r={_mean(wr_r):.4f} | groups={len(groups)}",
            flush=True,
        )
    return raw_rewards, advantages

