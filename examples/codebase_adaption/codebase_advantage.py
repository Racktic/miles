"""Two-stream GRPO advantage for codebase-adaptation memory co-training."""

from __future__ import annotations

import re
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


def gated_downstream_rewards(
    rewards: list[float],
    format_ok: dict[int, bool],
    *,
    bonus: float = 0.1,
) -> dict[int, float]:
    """WRITE rewards for the actwrite experiment (user-approved 2026-07-16).

    R(M_k) = format_ok_k * reward[k+1] + bonus * format_ok_k

    - ``reward[k+1]`` is the NEXT issue's raw train reward (no delta): the
      mean-reversion bias of delta rewards (corr(R, gain_k) = -0.736, see
      WRITE_COLLAPSE_ANALYSIS_0714) came from subtracting one's own past;
      instance difficulty is instead normalized by GRPO's within-group
      advantage (all samples of a group face the same next issue).
    - ``format_ok_k``: the stored memory contains BOTH required section
      headers. Dense, unbiased, cheap; targets the 94%-vs-33% compliance
      gate (post-failure rant mode) that ACT-only training never moved.
    - The bonus keeps a positive gradient toward well-formed memories even
      when the next issue fails — exactly the situation where rant mode
      dominates. Memories with no downstream issue are omitted (as legacy).
    """
    out: dict[int, float] = {}
    for k in range(len(rewards) - 1):
        f = 1.0 if format_ok.get(k) else 0.0
        out[k] = f * float(rewards[k + 1]) + float(bonus) * f
    return out


def gated_delta_window_rewards(
    rewards: list[float],
    format_ok: dict[int, bool],
    *,
    window: int = 3,
    bonus: float = 0.1,
) -> dict[int, float]:
    """WRITE rewards with a windowed downstream *difference* (user-approved 2026-08-03).

    R(M_k) = mean(reward[k+1..k+K]) - mean(reward[k-K+1..k]) + bonus * format_ok_k

    - Window K (default 3): the next K issues minus the previous K issues. The
      previous window INCLUDES issue k itself, which is the last issue solved
      before M_k was written, so the comparison is "after writing M_k" vs
      "before writing M_k".
    - **Additive** gating, unlike ``gated_downstream_rewards``' multiplicative
      one. A windowed difference can be NEGATIVE, and a multiplicative gate
      would then make a badly-formatted memory (R=0) score HIGHER than a
      well-formatted one before a downhill stretch (R<0) — turning format
      failure into an escape hatch. Additive keeps the format term a constant
      +bonus, decoupled from the sign of the difference.
    - Windows are truncated at episode boundaries (k<K-1 has a shorter previous
      window; k near the end has a shorter next window), same convention as
      ``downstream_improve_rewards``. Means (not sums) so truncation does not
      change the scale.
    - Memories with no downstream issue at all are omitted (as legacy).
    """
    k_win = max(1, int(window))
    out: dict[int, float] = {}
    n = len(rewards)
    for k in range(n - 1):
        nxt = rewards[k + 1 : min(n, k + 1 + k_win)]
        prv = rewards[max(0, k - k_win + 1) : k + 1]
        if not nxt or not prv:
            continue
        f = 1.0 if format_ok.get(k) else 0.0
        out[k] = _mean(nxt) - _mean(prv) + float(bonus) * f
    return out


_RK_HEADER = re.compile(r"^###\s*Repository Knowledge\s*$", re.M)
_LP_HEADER = re.compile(r"^###\s*Lessons\s*&\s*Pitfalls\s*$", re.M)
_ANY_HEADER = re.compile(r"^###\s+.*$", re.M)
_BULLET = re.compile(r"^\s*[-*]\s+\S")
# Prompt/scaffold markers that must never appear in a memory (prompt-echo hack).
_PROMPT_MARKERS = ("=== Brief ===", "=== Repository Knowledge ===", "COMPLETE_TASK_AND_SUBMIT")


def _section_bullets(text: str) -> int:
    return sum(1 for ln in text.splitlines() if _BULLET.match(ln))


def memory_format_ok(memory: str, prev_memory: str | None = None, *, min_bullets: int = 2) -> bool:
    """Strict format gate (2026-07-17, hardened after reward-hacking on the loose
    check: empty two-header shells, prompt echo, repeated-header collapse, and
    verbatim copy of the previous memory all scored format_ok=True and earned the
    bonus). Requires an actual well-formed memory:

    1. EXACTLY one ``### Repository Knowledge`` and one ``### Lessons & Pitfalls``
       header, each on its own line, and NO other ``###`` headers (kills
       multi-header collapse and prompt echo that dumps extra headers).
    2. No prompt/scaffold markers anywhere (kills copying the injected prompt).
    3. Each of the two sections has >= ``min_bullets`` non-empty bullet lines
       (kills the empty two-header shell).
    4. Not a verbatim copy of ``prev_memory`` (kills forward-copying garbage).
    """
    m = memory or ""
    if len(_RK_HEADER.findall(m)) != 1 or len(_LP_HEADER.findall(m)) != 1:
        return False
    if len(_ANY_HEADER.findall(m)) != 2:
        return False
    if any(mark in m for mark in _PROMPT_MARKERS):
        return False
    if prev_memory is not None and m.strip() == (prev_memory or "").strip():
        return False
    heads = list(_ANY_HEADER.finditer(m))
    for i, h in enumerate(heads):
        start = h.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(m)
        if _section_bullets(m[start:end]) < min_bullets:
            return False
    return True


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

