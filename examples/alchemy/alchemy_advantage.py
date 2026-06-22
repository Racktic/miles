"""Two-stream GRPO advantage for Symbolic Alchemy (per-trial co-train of writing + using memory).

The per-trial rollout (`alchemy_rollout._generate_trial`) emits two kinds of Sample, tagged in metadata:
  - ACT   (phase="act"):   one per executed step; reward = that trial's score.
  - WRITE (phase="write"): G' candidate memories per REWRITE point; reward = memory accuracy on F_k.

Each stream is whitened within its own grouping (GRPO: subtract group mean, optionally divide by group std):
  - ACT   grouped by (group_index, trial_pos)  -> compares a trial across the n_samples sibling rollouts.
  - WRITE grouped by (episode_id, rewrite_idx) -> compares the G' same-context candidates at one point.

Returns (raw_rewards, advantages) aligned to the flat `samples` list (matches
`train_data_conversion._post_process_rewards`). Set via `--custom-reward-post-process-path`.
"""
from __future__ import annotations

from collections import defaultdict


def _group_key(sample):
    md = sample.metadata or {}
    if md.get("phase") == "write":
        return ("write", md.get("episode_id"), md.get("rewrite_idx"))
    return ("act", sample.group_index, md.get("trial_pos"))


def _nomemory_reward(args, samples):
    """no-memory (full-history) branch: each Sample is ONE whole episode (a single continuous sequence)
    carrying `trial_spans` (response-space [start,end,trial_pos) segments) and `per_trial_scores` in
    metadata. We GRPO-normalize each trial_pos across sibling rollouts (group_index), then build a
    PER-TOKEN advantage list per sample (trial-k segment <- adv_k). Returns (raw_rewards, advantages)
    where each advantage is a per-token list (consumed by the per-token branch in advantages.py).
    """
    std_norm = getattr(args, "grpo_std_normalization", True)
    # per-sample scalar for logging/metrics = episode total score
    raw_rewards = [float(sum((s.metadata or {}).get("per_trial_scores") or [])) for s in samples]
    per_token = [[0.0] * int(s.response_length or 0) for s in samples]

    by_group: dict = defaultdict(list)
    for i, s in enumerate(samples):
        by_group[s.group_index].append(i)

    n_seg = 0
    for _g, idxs in by_group.items():
        trial_pos_set = set()
        for i in idxs:
            for (_s, _e, k) in ((samples[i].metadata or {}).get("trial_spans") or []):
                trial_pos_set.add(int(k))
        for k in trial_pos_set:
            scored = []
            for i in idxs:
                pts = (samples[i].metadata or {}).get("per_trial_scores") or []
                scored.append((i, float(pts[k]) if k < len(pts) else 0.0))
            vals = [v for (_i, v) in scored]
            n = len(vals)
            mean = sum(vals) / n
            denom = 1.0
            if std_norm and n > 1:
                var = sum((v - mean) ** 2 for v in vals) / n
                denom = (var ** 0.5) + 1e-6
            for (i, v) in scored:
                adv_k = (v - mean) / denom
                pt = per_token[i]
                for (s_, e_, kk) in ((samples[i].metadata or {}).get("trial_spans") or []):
                    if int(kk) == k:
                        for t in range(int(s_), min(int(e_), len(pt))):
                            pt[t] = adv_k
                        n_seg += 1

    mean_r = sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0
    print(f"[alchemy_advantage:no-memory] episodes={len(samples)} groups={len(by_group)} "
          f"segments={n_seg} mean_episode_score={mean_r:.3f}", flush=True)
    return raw_rewards, per_token


def reward_post_process(args, samples):
    """args, flat list[Sample] -> (raw_rewards, advantages), both len == len(samples)."""
    if samples and (samples[0].metadata or {}).get("no_memory"):
        return _nomemory_reward(args, samples)
    raw_rewards = [float(s.reward or 0.0) for s in samples]
    std_norm = getattr(args, "grpo_std_normalization", True)

    groups: dict = defaultdict(list)
    for i, s in enumerate(samples):
        groups[_group_key(s)].append(i)

    advantages = [0.0] * len(samples)
    for _key, idxs in groups.items():
        vals = [raw_rewards[i] for i in idxs]
        n = len(vals)
        mean = sum(vals) / n
        denom = 1.0
        if std_norm and n > 1:
            var = sum((v - mean) ** 2 for v in vals) / n
            denom = (var ** 0.5) + 1e-6
        for i in idxs:
            advantages[i] = (raw_rewards[i] - mean) / denom

    # lightweight logging: mean raw reward per stream
    act_r = [raw_rewards[i] for i, s in enumerate(samples) if (s.metadata or {}).get("phase") != "write"]
    wr_r = [raw_rewards[i] for i, s in enumerate(samples) if (s.metadata or {}).get("phase") == "write"]
    if act_r or wr_r:
        a = sum(act_r) / len(act_r) if act_r else 0.0
        w = sum(wr_r) / len(wr_r) if wr_r else 0.0
        print(f"[alchemy_advantage] ACT n={len(act_r)} mean_r={a:.3f} | "
              f"WRITE n={len(wr_r)} mean_r={w:.3f} | groups={len(groups)}", flush=True)

    return raw_rewards, advantages
