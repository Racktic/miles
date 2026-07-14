"""Episode-level GRPO advantage for ARC-AGI-3 memory rollouts.

Each episode is exploded into one training sample per turn (see ``arc_rollout.py``). The standard
miles reward post-processing assumes exactly ``n_samples_per_prompt`` samples per group and reshapes
``(-1, n_samples_per_prompt)`` — which is wrong here because episodes have variable turn counts.

This custom ``--custom-reward-post-process-path`` instead:
  1. recovers the per-episode reward (= episode ``levels_completed``, identical across a turn's siblings),
  2. groups episodes by game (``group_index``), whitens the episode rewards within the game
     (GRPO: subtract group mean, optionally divide by group std),
  3. broadcasts each episode's advantage to all of that episode's turn-samples.

Returns ``(raw_rewards, advantages)`` aligned to the flat ``samples`` list, matching
``train_data_conversion._post_process_rewards``.
"""
from __future__ import annotations

from collections import defaultdict


def _episode_reward(sample) -> float:
    md = sample.metadata or {}
    if "arc_levels" in md:
        return float(md["arc_levels"])
    return float(sample.reward or 0.0)


def _episode_key(sample):
    md = sample.metadata or {}
    return md.get("episode_id", (sample.group_index, sample.index))


def reward_post_process(args, samples):
    """args, flat list[Sample] -> (raw_rewards, advantages), both len == len(samples)."""
    raw_rewards = [_episode_reward(s) for s in samples]

    # game (group_index) -> {episode_key: episode_reward}
    games: dict = defaultdict(dict)
    for s in samples:
        games[s.group_index][_episode_key(s)] = _episode_reward(s)

    std_norm = getattr(args, "grpo_std_normalization", True)

    # (group_index, episode_key) -> advantage
    adv: dict = {}
    for g, episodes in games.items():
        vals = list(episodes.values())
        n = len(vals)
        mean = sum(vals) / n
        if std_norm and n > 1:
            var = sum((v - mean) ** 2 for v in vals) / n
            denom = (var ** 0.5) + 1e-6
        else:
            denom = 1.0
        for ep_key, v in episodes.items():
            adv[(g, ep_key)] = (v - mean) / denom

    advantages = [adv[(s.group_index, _episode_key(s))] for s in samples]
    return raw_rewards, advantages
