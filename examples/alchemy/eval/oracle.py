"""Normalized-to-oracle scoring for the Alchemy eval (TMLR §4.2 口径).

A trial's normalized score = agent_trial_score / oracle_optimal_trial_score. The oracle is
dm_alchemy's *ideal observer* — a chemistry-known agent that plays each trial optimally
(`get_multi_trial_ideal_observer_reward`). reward_weights match the env default
(`RewardWeights([1,1,1], 0, bonus=12)`; best stone = 3*1 + 12 = 15) so agent and oracle are on the
same scale. Oracle scores are cached per episode (the search is the expensive part).
"""
from __future__ import annotations

import functools
import os
import sys
import types

# dm_alchemy's vendored `stones_and_potions.py` calls `np.stack((gen for ...))`, which numpy>=1.24
# rejects (generators must be materialised). Wrap np.stack to accept generators rather than editing
# the package inside site-packages. Idempotent; behaviour for non-generators is unchanged.
import numpy as _np  # noqa: E402

if not getattr(_np.stack, "_accepts_gen", False):
    _orig_stack = _np.stack

    def _stack_accepting_gen(arrays, *a, **k):
        if isinstance(arrays, types.GeneratorType):
            arrays = list(arrays)
        return _orig_stack(arrays, *a, **k)

    _stack_accepting_gen._accepts_gen = True
    _np.stack = _stack_accepting_gen

FIXED_LEVEL = "perceptual_mapping_randomized_with_random_bottleneck"

# Make ``import env_alchemy`` work regardless of cwd (examples/alchemy is the parent dir).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@functools.lru_cache(maxsize=4)
def _precomputed(level_name: str = FIXED_LEVEL):
    """Load (and cache) the ideal-observer precomputed maps for a level — slow first call."""
    from dm_alchemy.ideal_observer import precomputed_maps
    return precomputed_maps.load_from_level_name(level_name)


@functools.lru_cache(maxsize=1)
def _reward_weights():
    from dm_alchemy.types import stones_and_potions
    return stones_and_potions.RewardWeights([1, 1, 1], 0, 12)  # matches env default


def _add_trackers(env, reward_weights, precomputed):
    """Attach the trackers the IdealObserverBot needs (score + belief-state + matrix events)."""
    from dm_alchemy import symbolic_alchemy_trackers as T
    env.add_trackers({
        T.AddMatrixEventTracker.NAME: T.AddMatrixEventTracker(),
        T.ScoreTracker.NAME: T.ScoreTracker(reward_weights),
        T.BeliefStateTracker.NAME: T.BeliefStateTracker(precomputed, env, None),
    })


@functools.lru_cache(maxsize=4096)
def oracle_per_trial(episode_index: int, level_name: str = FIXED_LEVEL) -> tuple:
    """Per-trial optimal (oracle) reward for one prebuilt episode. Cached per episode."""
    from env_alchemy import _load_dataset
    from dm_alchemy import symbolic_alchemy_bots
    chem, items = _load_dataset(level_name)[int(episode_index)]
    rw, pc = _reward_weights(), _precomputed(level_name)
    results = symbolic_alchemy_bots.get_multi_trial_ideal_observer_reward(
        items, chem, rw, pc, minimise_world_states=False,
        add_trackers_to_env=functools.partial(_add_trackers, reward_weights=rw, precomputed=pc))
    return tuple(float(x) for x in results["score"]["per_trial"])


def normalize(agent_per_trial, oracle_scores):
    """agent/oracle per trial; oracle<=0 -> None (excluded from means, as the ratio is undefined)."""
    out = []
    for a, o in zip(agent_per_trial, oracle_scores):
        out.append(None if o <= 0 else float(a) / float(o))
    return out


if __name__ == "__main__":  # quick sanity: python examples/alchemy/eval/oracle.py
    op = oracle_per_trial(0)
    print("oracle_per_trial(episode 0):", op)
    assert len(op) == 10 and all(o >= 0 for o in op), op
    print("normalize(oracle, oracle):", normalize(op, op))   # all ~1.0 (or None for 0-oracle trials)
    print("oracle.py OK")
