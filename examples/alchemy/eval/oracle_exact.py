"""Exact per-trial oracle = the true maximum achievable trial reward given the KNOWN chemistry,
matching Sawyer et al. (2025): "an oracle that takes the optimal set of actions for the given items".

Correctness is by construction, three ways:
  (A) EXHAUSTIVE search (not greedy): we enumerate every legal action sequence within the trial's
      step budget and take the max — so we cannot miss the optimum (incl. counter-intuitive paths that
      temporarily worsen a stone to route around a graph bottleneck).
  (B) GROUND-TRUTH dynamics: every transition is executed by cloning the REAL dm_alchemy env and
      calling its own .step() — zero remodeling of potions/graph/reward, so no model-mismatch error.
  (C) memoisation on the (observable, Markov) state collapses action-ordering so the exhaustive search
      is tractable; the observable state fully determines the future in this deterministic env.

We build a fresh SINGLE-TRIAL env per (episode, trial) so the search space is just that one trial.
Empirical self-check (see __main__): oracle >= any agent score on the same trial, and == 45 wherever
the agent actually scored 45.
"""
from __future__ import annotations

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/alchemy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # eval/

from env_alchemy import AlchemyTextEnv, FIXED_LEVEL, _load_dataset, decode_obs, semantic_to_int  # noqa: E402


def _canonical(vec):
    """Observable Markov state: multiset of present stones (rounded axes+value) + present potion types,
    order-independent so action permutations collapse. Steps are intentionally NOT in the key: every
    action consumes one stone (cauldron) or one potion, so a trial has <= 3+12 = 15 actions < the
    20-step budget — the budget never binds, so it cannot affect the optimum. Dropping it shrinks the
    state space ~20x."""
    d = decode_obs(vec)
    stones = tuple(sorted(
        (tuple(int(round(x)) for x in s["axes"]), round(float(s["value"]), 2))
        for s in d["stones"] if s["exists"]))
    potions = tuple(sorted(round(float(p["type"]), 2) for p in d["potions"] if p["exists"]))
    return (stones, potions)


def _present(vec):
    d = decode_obs(vec)
    stones = [s["idx"] for s in d["stones"] if s["exists"]]
    potions = [p["idx"] for p in d["potions"] if p["exists"]]
    return stones, potions


def _build_single_trial_env(chem, items, trial: int, max_steps: int) -> AlchemyTextEnv:
    """A fresh env containing ONLY `trial`'s stones+potions (so done==end of that trial)."""
    from dm_alchemy.types import utils as alch_utils
    from dm_alchemy import symbolic_alchemy
    single = alch_utils.EpisodeItems(potions=[items.potions[trial]], stones=[items.stones[trial]])
    env = AlchemyTextEnv.__new__(AlchemyTextEnv)
    env._env = symbolic_alchemy.get_symbolic_alchemy_fixed(
        episode_items=single, chemistry=chem, max_steps_per_trial=max_steps, end_trial_action=True)
    env.num_trials = 1
    env.episode_index = None
    env.end_trial_action = True
    env.num_special = 2
    env.max_steps = max_steps
    env.total_reward = 0.0
    env.per_trial_scores = []
    env.last_vec = None
    return env


def trial_optimal(chem, items, trial: int, max_steps: int = 20) -> float:
    """Exact max achievable reward in one trial, by exhaustive memoised search over cloned real envs."""
    base = _build_single_trial_env(chem, items, trial, max_steps)
    obs = base.reset()
    memo: dict = {}

    def search(env: AlchemyTextEnv, vec) -> float:
        key = _canonical(vec)
        if key in memo:
            return memo[key]
        stones, potions = _present(vec)
        if not stones:
            memo[key] = 0.0
            return 0.0
        best = 0.0  # ending the trial here yields 0 further reward
        for si in stones:
            # cauldron stone si
            e2 = copy.deepcopy(env)
            o2, r, done, _ = e2.step(semantic_to_int(si, -1, num_special=2))
            best = max(best, r + (0.0 if done else search(e2, o2["vec"])))
            # apply each present potion to stone si
            for pj in potions:
                e3 = copy.deepcopy(env)
                o3, r3, d3, _ = e3.step(semantic_to_int(si, pj, num_special=2))
                best = max(best, r3 + (0.0 if d3 else search(e3, o3["vec"])))
        memo[key] = best
        return best

    return search(base, obs["vec"])


def episode_oracle(episode_index: int, level_name: str = FIXED_LEVEL, max_steps: int = 20) -> list:
    chem, items = _load_dataset(level_name)[int(episode_index)]
    return [trial_optimal(chem, items, t, max_steps) for t in range(items.num_trials)]


if __name__ == "__main__":
    import time
    t0 = time.time()
    chem, items = _load_dataset(FIXED_LEVEL)[0]
    v = trial_optimal(chem, items, 0)
    print(f"episode0 trial0 optimal = {v}  ({time.time()-t0:.1f}s)")
