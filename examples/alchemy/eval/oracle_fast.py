"""Fast exact per-trial oracle — same exhaustive optimum as oracle_exact.py, but ~1000x faster.

oracle_exact clones the whole dm_alchemy env (deepcopy ~ms) and calls env.step for EVERY (search-node,
action) — recomputing the same fixed transition thousands of times. Here we exploit that, within an
episode, the dynamics are a tiny deterministic system read straight off the dataset objects:
  - each Stone exposes its true LATENT coords  (`Stone.latent`, a vertex of {-1,1}^3),
  - each Potion exposes its true LATENT effect (`Potion.dimension`, `Potion.direction`),
  - the chemistry graph (with bottleneck) is `graphs.convert_graph_to_adj_mat(chem.graph)` (8x8).
A potion (dim,dir) applied to a stone at coords c: if c[dim] != dir AND the graph has the edge
c<->c' (c' = c flipped at dim), the stone moves to c'; else no change. Cauldron reward = RewardWeights(c).
The exhaustive search then runs in pure Python on (multiset of stone coords, multiset of potion (dim,dir))
with one transition table per episode — identical optimum, no env, no deepcopy.

VALIDATED against oracle_exact's cached values (oracle_cache.json) before use — `python oracle_fast.py`.
"""
from __future__ import annotations

import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/alchemy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # eval/

from env_alchemy import _load_dataset, FIXED_LEVEL  # noqa: E402


@functools.lru_cache(maxsize=1)
def _reward_weights():
    from dm_alchemy.types import stones_and_potions as sp
    return sp.RewardWeights([1, 1, 1], 0, 12)


@functools.lru_cache(maxsize=16)
def _coord_index_map():
    """coords-tuple (in {-1,1}^3) -> graph adjacency-matrix index, matching convert_graph_to_adj_mat."""
    from dm_alchemy.types import stones_and_potions as sp
    m = {}
    for c in [(a, b, d) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]:
        m[c] = int(sp.LatentStone(np.array(c)).index())
    return m


def _episode_adj(chem):
    from dm_alchemy.types import graphs
    return np.asarray(graphs.convert_graph_to_adj_mat(chem.graph))


def trial_optimal_fast(chem, items, trial: int, adj=None) -> float:
    rw = _reward_weights()
    idx = _coord_index_map()
    if adj is None:
        adj = _episode_adj(chem)
    stones0 = tuple(sorted(tuple(int(x) for x in s.latent) for s in items.stones[trial]))
    potions0 = tuple(sorted((int(p.dimension), int(p.direction)) for p in items.potions[trial]))

    def move(c, dim, direction):
        if c[dim] == direction:
            return c                                  # already at target end -> no change
        c2 = c[:dim] + (direction,) + c[dim + 1:]
        return c2 if adj[idx[c]][idx[c2]] else c      # graph edge gates the transition (bottleneck)

    def reward(c):
        return float(rw(list(c)))

    memo: dict = {}

    def search(stones, potions):
        if not stones:
            return 0.0
        key = (stones, potions)
        if key in memo:
            return memo[key]
        best = 0.0                                     # ending the trial here yields 0 further reward
        for i, c in enumerate(stones):
            rest = stones[:i] + stones[i + 1:]
            best = max(best, reward(c) + search(rest, potions))     # cauldron stone c
            tried = set()
            for j, (dim, dr) in enumerate(potions):
                if (dim, dr) in tried:
                    continue                           # same-type potions are interchangeable
                tried.add((dim, dr))
                c2 = move(c, dim, dr)
                ns = tuple(sorted(rest + (c2,)))
                npot = tuple(sorted(potions[:j] + potions[j + 1:]))
                best = max(best, search(ns, npot))
        memo[key] = best
        return best

    return search(stones0, potions0)


def episode_oracle_fast(episode_index: int, level_name: str = FIXED_LEVEL) -> list:
    chem, items = _load_dataset(level_name)[int(episode_index)]
    adj = _episode_adj(chem)
    return [trial_optimal_fast(chem, items, t, adj) for t in range(items.num_trials)]


def _validate_against_cache(cache_path: str):
    """Compute fast oracle for every episode already in the cache and compare — must match exactly."""
    import json
    import time
    cache = json.load(open(cache_path))
    eps = sorted(int(k) for k in cache)
    print(f"[validate] {len(eps)} cached episodes from {cache_path}", flush=True)
    t0 = time.time()
    n_trials = 0
    mism = []
    for ep in eps:
        got = episode_oracle_fast(ep)
        exp = cache[str(ep)]
        n_trials += len(got)
        for t, (g, e) in enumerate(zip(got, exp)):
            if abs(g - e) > 1e-9:
                mism.append((ep, t, g, e))
    dt = time.time() - t0
    print(f"[validate] {n_trials} trials in {dt:.1f}s ({1000*dt/max(1,n_trials):.1f} ms/trial)", flush=True)
    if mism:
        print(f"[validate] *** {len(mism)} MISMATCHES *** (first 20):", flush=True)
        for ep, t, g, e in mism[:20]:
            print(f"    ep{ep} trial{t}: fast={g} vs exact={e}", flush=True)
    else:
        print(f"[validate] ALL {n_trials} trials MATCH the exact oracle ✓", flush=True)
    return not mism


if __name__ == "__main__":
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "oracle_cache.json")
    ok = _validate_against_cache(cache)
    sys.exit(0 if ok else 1)
