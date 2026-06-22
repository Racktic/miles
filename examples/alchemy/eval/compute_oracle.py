"""Batch-compute the ideal-observer oracle per-trial scores for a range of episodes, ONCE.

`precomputed_maps.load_from_level_name` for the random-bottleneck level is very slow (it reconstructs
~GB of maps from protos in pure Python — minutes, not seconds) and dominates everything, so we pay it
exactly once per process and then loop all episodes reusing it. Results stream to a JSON cache keyed by
episode index (`oracle_cache.json`) so progress survives a kill/timeout and `finalize.py` can merge them
into the eval results later. Optionally pickles the loaded maps so a re-run skips the reconstruction.

  python examples/alchemy/eval/compute_oracle.py --episodes 0-9 --cache examples/alchemy/eval/oracle_cache.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/alchemy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # eval/

import numpy as _np
if not getattr(_np.stack, "_accepts_gen", False):     # numpy>=1.24 generator fix (see oracle.py)
    _o = _np.stack
    def _s(a, *x, **k):
        return _o(list(a) if isinstance(a, types.GeneratorType) else a, *x, **k)
    _s._accepts_gen = True
    _np.stack = _s

FIXED_LEVEL = "perceptual_mapping_randomized_with_random_bottleneck"


def _parse_episodes(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="0-9", help="e.g. '0-9' or '0,1,5'")
    ap.add_argument("--level", default=FIXED_LEVEL)
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "oracle_cache.json"))
    ap.add_argument("--minimise-world-states", action="store_true",
                    help="objective = minimise #world-states instead of maximise reward (faster, "
                         "but NOT the reward-optimal oracle — leave off for normalization).")
    ap.add_argument("--per-episode-timeout", type=int, default=0,
                    help="seconds before giving up on one episode's ideal-observer search (0=none).")
    args = ap.parse_args()

    episodes = _parse_episodes(args.episodes)
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)
    print(f"[oracle] level={args.level} episodes={episodes} cache={args.cache} "
          f"(already cached: {sorted(cache)})", flush=True)

    from env_alchemy import _load_dataset
    from dm_alchemy import symbolic_alchemy_bots
    from dm_alchemy import symbolic_alchemy_trackers as T
    from dm_alchemy.types import stones_and_potions

    # ---- one-time precomputed-map load (measured at ~7s; the ideal-observer SEARCH is the cost) ----
    from dm_alchemy.ideal_observer import precomputed_maps
    t = time.time()
    pc = precomputed_maps.load_from_level_name(args.level)
    print(f"[oracle] load_from_level_name took {time.time()-t:.1f}s", flush=True)

    class _Timeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _Timeout()
    if args.per_episode_timeout > 0:
        signal.signal(signal.SIGALRM, _alarm)

    rw = stones_and_potions.RewardWeights([1, 1, 1], 0, 12)

    def addt(env):
        env.add_trackers({
            T.AddMatrixEventTracker.NAME: T.AddMatrixEventTracker(),
            T.ScoreTracker.NAME: T.ScoreTracker(rw),
            T.BeliefStateTracker.NAME: T.BeliefStateTracker(pc, env, None)})

    dataset = _load_dataset(args.level)
    for idx in episodes:
        if str(idx) in cache:
            print(f"[oracle] ep {idx} already cached: {cache[str(idx)]}", flush=True)
            continue
        chem, items = dataset[idx]
        t = time.time()
        if args.per_episode_timeout > 0:
            signal.alarm(args.per_episode_timeout)
        try:
            res = symbolic_alchemy_bots.get_multi_trial_ideal_observer_reward(
                items, chem, rw, pc, args.minimise_world_states, addt)
        except _Timeout:
            print(f"[oracle] ep {idx}: TIMEOUT after {args.per_episode_timeout}s — skipped",
                  flush=True)
            continue
        finally:
            if args.per_episode_timeout > 0:
                signal.alarm(0)
        per_trial = [float(x) for x in res["score"]["per_trial"]]
        cache[str(idx)] = per_trial
        with open(args.cache, "w") as f:                # write after each episode (resumable)
            json.dump(cache, f, indent=2)
        print(f"[oracle] ep {idx}: {[round(x, 1) for x in per_trial]}  ({time.time()-t:.1f}s)",
              flush=True)

    print(f"[oracle] DONE. cache has {len(cache)} episodes -> {args.cache}", flush=True)


if __name__ == "__main__":
    main()
