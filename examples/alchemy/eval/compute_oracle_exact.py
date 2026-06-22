"""Batch-compute the EXACT per-trial oracle (oracle_exact.trial_optimal) over many episodes, in
parallel across (episode, trial) tasks. Each trial is independent, so this is embarrassingly parallel.
Results stream to a JSON cache keyed by episode index ({ "<ep>": [10 per-trial optima] }), the same
format finalize.py consumes; the cache is resumable (already-done trials are skipped).

  python examples/alchemy/eval/compute_oracle_exact.py --episodes 0-19 --workers 12 \
      --cache examples/alchemy/eval/oracle_cache.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # examples/alchemy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # eval/

from env_alchemy import FIXED_LEVEL, _load_dataset  # noqa: E402

LEVEL = FIXED_LEVEL


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


def _work(task):
    """Worker: compute one trial's exact optimum. Dataset load is lru_cached per process.
    task = (ep, trial, backend); backend 'fast' uses oracle_fast (pure-python, ~1000x faster per node),
    'exact' uses oracle_exact (clone-env, ground truth). Both give the identical optimum."""
    ep, trial, backend = task
    chem, items = _load_dataset(LEVEL)[ep]
    t0 = time.time()
    if backend == "fast":
        from oracle_fast import trial_optimal_fast
        val = trial_optimal_fast(chem, items, trial)
    else:
        from oracle_exact import trial_optimal
        val = trial_optimal(chem, items, trial)
    return ep, trial, float(val), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="0-19")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--backend", choices=["exact", "fast"], default="exact",
                    help="fast: pure-python oracle_fast (~1000x faster/node); exact: clone-env oracle_exact.")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "oracle_cache.json"))
    args = ap.parse_args()

    episodes = _parse_episodes(args.episodes)
    n_trials = {ep: _load_dataset(LEVEL)[ep][1].num_trials for ep in episodes}

    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)

    # build task list, skipping trials already fully cached
    tasks = []
    for ep in episodes:
        done = cache.get(str(ep))
        for t in range(n_trials[ep]):
            if done is not None and t < len(done) and done[t] is not None:
                continue
            tasks.append((ep, t, args.backend))
    # partial holders so we can write per-episode rows as they complete
    partial = {ep: list(cache.get(str(ep), [None] * n_trials[ep])) for ep in episodes}
    for ep in episodes:
        if len(partial[ep]) < n_trials[ep]:
            partial[ep] += [None] * (n_trials[ep] - len(partial[ep]))

    print(f"[oracle-exact] {len(tasks)} trials to compute over {len(episodes)} episodes "
          f"with {args.workers} workers", flush=True)
    t_start = time.time()
    done_count = 0
    with Pool(args.workers) as pool:
        for ep, trial, val, dt in pool.imap_unordered(_work, tasks):
            partial[ep][trial] = val
            done_count += 1
            # flush a fully-finished episode into the cache file
            if all(x is not None for x in partial[ep]):
                cache[str(ep)] = partial[ep]
                with open(args.cache, "w") as f:
                    json.dump(cache, f, indent=2)
            print(f"[oracle-exact] ep{ep} trial{trial} = {val:.0f}  ({dt:.0f}s)  "
                  f"[{done_count}/{len(tasks)}]", flush=True)

    # final flush (in case some episodes weren't fully complete above)
    for ep in episodes:
        if all(x is not None for x in partial[ep]):
            cache[str(ep)] = partial[ep]
    with open(args.cache, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"[oracle-exact] DONE {done_count} trials in {time.time()-t_start:.0f}s -> {args.cache}",
          flush=True)


if __name__ == "__main__":
    main()
