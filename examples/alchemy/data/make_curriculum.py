#!/usr/bin/env python3
"""Build a difficulty-ordered (easy->hard) training jsonl for curriculum learning.

Curriculum = method A (static ordering): episodes sorted by graph-edge count DESCENDING
(12=trivial -> 7=hardest); train with `--rollout-shuffle` OFF so the data source feeds them in
this order. Within an edge tier the order is shuffled with a fixed seed (so a step's batch isn't
biased by episode_index). Eval episodes (hard_set_20) are EXCLUDED to avoid train/eval leakage.

Each line matches the existing train jsonl schema exactly:
  {"prompt": [{"role":"user","content":[{"type":"text","text":"Symbolic Alchemy, prebuilt episode N."}]}],
   "metadata": {"episode_index": N, "max_steps_per_trial": 20, "rewrite_granularity": "trial"}}
The prompt is a placeholder; the real observation is rebuilt deterministically from episode_index
in the rollout, so any episode 0..999 can be emitted without running the env.

Usage:
  python make_curriculum.py --n 980                 # all non-eval episodes (default)
  python make_curriculum.py --n 950                 # stratified subsample to 950 (keeps tier ratios)
"""
import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
TIER_ORDER = [12, 10, 9, 8, 7]  # easy -> hard (descending edges)
LEG = {7: "hard", 8: "med-hard", 9: "medium", 10: "easy-ish", 12: "trivial"}


def make_row(ep: int) -> dict:
    return {
        "prompt": [{"role": "user", "content": [
            {"type": "text", "text": f"Symbolic Alchemy, prebuilt episode {ep}."}]}],
        "metadata": {"episode_index": ep, "max_steps_per_trial": 20, "rewrite_granularity": "trial"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=980, help="how many episodes to keep (<=980)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "alchemy_train_curriculum.jsonl"))
    args = ap.parse_args()

    edges = json.load(open(os.path.join(HERE, "difficulty_index.json")))["edges_by_episode"]
    eval_eps = set(json.load(open(os.path.join(HERE, "hard_set_20.json")))["episodes"])

    # candidates = all 1000 minus eval set, grouped by tier
    by_tier = {t: [] for t in TIER_ORDER}
    for ep_str, e in edges.items():
        ep = int(ep_str)
        if ep in eval_eps:
            continue
        by_tier[e].append(ep)
    total = sum(len(v) for v in by_tier.values())
    assert args.n <= total, f"--n {args.n} > available {total}"

    rng = random.Random(args.seed)
    for t in TIER_ORDER:
        rng.shuffle(by_tier[t])  # shuffle within tier (fixed seed)

    # stratified subsample to n (proportional per tier), preserving difficulty ratios
    keep = {t: round(args.n * len(by_tier[t]) / total) for t in TIER_ORDER}
    # fix rounding drift to hit exactly n
    drift = args.n - sum(keep.values())
    for t in sorted(TIER_ORDER, key=lambda t: -len(by_tier[t])):
        if drift == 0:
            break
        step = 1 if drift > 0 else -1
        keep[t] += step
        drift -= step
    sel = {t: by_tier[t][: keep[t]] for t in TIER_ORDER}

    # write easy -> hard
    with open(args.out, "w") as f:
        for t in TIER_ORDER:
            for ep in sel[t]:
                f.write(json.dumps(make_row(ep)) + "\n")

    n_written = sum(len(sel[t]) for t in TIER_ORDER)
    print(f"wrote {n_written} episodes -> {args.out}")
    print(f"order: easy->hard {TIER_ORDER}")
    for t in TIER_ORDER:
        print(f"  edges={t} ({LEG[t]}): {len(sel[t])}")
    print(f"excluded eval episodes: {sorted(eval_eps)}")
    # leakage check
    written = set()
    for line in open(args.out):
        written.add(json.loads(line)["metadata"]["episode_index"])
    overlap = written & eval_eps
    print(f"leakage check (train ∩ eval): {sorted(overlap) if overlap else 'NONE ✓'}")


if __name__ == "__main__":
    main()
