#!/usr/bin/env python3
"""Offline validation of the ACT exploration-reward judge on real trajectories.

Rebuilds (M_{k-1}, M_k) memory-delta pairs from saved episode trajectory JSONs and
scores them with codebase_judge.judge_explore (same prompt/model/parse as training
would use), then reports what the training signal would look like:
  - score / per-dimension distributions per rollout era (early vs mid vs late);
  - within-GRPO-group spread (8 siblings x same trial index): std and zero-std
    fraction — the shaped advantage is zero for zero-std groups, so this is the
    "does the signal carry gradient" check;
  - rough cost estimate per call and extrapolated per full training run.

Usage (from repo root, needs OPENAI_API_KEY or CODEBASE_JUDGE_API_KEY in env):
  python3 examples/codebase_adaption/scripts/validate_explore_judge.py \
    --traj-root /project/flame/qixinx/backups/smith-4b-v3nocurr-gated/traj/train \
    --rollouts 8 12 60 88 120 140 --trials 0 3 6 9 11 \
    --out /tmp/explore_judge_validation.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import codebase_judge  # noqa: E402

_DIMS = codebase_judge._DIMS


def collect_pairs(traj_root: str, rollouts: list[int], trials: list[int], groups_cap: int):
    """Yield dicts {rollout, group, sibling, k, prev, cur} for the sampled deltas."""
    pairs = []
    for rid in rollouts:
        files = sorted(glob.glob(os.path.join(traj_root, f"rollout_{rid}", "*.json")))
        if not files:
            print(f"[collect] rollout_{rid}: no files, skipped")
            continue
        by_group: dict[int, list[dict]] = defaultdict(list)
        for f in files:
            try:
                d = json.load(open(f))
            except Exception as exc:
                print(f"[collect] unreadable {f}: {exc}")
                continue
            if d.get("evaluation"):
                continue
            by_group[int(d["group_index"])].append(d)
        for g in sorted(by_group)[:groups_cap]:
            for sib, ep in enumerate(by_group[g]):
                summaries = ep.get("summaries") or []
                for k in trials:
                    if k >= len(summaries):
                        continue
                    pairs.append({
                        "rollout": rid,
                        "group": g,
                        "sibling": sib,
                        "k": k,
                        "prev": "" if k == 0 else (summaries[k - 1] or ""),
                        "cur": summaries[k] or "",
                    })
    return pairs


async def judge_all(pairs: list[dict]):
    results = await asyncio.gather(
        *[codebase_judge.judge_explore(p["prev"], p["cur"]) for p in pairs]
    )
    return results


def report(pairs: list[dict], results: list[dict | None]) -> None:
    rows = [dict(p, **r) for p, r in zip(pairs, results) if r is not None]
    failed = sum(1 for r in results if r is None)
    print(f"\n=== judged {len(rows)}/{len(pairs)} pairs ({failed} failures) ===")
    if not rows:
        return

    def stats(vals):
        vals = sorted(vals)
        n = len(vals)
        mean = sum(vals) / n
        return f"n={n} mean={mean:.3f} p10={vals[n // 10]:.2f} p50={vals[n // 2]:.2f} p90={vals[(9 * n) // 10]:.2f}"

    by_rollout: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_rollout[r["rollout"]].append(r)
    print("\n-- explore_score by rollout era --")
    for rid in sorted(by_rollout):
        rs = by_rollout[rid]
        print(f"rollout_{rid:>3}: {stats([r['explore_score'] for r in rs])}")
        dims = "  ".join(f"{d}={sum(r[d] for r in rs) / len(rs):.2f}" for d in _DIMS)
        print(f"             {dims}")

    print("\n-- within-GRPO-group spread (siblings x same trial k) --")
    zero_std_by_rollout: dict[int, list[float]] = defaultdict(list)
    stds_all = []
    for rid in sorted(by_rollout):
        groups: dict[tuple, list[float]] = defaultdict(list)
        for r in by_rollout[rid]:
            groups[(r["group"], r["k"])].append(r["explore_score"])
        stds = []
        zero = 0
        for vals in groups.values():
            if len(vals) < 2:
                continue
            m = sum(vals) / len(vals)
            std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            stds.append(std)
            if std < 1e-8:
                zero += 1
        if stds:
            stds_all.extend(stds)
            print(
                f"rollout_{rid:>3}: groups={len(stds)} mean_std={sum(stds) / len(stds):.3f} "
                f"zero_std_frac={zero / len(stds):.2f}"
            )
    if stds_all:
        print(f"overall    : mean_std={sum(stds_all) / len(stds_all):.3f}")

    # rough cost: chars/4 as input tokens + ~400 output tokens (JSON + minimal reasoning)
    sys_tokens = len(codebase_judge.SYSTEM_PROMPT) / 4
    in_tokens = sum(sys_tokens + (len(p["prev"]) + len(p["cur"])) / 4 for p in pairs)
    out_tokens = 400 * len(pairs)
    cost = in_tokens / 1e6 * 0.25 + out_tokens / 1e6 * 2.0  # gpt-5-mini list price
    per_call = cost / max(1, len(pairs))
    # full run: 191 rollouts x 16 episodes x ~12 deltas
    full = per_call * 191 * 16 * 12
    print(
        f"\n-- cost (rough, gpt-5-mini) --\nthis batch: ${cost:.2f} "
        f"({per_call * 100:.3f} cents/call) | full 191-rollout run: ~${full:.0f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-root", required=True)
    ap.add_argument("--rollouts", type=int, nargs="+", required=True)
    ap.add_argument("--trials", type=int, nargs="+", default=[0, 3, 6, 9, 11])
    ap.add_argument("--groups-cap", type=int, default=1, help="groups per rollout")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry-run", action="store_true", help="collect + count only, no API calls")
    args = ap.parse_args()

    pairs = collect_pairs(args.traj_root, args.rollouts, args.trials, args.groups_cap)
    print(f"[collect] {len(pairs)} memory-delta pairs sampled")
    if args.dry_run:
        return
    results = asyncio.run(judge_all(pairs))
    report(pairs, results)
    if args.out:
        with open(args.out, "w") as fh:
            for p, r in zip(pairs, results):
                fh.write(json.dumps(dict(p, judge=r)) + "\n")
        print(f"[out] rows written to {args.out}")


if __name__ == "__main__":
    main()
