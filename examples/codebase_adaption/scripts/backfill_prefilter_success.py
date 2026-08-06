#!/usr/bin/env python3
"""Compute pre-filter success_frac from saved trajectories and backfill it into wandb.

The live metric `codebase_reward/success_frac_prefilter` (added 2026-08-05, computed in
codebase_rollout_filter before the zero-std group filter drops all-pass/all-fail groups)
only exists for runs launched after that date. This script reproduces the same number
offline from trajectory files and appends it to an existing wandb run, so historical
runs become comparable with live ones.

Metric definition (identical to the online version):
  per rollout, success_frac_prefilter = (# trials with outcomes[i].success == True)
                                        / (# trials with success not None)
  over ALL episodes of the rollout (2 groups x 8 siblings x 12 trials = 192 nominally;
  fewer if an episode aborted early). No zero-std filtering is applied.

The points are logged as {"rollout/step": k, "codebase_reward/success_frac_prefilter": v}
rows (no explicit step), relying on wandb's within-row pairing: set the workspace x-axis
to `rollout/step` and the curve lines up with the run's other metrics.

Usage:
  python3 backfill_prefilter_success.py \
    --traj-root /path/to/<run>/traj/train \
    --wandb-project miles-codebase-adaption \
    --wandb-run-id smith-4b-v3nocurr-gated \
    [--dry-run]        # compute + print only, no wandb writes
  Requires WANDB_API_KEY in the environment (e.g. `source ~/.wandb.env`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def compute_series(traj_root: str) -> dict[int, float]:
    series: dict[int, float] = {}
    rdirs = sorted(
        glob.glob(os.path.join(traj_root, "rollout_*")),
        key=lambda p: int(p.rsplit("_", 1)[1]),
    )
    for rdir in rdirs:
        rid = int(rdir.rsplit("_", 1)[1])
        succ = tot = 0
        for f in glob.glob(os.path.join(rdir, "*.json")):
            try:
                d = json.load(open(f))
            except Exception as exc:
                print(f"[warn] unreadable {f}: {exc}")
                continue
            if d.get("evaluation"):
                continue
            for o in d.get("outcomes") or []:
                if o.get("success") is None:
                    continue
                tot += 1
                succ += bool(o.get("success"))
        if tot:
            series[rid] = succ / tot
            print(f"r{rid}: {succ}/{tot} = {succ / tot:.4f}", flush=True)
    return series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-root", required=True, help="<run>/traj/train directory")
    ap.add_argument("--wandb-project", default="miles-codebase-adaption")
    ap.add_argument("--wandb-run-id", required=True, help="existing run id to resume/append")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    series = compute_series(args.traj_root)
    print(f"[done] {len(series)} rollouts computed")
    if args.dry_run or not series:
        return

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        resume="allow",
        settings=wandb.Settings(init_timeout=120),
    )
    for rid in sorted(series):
        run.log({"rollout/step": rid, "codebase_reward/success_frac_prefilter": series[rid]})
    run.finish()
    print(f"[done] {len(series)} points appended to {args.wandb_run_id}")


if __name__ == "__main__":
    main()
