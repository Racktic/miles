#!/usr/bin/env python3
"""Backfill ONLY alchemy_online/turns_trial{k} into an EXISTING wandb run, on the rollout/step axis.
Matches the live _improve_and_turn_metrics turns logic (each train dump = one episode instance;
count turns per trial from dump['turns']; mean over instances).

SAFETY: uses resume='must' + an explicit WANDB_ID -> if that run id does NOT exist, wandb ERRORS
(it will NEVER create a new/garbage run). Does NOT touch improve_mean/norm_improve.

Args:  LOCAL_RUN (traj dir name)   MAX_STEP   WANDB_ID (the REAL target run id)   [--dry]
"""
import os, sys, json, glob
from collections import defaultdict
import wandb

ALC = "/home/qixinx/miles/examples/alchemy"
LOCAL_RUN = sys.argv[1]
# arg2 = "N" (0..N) 或 "A-B" (A..B,含端点) —— 用于 resume 段只传子区间
_rng = sys.argv[2]
if "-" in _rng:
    MIN_STEP, MAX_STEP = (int(x) for x in _rng.split("-", 1))
else:
    MIN_STEP, MAX_STEP = 0, int(_rng)
WANDB_ID = sys.argv[3]
DRY = "--dry" in sys.argv[4:]
T = f"{ALC}/logs/{LOCAL_RUN}/traj/train"
def mean(xs): return sum(xs) / len(xs) if xs else float("nan")

rollouts = sorted(glob.glob(f"{T}/rollout_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
rollouts = [rd for rd in rollouts if MIN_STEP <= int(rd.rsplit("_", 1)[1]) <= MAX_STEP]

# compute per-step turns first (no wandb yet)
plan = []
for rd in rollouts:
    step = int(rd.rsplit("_", 1)[1])
    tbt = defaultdict(list)
    for f in glob.glob(f"{rd}/*.json"):
        d = json.load(open(f))
        tpt = defaultdict(int)
        for rec in d.get("turns", []):
            tpt[int(rec.get("trial", 0))] += 1
        for t, c in tpt.items():
            tbt[t].append(c)
    log = {"rollout/step": step}
    for t in sorted(tbt):
        log[f"alchemy_online/turns_trial{t}"] = mean(tbt[t])
    if len(log) > 1:
        plan.append(log)

print(f"[plan] LOCAL_RUN={LOCAL_RUN}  steps_to_log={len(plan)}  target_wandb_id={WANDB_ID}")
if plan:
    s0 = plan[0]
    print(f"  e.g. step {s0['rollout/step']}: trial0={s0.get('alchemy_online/turns_trial0'):.2f} "
          f"trial9={s0.get('alchemy_online/turns_trial9', float('nan')):.2f}")
if DRY:
    print("  (dry-run: nothing written)")
    sys.exit(0)

# resume='must' => errors out if WANDB_ID doesn't exist; NEVER creates a new run
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "miles-alchemy"), id=WANDB_ID, resume="must")
wandb.define_metric("rollout/step")
wandb.define_metric("alchemy_online/*", step_metric="rollout/step")
for log in plan:
    wandb.log(log)
wandb.finish()
print(f"[done] backfilled turns to {WANDB_ID}: {len(plan)} steps -> {getattr(run,'url','(check wandb)')}")
