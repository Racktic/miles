#!/usr/bin/env python3
"""Backfill online improve+turns into an EXISTING wandb run (resume=allow), on rollout/step.
Run ONLY while training is stopped (no concurrent writer). Args: RUN_ID  MAX_STEP  [WANDB_ID].
Logs alchemy_score/improve_mean + alchemy_online/turns_trial{0..9} for rollout steps 0..MAX_STEP,
matching the live code (dedupe per instance; oracle-normalized improvement)."""
import os, sys, json, glob
from collections import defaultdict
import wandb

ALC = "/home/qixinx/miles/examples/alchemy"
RUN = sys.argv[1]
MAX_STEP = int(sys.argv[2])
WANDB_ID = sys.argv[3] if len(sys.argv) > 3 else RUN   # training uses ALCHEMY_WANDB_RUN_ID == RUN
T = f"{ALC}/logs/{RUN}/traj/train"
OC = json.load(open(f"{ALC}/eval/oracle_cache.json"))
def mean(xs): return sum(xs) / len(xs) if xs else float("nan")

run = wandb.init(project=os.environ.get("WANDB_PROJECT", "miles-alchemy"), id=WANDB_ID, resume="allow")
wandb.define_metric("rollout/step")
wandb.define_metric("alchemy_score/*", step_metric="rollout/step")
wandb.define_metric("alchemy_online/*", step_metric="rollout/step")

rollouts = sorted(glob.glob(f"{T}/rollout_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
rollouts = [rd for rd in rollouts if int(rd.rsplit("_", 1)[1]) <= MAX_STEP]
print(f"backfill {len(rollouts)} steps (0..{MAX_STEP}) into wandb id={WANDB_ID}")
for rd in rollouts:
    step = int(rd.rsplit("_", 1)[1])
    improves, tbt = [], defaultdict(list)
    for f in glob.glob(f"{rd}/*.json"):
        d = json.load(open(f))
        pts = d.get("per_trial_scores") or []
        epi = d.get("episode_index")
        oracle = OC.get(str(int(epi))) if epi is not None else None
        if oracle and pts:
            norm = [None if o <= 0 else a / o for a, o in zip(pts, oracle)]
            first = [x for x in norm[:5] if x is not None]; last = [x for x in norm[5:] if x is not None]
            if first and last: improves.append(sum(last)/len(last) - sum(first)/len(first))
        tpt = defaultdict(int)
        for rec in d.get("turns", []): tpt[int(rec.get("trial", 0))] += 1
        for t, c in tpt.items(): tbt[t].append(c)
    log = {"rollout/step": step}
    if improves: log["alchemy_score/improve_mean"] = mean(improves)
    for t in sorted(tbt): log[f"alchemy_online/turns_trial{t}"] = mean(tbt[t])
    wandb.log(log)
wandb.finish()
print("backfill done ->", getattr(run, "url", "(check wandb)"))
