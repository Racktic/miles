#!/usr/bin/env python3
"""Plot online (training-rollout) improvement vs step.
Per step = mean over the 64 rollout episodes of the oracle-normalized improvement
i_score = mean(norm[5:]) - mean(norm[:5]),  norm[t] = score[t]/oracle[t]  (oracle<=0 -> None),
exactly matching the offline finalize.py definition."""
import os, re, json, glob, statistics

ALC = "/home/qixinx/miles/examples/alchemy"
ORACLE = json.load(open(f"{ALC}/eval/oracle_cache.json"))

# label -> train-dump dir  (only those that exist are plotted)
RUNS = {
#   "actonly-w3 (β0)":          f"{ALC}/logs/qwen3-4b-curr950-actonly-w3-r120-e10-20260627/traj/train",
  "sig4norm-w3 co-train (β0)":f"{ALC}/logs/qwen3-4b-curr950-sig4norm-w3-r120-e10-20260627/traj/train",
  "sig4norm-w3-expl03 (β0.3)":f"{ALC}/logs/qwen3-4b-curr950-sig4norm-w3-expl03-r120-e10-20260628b/traj/train",
#   "actonly-w3-expl03 (β0.3)": f"{ALC}/logs/qwen3-4b-curr950-actonly-w3-expl03-r120-e10-20260628b/traj/train",
  "sig4normimprove win1":     f"{ALC}/logs/qwen3-4b-curr950-sig4normimprove-userdata-r120-e10-20260625-232325/traj/train",
}

_re_ep = re.compile(r'"episode_index":\s*(\d+)')
_re_pts = re.compile(r'"per_trial_scores":\s*\[(.*?)\]', re.S)

def ep_iscore(text):
    me, mp = _re_ep.search(text), _re_pts.search(text)
    if not (me and mp): return None
    ep = me.group(1)
    if ep not in ORACLE: return None
    scores = [float(x) for x in re.findall(r'-?\d+\.?\d*', mp.group(1))]
    oracle = ORACLE[ep]
    norm = [None if o <= 0 else s / o for s, o in zip(scores, oracle)]
    first = [x for x in norm[:5] if x is not None]
    last  = [x for x in norm[5:] if x is not None]
    return (sum(last)/len(last) - sum(first)/len(first)) if (first and last) else None

def run_curve(d):
    steps = {}
    for rd in glob.glob(os.path.join(d, "rollout_*")):
        m = re.search(r"rollout_(\d+)$", rd)
        if not m: continue
        step = int(m.group(1))
        vals = []
        for f in glob.glob(os.path.join(rd, "*.json")):
            v = ep_iscore(open(f, errors="ignore").read())
            if v is not None: vals.append(v)
        if vals: steps[step] = (statistics.mean(vals), len(vals))
    return dict(sorted(steps.items()))

curves = {}
for label, d in RUNS.items():
    if not os.path.isdir(d):
        print(f"[skip] {label}: no dir {d}"); continue
    c = run_curve(d)
    if c: curves[label] = c
    print(f"[ok] {label}: {len(c)} steps, n_ep/step≈{c[next(iter(c))][1] if c else 0}")

json.dump({k: {str(s): v[0] for s, v in c.items()} for k, c in curves.items()},
          open(f"{ALC}/notes/online_improve_curves.json", "w"), indent=1)

# ---- plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def roll(xs, ys, w=11):
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w // 2), min(len(ys), i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out

# raw (faint) + rolling-mean (bold)
plt.figure(figsize=(9.5, 5.8))
for label, c in curves.items():
    xs = sorted(c); ys = [c[s][0] for s in xs]
    line, = plt.plot(xs, roll(xs, ys), lw=2.2, label=label)
    plt.plot(xs, ys, lw=0.7, alpha=0.18, color=line.get_color())
plt.axhline(0, color="#888", lw=0.7, ls="--")
plt.xlabel("training step (rollout id)")
plt.ylabel("mean online improvement (oracle-norm, mean over 64 ep)")
plt.title("Online improvement vs step — bold = 11-step rolling mean, faint = raw")
plt.legend(fontsize=9); plt.grid(alpha=.25)
out = f"{ALC}/notes/online_improve_vs_step.png"
plt.tight_layout(); plt.savefig(out, dpi=130)
print("wrote", out)

# ---- numeric summary: mean improvement over step windows ----
def wmean(c, lo, hi):
    vs = [c[s][0] for s in c if lo <= s < hi]
    return statistics.mean(vs) if vs else float("nan")
print("\n=== mean online improvement by step window ===")
print(f"  {'run':<28}{'[0,20)':>9}{'[40,60)':>9}{'[80,100)':>10}{'[100,120)':>11}")
for label, c in curves.items():
    print(f"  {label:<28}{wmean(c,0,20):>9.3f}{wmean(c,40,60):>9.3f}{wmean(c,80,100):>10.3f}{wmean(c,100,120):>11.3f}")
print("curves json ->", f"{ALC}/notes/online_improve_curves.json")
