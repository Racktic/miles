#!/usr/bin/env python3
"""Plot per-trial turn-count vs training step.
For each trial k (0..9): y = mean over the 64 rollout episodes of (#turns in trial k), x = step.
One figure per k (10 figures) + a 2x5 overview grid. Config = RUNS only."""
import os, re, json, glob, statistics

ALC = "/home/qixinx/miles/examples/alchemy"
NTRIALS = 10

# label -> train-dump dir  (only existing dirs are plotted)
RUNS = {
#   "ACT-only · window=1 · β=0":                       f"{ALC}/logs/qwen3-4b-curr950-actonly-r120-e10-20260620-014759/traj/train",
#   "ACT-only · window=3 · β=0":                       f"{ALC}/logs/qwen3-4b-curr950-actonly-w3-r120-e10-20260627/traj/train",
  "ACT-only · window=3 · β=0.3 (explore)":           f"{ALC}/logs/qwen3-4b-curr950-actonly-w3-expl03-r120-e10-20260628b/traj/train",
#   "ACT+WRITE(norm_improve) · window=1 · β=0":        f"{ALC}/logs/qwen3-4b-curr950-sig4normimprove-userdata-r120-e10-20260625-232325/traj/train",
#   "ACT+WRITE(norm_improve) · window=3 · β=0":        f"{ALC}/logs/qwen3-4b-curr950-sig4norm-w3-r120-e10-20260627/traj/train",
  "ACT+WRITE(norm_improve) · window=3 · β=0.3 (explore)": f"{ALC}/logs/qwen3-4b-curr950-sig4norm-w3-expl03-r120-e10-20260628b/traj/train",
  "ACT-only · window=3 · β=0.3 (explore) · budget": f"{ALC}/logs/qwen3-4b-curr950-actonly-w3-expl03-budgetv3-r120-e10-20260630/traj/train",
  "ACT+WRITE(norm_improve) · window=3 · β=0.3 (explore) · budget": f"{ALC}/logs/qwen3-4b-curr950-sig4norm-w3-expl03-budgetv3-r120-e10-20260630b/traj/train",
}

_re_trial = re.compile(r'"trial":\s*(\d+)')   # one per turn (turn.trial); safe vs other keys

def ep_turns_per_trial(text):
    """Return dict trial -> #turns for one episode dump (count by the per-turn `trial` field)."""
    cnt = {}
    for t in _re_trial.findall(text):
        k = int(t); cnt[k] = cnt.get(k, 0) + 1
    return cnt

def run_curves(d):
    # step -> trial -> [counts over episodes]
    steps = {}
    for rd in glob.glob(os.path.join(d, "rollout_*")):
        m = re.search(r"rollout_(\d+)$", rd)
        if not m: continue
        step = int(m.group(1))
        per_trial = {k: [] for k in range(NTRIALS)}
        for f in glob.glob(os.path.join(rd, "*.json")):
            c = ep_turns_per_trial(open(f, errors="ignore").read())
            for k in range(NTRIALS):
                if k in c: per_trial[k].append(c[k])
        steps[step] = {k: (statistics.mean(v) if v else None) for k, v in per_trial.items()}
    return dict(sorted(steps.items()))

curves = {}
for label, d in RUNS.items():
    if not os.path.isdir(d): print(f"[skip] {label}: no dir"); continue
    curves[label] = run_curves(d)
    print(f"[ok] {label}: {len(curves[label])} steps")

# save data
json.dump({lab: {str(s): {str(k): tv for k, tv in kv.items()} for s, kv in c.items()} for lab, c in curves.items()},
          open(f"{ALC}/notes/turns_per_trial_curves.json", "w"))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def roll(ys, w=11):
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - w // 2), min(len(ys), i + w // 2 + 1)
        seg = [y for y in ys[lo:hi] if y is not None]
        out.append(sum(seg) / len(seg) if seg else None)
    return out

def plot_trial(ax, k):
    for label, c in curves.items():
        xs = [s for s in sorted(c) if c[s][k] is not None]
        ys = [c[s][k] for s in xs]
        if not xs: continue
        line, = ax.plot(xs, roll(ys), lw=2.0, label=label)
        ax.plot(xs, ys, lw=0.6, alpha=0.18, color=line.get_color())
    ax.set_title(f"Trial {k}", fontsize=11)
    ax.grid(alpha=.25)

# 2x5 overview grid — legend kept OUT of the panels, in its own row at the bottom
fig, axes = plt.subplots(2, 5, figsize=(20, 9.2), sharex=True)
handles = labels = None
for k, ax in enumerate(axes.flat):
    plot_trial(ax, k)
    if k == 0: handles, labels = ax.get_legend_handles_labels()
    if k % 5 == 0: ax.set_ylabel("mean #turns")
    if k >= 5: ax.set_xlabel("training step")
fig.suptitle("Per-trial mean turn count vs training step  (bold = 11-step rolling mean, faint = raw)", fontsize=15)
# Fixed legend layout. Entries not present in the current RUNS are filtered out (if d in labels).
# With the 4 current runs + ncol=2: col1 = explore pair, col2 = budget pair (ACT-only on top, ACT+WRITE below).
_desired = [
    "ACT-only · window=1 · β=0",
    "ACT+WRITE(norm_improve) · window=1 · β=0",
    "ACT-only · window=3 · β=0",
    "ACT-only · window=3 · β=0.3 (explore)",
    "ACT+WRITE(norm_improve) · window=3 · β=0",
    "ACT+WRITE(norm_improve) · window=3 · β=0.3 (explore)",
    "ACT-only · window=3 · β=0.3 (explore) · budget",
    "ACT+WRITE(norm_improve) · window=3 · β=0.3 (explore) · budget",
]
_order = [labels.index(d) for d in _desired if d in labels]
handles = [handles[i] for i in _order]
labels = [labels[i] for i in _order]
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12, frameon=True,
           bbox_to_anchor=(0.5, 0.01), title="run", title_fontsize=12)
fig.tight_layout(rect=[0, 0.11, 1, 0.96])   # reserve bottom ~11% for the legend row, top for the title
fig.savefig(f"{ALC}/notes/turns_per_trial_grid.png", dpi=120); plt.close(fig)

print("wrote notes/turns_per_trial_grid.png + turns_per_trial_curves.json")
