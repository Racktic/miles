#!/usr/bin/env python3
"""Plot per-episode adjacent-improve delta lines for Alchemy trajectories.

Each line is one episode:
  delta_k = score[k+1] - score[k]
or normalized:
  norm_delta_k = score[k+1]/oracle[k+1] - score[k]/oracle[k]

These plots are meant to make trial-to-trial noisiness visually obvious.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    "sig3": "qwen3-4b-curr950-sig3down-fixed-r120-e10-20260625",
    "sig4_raw": "qwen3-4b-curr950-sig4improve-fixed-r120-e10-20260625",
    "sig4_norm": "qwen3-4b-curr950-sig4normimprove-userdata-r120-e10-20260625-232325",
    "freeform": "qwen3-4b-curr950-freeform-r120-e10-20260621-022402",
}


def load_oracle(path: Path) -> dict[int, list[float]]:
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        try:
            ep = int(k)
        except Exception:
            continue
        if isinstance(v, list):
            out[ep] = [float(x) for x in v]
        elif isinstance(v, dict):
            for kk in ("oracle_per_trial", "per_trial_scores", "scores"):
                if kk in v:
                    out[ep] = [float(x) for x in v[kk]]
                    break
    return out


def collect_run(log_root: Path, run: str, oracle: dict[int, list[float]], max_lines: int, seed: int):
    raw_lines = []
    norm_lines = []
    files = sorted((log_root / run / "traj" / "train").glob("rollout_*/*.json"))
    rng = random.Random(seed)
    rng.shuffle(files)
    for path in files:
        if len(raw_lines) >= max_lines and len(norm_lines) >= max_lines:
            break
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        pts = rec.get("per_trial_scores") or []
        if len(pts) < 10:
            continue
        scores = [float(x) for x in pts[:10]]
        raw_delta = [scores[i + 1] - scores[i] for i in range(9)]
        if len(raw_lines) < max_lines:
            raw_lines.append(raw_delta)

        ep = rec.get("episode_index")
        ors = oracle.get(ep) if isinstance(ep, int) else None
        if not ors or len(ors) < 10 or any(float(o) <= 0 for o in ors[:10]):
            continue
        norm = [scores[i] / float(ors[i]) for i in range(10)]
        norm_delta = [norm[i + 1] - norm[i] for i in range(9)]
        if len(norm_lines) < max_lines:
            norm_lines.append(norm_delta)
    return raw_lines, norm_lines


def plot_lines(data: dict[str, list[list[float]]], out: Path, title: str, ylabel: str, ylim=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    xs = list(range(1, 10))
    for ax, (label, lines) in zip(axes.ravel(), data.items()):
        for ys in lines:
            ax.plot(xs, ys, color="#2563eb", alpha=0.055, linewidth=0.8)
        if lines:
            avg = [mean(line[i] for line in lines) for i in range(9)]
            ax.plot(xs, avg, color="#dc2626", linewidth=2.0, label="mean")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{label} (n={len(lines)})")
        ax.set_xlabel("boundary k: trial k -> k+1")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_late_vs_early(log_root: Path, oracle: dict[int, list[float]], out_dir: Path, max_lines: int, seed: int):
    # One targeted plot for sig4_raw: early vs late norm deltas, because this is the run under discussion.
    run = RUNS["sig4_raw"]
    buckets = {"sig4_raw early r0-19": [], "sig4_raw late r100-119": []}
    files = sorted((log_root / run / "traj" / "train").glob("rollout_*/*.json"))
    rng = random.Random(seed)
    rng.shuffle(files)
    for path in files:
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        rid = rec.get("rollout_id")
        key = None
        if isinstance(rid, int) and 0 <= rid <= 19:
            key = "sig4_raw early r0-19"
        elif isinstance(rid, int) and 100 <= rid <= 119:
            key = "sig4_raw late r100-119"
        if key is None or len(buckets[key]) >= max_lines:
            continue
        pts = rec.get("per_trial_scores") or []
        ep = rec.get("episode_index")
        ors = oracle.get(ep) if isinstance(ep, int) else None
        if len(pts) < 10 or not ors or len(ors) < 10 or any(float(o) <= 0 for o in ors[:10]):
            continue
        norm = [float(pts[i]) / float(ors[i]) for i in range(10)]
        buckets[key].append([norm[i + 1] - norm[i] for i in range(9)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True, constrained_layout=True)
    xs = list(range(1, 10))
    for ax, (label, lines) in zip(axes, buckets.items()):
        for ys in lines:
            ax.plot(xs, ys, color="#7c3aed", alpha=0.08, linewidth=0.8)
        avg = [mean(line[i] for line in lines) for i in range(9)] if lines else []
        if avg:
            ax.plot(xs, avg, color="#dc2626", linewidth=2.0, label="mean")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{label} (n={len(lines)})")
        ax.set_xlabel("boundary k: trial k -> k+1")
        ax.set_ylabel("normalized adjacent improve")
        ax.grid(alpha=0.25)
        ax.set_ylim(-1.5, 1.5)
        ax.legend(fontsize=8)
    fig.suptitle("sig4_raw: adjacent normalized improve remains noisy within episodes")
    fig.savefig(out_dir / "sig4raw_early_vs_late_norm_delta_lines.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default="examples/alchemy/logs")
    ap.add_argument("--oracle", default="examples/alchemy/eval/oracle_cache.json")
    ap.add_argument("--out-dir", default="examples/alchemy/logs/reward_noise_figures")
    ap.add_argument("--max-lines", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle = load_oracle(Path(args.oracle))

    raw_by_run = {}
    norm_by_run = {}
    for label, run in RUNS.items():
        raw, norm = collect_run(Path(args.log_root), run, oracle, args.max_lines, args.seed)
        raw_by_run[label] = raw
        norm_by_run[label] = norm

    plot_lines(
        raw_by_run,
        out_dir / "adjacent_raw_improve_lines.png",
        "Per-episode adjacent raw improve lines",
        "raw score[k+1] - raw score[k]",
        ylim=(-35, 35),
    )
    plot_lines(
        norm_by_run,
        out_dir / "adjacent_norm_improve_lines.png",
        "Per-episode adjacent normalized improve lines",
        "norm score[k+1] - norm score[k]",
        ylim=(-1.5, 1.5),
    )
    plot_late_vs_early(Path(args.log_root), oracle, out_dir, args.max_lines, args.seed)
    print(f"wrote adjacent-improve line plots to {out_dir}")


if __name__ == "__main__":
    main()
