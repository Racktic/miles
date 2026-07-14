#!/usr/bin/env python3
"""Create plots for Alchemy WRITE reward signal diagnostics."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    "sig3": "qwen3-4b-curr950-sig3down-fixed-r120-e10-20260625",
    "sig4_raw": "qwen3-4b-curr950-sig4improve-fixed-r120-e10-20260625",
    "sig4_norm": "qwen3-4b-curr950-sig4normimprove-userdata-r120-e10-20260625-232325",
    "freeform": "qwen3-4b-curr950-freeform-r120-e10-20260621-022402",
}
GAMMAS = [0.5, 0.8, 0.95]
LOG_ROOT = Path("examples/alchemy/logs")
ORACLE_PATH = Path("examples/alchemy/eval/oracle_cache.json")
OUT_DIR = Path("examples/alchemy/logs/reward_noise_figures")


def load_oracle() -> dict[int, list[float]]:
    raw = json.load(open(ORACLE_PATH))
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


def discounted_future(xs: list[float], gamma: float) -> list[float]:
    ret = [0.0] * len(xs)
    acc = 0.0
    for i in range(len(xs) - 1, -1, -1):
        ret[i] = acc
        acc = xs[i] + gamma * acc
    return ret


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def std(xs: list[float]) -> float:
    return stdev(xs) if len(xs) > 1 else 0.0


def collect() -> dict[str, dict[str, list[float]]]:
    oracle = load_oracle()
    data: dict[str, dict[str, list[float]]] = {}
    for label, run in RUNS.items():
        d = {
            "raw_delta": [],
            "norm_delta": [],
            "norm_window3": [],
            "raw_future_0.5": [],
            "raw_future_0.8": [],
            "raw_future_0.95": [],
            "norm_future_0.5": [],
            "norm_future_0.8": [],
            "norm_future_0.95": [],
        }
        for path in (LOG_ROOT / run / "traj" / "train").glob("rollout_*/*.json"):
            try:
                rec = json.load(open(path))
            except Exception:
                continue
            raw = [float(x) for x in (rec.get("per_trial_scores") or [])]
            if len(raw) < 2:
                continue
            ep = rec.get("episode_index")
            ors = oracle.get(ep) if isinstance(ep, int) else None
            if not ors or len(ors) < len(raw) or any(float(o) <= 0 for o in ors[: len(raw)]):
                continue
            norm = [r / float(o) for r, o in zip(raw, ors)]
            raw_ret = {g: discounted_future(raw, g) for g in GAMMAS}
            norm_ret = {g: discounted_future(norm, g) for g in GAMMAS}
            for k in range(len(raw) - 1):
                d["raw_delta"].append(raw[k + 1] - raw[k])
                d["norm_delta"].append(norm[k + 1] - norm[k])
                d["norm_window3"].append(mean(norm[k + 1 : min(len(norm), k + 4)]) - norm[k])
                for g in GAMMAS:
                    d[f"raw_future_{g}"].append(raw_ret[g][k])
                    d[f"norm_future_{g}"].append(norm_ret[g][k])
        data[label] = d
    return data


def sample_pair(x: list[float], y: list[float], n: int = 4000):
    idx = list(range(len(x)))
    random.Random(0).shuffle(idx)
    idx = idx[: min(n, len(idx))]
    return [x[i] for i in idx], [y[i] for i in idx]


def plot_std_vs_gamma(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, d in data.items():
        axes[0].plot(GAMMAS, [std(d[f"raw_future_{g}"]) for g in GAMMAS], marker="o", label=label)
        axes[1].plot(GAMMAS, [std(d[f"norm_future_{g}"]) for g in GAMMAS], marker="o", label=label)
    axes[0].set_title("Raw discounted future reward: std")
    axes[1].set_title("Oracle-normalized discounted future reward: std")
    for ax in axes:
        ax.set_xlabel("gamma")
        ax.set_ylabel("standard deviation")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "future_return_std_vs_gamma.png", dpi=180)
    plt.close(fig)


def plot_delta_vs_window(data):
    labels = list(data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].boxplot([data[l]["norm_delta"] for l in labels], labels=labels, showfliers=False)
    axes[0].set_title("One-step normalized improve")
    axes[0].set_ylabel("norm score change")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].boxplot([data[l]["norm_window3"] for l in labels], labels=labels, showfliers=False)
    axes[1].set_title("Window-3 normalized improve")
    axes[1].set_ylabel("mean(next 3 norm scores) - current")
    axes[1].grid(axis="y", alpha=0.3)
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(OUT_DIR / "norm_delta_vs_window3_boxplot.png", dpi=180)
    plt.close(fig)


def plot_scatter(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (label, d) in zip(axes.ravel(), data.items()):
        x, y = sample_pair(d["norm_delta"], d["norm_future_0.8"])
        corr = pearson(d["norm_delta"], d["norm_future_0.8"])
        ax.scatter(x, y, s=5, alpha=0.15)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.axvline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_title(f"{label}: one-step vs future (r={corr:.2f})")
        ax.set_xlabel("one-step norm improve")
        ax.set_ylabel("discounted future norm return, gamma=0.8")
        ax.grid(alpha=0.2)
    fig.savefig(OUT_DIR / "one_step_vs_future_scatter.png", dpi=180)
    plt.close(fig)


def plot_signal_distributions(data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for label, d in data.items():
        for ax, key, title in [
            (axes[0], "norm_delta", "one-step improve"),
            (axes[1], "norm_window3", "window-3 improve"),
            (axes[2], "norm_future_0.8", "discounted future return"),
        ]:
            vals = d[key]
            ax.hist(vals, bins=80, density=True, histtype="step", linewidth=1.2, label=label)
            ax.set_title(title)
            ax.grid(alpha=0.2)
    for ax in axes:
        ax.legend(fontsize=8)
    axes[0].set_xlim(-1.5, 1.5)
    axes[1].set_xlim(-1.5, 1.5)
    axes[2].set_xlim(-0.5, 5.0)
    fig.savefig(OUT_DIR / "norm_signal_distributions.png", dpi=180)
    plt.close(fig)


def write_summary(data):
    rows = []
    for label, d in data.items():
        rows.append(
            {
                "run": label,
                "n": len(d["norm_delta"]),
                "std_norm_delta": std(d["norm_delta"]),
                "std_norm_window3": std(d["norm_window3"]),
                "std_norm_future_0.8": std(d["norm_future_0.8"]),
                "corr_delta_future_0.8": pearson(d["norm_delta"], d["norm_future_0.8"]),
                "corr_window3_future_0.8": pearson(d["norm_window3"], d["norm_future_0.8"]),
            }
        )
    (OUT_DIR / "plot_summary.json").write_text(json.dumps(rows, indent=2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = collect()
    write_summary(data)
    plot_std_vs_gamma(data)
    plot_delta_vs_window(data)
    plot_scatter(data)
    plot_signal_distributions(data)
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
