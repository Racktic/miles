#!/usr/bin/env python3
"""Analyze trial-to-trial reward noise from saved Alchemy training trajectories.

This is read-only and intended for offline diagnosis of WRITE reward signals:
- sig4 raw improve: r[k+1] - r[k]
- sig4 norm improve: r[k+1]/oracle[k+1] - r[k]/oracle[k]
- less myopic alternatives: discounted future returns from k+1 onward.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _quantile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    idx = round((len(ys) - 1) * p)
    return ys[max(0, min(len(ys) - 1, idx))]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": mean(xs),
        "std": stdev(xs) if len(xs) > 1 else 0.0,
        "abs_mean": mean(abs(x) for x in xs),
        "p05": _quantile(xs, 0.05),
        "p10": _quantile(xs, 0.10),
        "p25": _quantile(xs, 0.25),
        "p50": _quantile(xs, 0.50),
        "p75": _quantile(xs, 0.75),
        "p90": _quantile(xs, 0.90),
        "p95": _quantile(xs, 0.95),
        "frac_pos": sum(x > 0 for x in xs) / len(xs),
        "frac_neg": sum(x < 0 for x in xs) / len(xs),
        "frac_zero": sum(x == 0 for x in xs) / len(xs),
    }


def _oracle_table(path: Path) -> dict[int, list[float]]:
    raw = _load_json(path)
    out: dict[int, list[float]] = {}
    for k, v in raw.items():
        try:
            ep = int(k)
        except Exception:
            continue
        if isinstance(v, list):
            out[ep] = [float(x) for x in v]
        elif isinstance(v, dict):
            for kk in ("oracle_per_trial", "per_trial_scores", "scores"):
                if kk in v and isinstance(v[kk], list):
                    out[ep] = [float(x) for x in v[kk]]
                    break
    return out


def _discounted_future(xs: list[float], gamma: float) -> list[float]:
    # ret[k] = sum_{j=k+1}^{T-1} gamma^(j-k-1) xs[j], i.e. reward after WRITE at boundary k.
    ret = [0.0] * len(xs)
    acc = 0.0
    for i in range(len(xs) - 1, -1, -1):
        ret[i] = acc
        acc = xs[i] + gamma * acc
    return ret


def _rolling_future_mean(xs: list[float], k: int, window: int) -> float | None:
    vals = xs[k + 1 : k + 1 + window]
    return mean(vals) if vals else None


def _iter_traj_files(run_dir: Path):
    train = run_dir / "traj" / "train"
    if not train.exists():
        return
    yield from train.glob("rollout_*/*.json")


def analyze_run(log_root: Path, run: str, oracle: dict[int, list[float]], gammas: list[float]) -> dict:
    run_dir = log_root / run
    raw_next_diffs: list[float] = []
    norm_next_diffs: list[float] = []
    raw_future: dict[str, list[float]] = {str(g): [] for g in gammas}
    norm_future: dict[str, list[float]] = {str(g): [] for g in gammas}
    raw_window3_diffs: list[float] = []
    norm_window3_diffs: list[float] = []
    raw_diff_pairs: list[tuple[float, float]] = []
    norm_diff_pairs: list[tuple[float, float]] = []
    by_pos_raw: dict[int, list[float]] = defaultdict(list)
    by_pos_norm: dict[int, list[float]] = defaultdict(list)
    n_files = 0
    n_bad = 0
    n_norm_missing = 0
    n_write = 0
    n_write_kept = 0

    for path in _iter_traj_files(run_dir) or []:
        try:
            rec = _load_json(path)
        except Exception:
            n_bad += 1
            continue
        pts = rec.get("per_trial_scores") or []
        if len(pts) < 2:
            continue
        try:
            raw = [float(x) for x in pts]
        except Exception:
            n_bad += 1
            continue
        n_files += 1
        audit = rec.get("write_audit") or []
        n_write += len(audit)
        n_write_kept += sum(1 for w in audit if w.get("kept"))

        ep = rec.get("episode_index")
        norms: list[float | None] = [None] * len(raw)
        if isinstance(ep, int) and ep in oracle and len(oracle[ep]) >= len(raw):
            for i, (r, o) in enumerate(zip(raw, oracle[ep])):
                norms[i] = r / o if o > 0 else None
        else:
            n_norm_missing += 1

        raw_returns = {g: _discounted_future(raw, g) for g in gammas}
        norm_values = [x if x is not None else math.nan for x in norms]
        norm_returns = {
            g: _discounted_future(norm_values, g) if all(x is not None for x in norms) else None
            for g in gammas
        }

        prev_raw_diff = None
        prev_norm_diff = None
        for k in range(len(raw) - 1):
            rd = raw[k + 1] - raw[k]
            raw_next_diffs.append(rd)
            by_pos_raw[k].append(rd)
            if prev_raw_diff is not None:
                raw_diff_pairs.append((prev_raw_diff, rd))
            prev_raw_diff = rd

            w3 = _rolling_future_mean(raw, k, 3)
            if w3 is not None:
                raw_window3_diffs.append(w3 - raw[k])
            for g in gammas:
                raw_future[str(g)].append(raw_returns[g][k])

            if norms[k] is not None and norms[k + 1] is not None:
                nd = float(norms[k + 1]) - float(norms[k])
                norm_next_diffs.append(nd)
                by_pos_norm[k].append(nd)
                if prev_norm_diff is not None:
                    norm_diff_pairs.append((prev_norm_diff, nd))
                prev_norm_diff = nd
                nw3_vals = [x for x in norms[k + 1 : k + 4] if x is not None]
                if nw3_vals:
                    norm_window3_diffs.append(mean(nw3_vals) - float(norms[k]))
                for g in gammas:
                    nr = norm_returns[g]
                    if nr is not None and not math.isnan(nr[k]):
                        norm_future[str(g)].append(nr[k])

    def sign_flip_frac(pairs: list[tuple[float, float]]) -> float | None:
        nz = [(a, b) for a, b in pairs if a != 0 and b != 0]
        if not nz:
            return None
        return sum(a * b < 0 for a, b in nz) / len(nz)

    return {
        "run": run,
        "n_episodes": n_files,
        "n_bad_files": n_bad,
        "n_norm_missing_episodes": n_norm_missing,
        "n_write_audit": n_write,
        "n_write_kept": n_write_kept,
        "raw_next_diff": _stats(raw_next_diffs),
        "norm_next_diff": _stats(norm_next_diffs),
        "raw_next_diff_sign_flip_frac": sign_flip_frac(raw_diff_pairs),
        "norm_next_diff_sign_flip_frac": sign_flip_frac(norm_diff_pairs),
        "raw_window3_minus_current": _stats(raw_window3_diffs),
        "norm_window3_minus_current": _stats(norm_window3_diffs),
        "raw_discounted_future": {g: _stats(v) for g, v in raw_future.items()},
        "norm_discounted_future": {g: _stats(v) for g, v in norm_future.items()},
        "by_trial_pos_raw_next_diff": {str(k): _stats(v) for k, v in sorted(by_pos_raw.items())},
        "by_trial_pos_norm_next_diff": {str(k): _stats(v) for k, v in sorted(by_pos_norm.items())},
    }


def _fmt(x, nd=3):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_markdown(results: list[dict], out: Path) -> None:
    lines = [
        "# Alchemy Reward Noise Analysis",
        "",
        "Read-only analysis from saved training trajectories.",
        "",
        "## Adjacent Improve Noise",
        "",
        "| run | episodes | raw diff mean | raw diff std | raw abs | raw sign flip | norm diff mean | norm diff std | norm abs | norm sign flip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        raw = r["raw_next_diff"]
        norm = r["norm_next_diff"]
        lines.append(
            "| {run} | {n} | {rm} | {rs} | {ra} | {rf} | {nm} | {ns} | {na} | {nf} |".format(
                run=r["run"],
                n=r["n_episodes"],
                rm=_fmt(raw.get("mean")),
                rs=_fmt(raw.get("std")),
                ra=_fmt(raw.get("abs_mean")),
                rf=_fmt(r.get("raw_next_diff_sign_flip_frac")),
                nm=_fmt(norm.get("mean")),
                ns=_fmt(norm.get("std")),
                na=_fmt(norm.get("abs_mean")),
                nf=_fmt(r.get("norm_next_diff_sign_flip_frac")),
            )
        )

    lines += [
        "",
        "## Less-Myopic Candidate Signals",
        "",
        "`discounted_future[k] = sum_{j>k} gamma^(j-k-1) r[j]`, the total future reward after a WRITE at boundary `k`.",
        "",
        "| run | raw gamma=0.5 std | raw gamma=0.8 std | raw gamma=0.95 std | norm gamma=0.5 std | norm gamma=0.8 std | norm gamma=0.95 std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        rf = r["raw_discounted_future"]
        nf = r["norm_discounted_future"]
        lines.append(
            f"| {r['run']} | {_fmt(rf['0.5'].get('std'))} | {_fmt(rf['0.8'].get('std'))} | {_fmt(rf['0.95'].get('std'))} | "
            f"{_fmt(nf['0.5'].get('std'))} | {_fmt(nf['0.8'].get('std'))} | {_fmt(nf['0.95'].get('std'))} |"
        )

    lines += [
        "",
        "## Interpretation Pointers",
        "",
        "- High adjacent-diff std and high sign-flip fraction indicate that `r[k+1]-r[k]` is noisy as a WRITE reward.",
        "- If normalized adjacent diff has lower std/sign-flip than raw adjacent diff, oracle normalization is helping.",
        "- Discounted future reward is less myopic but also has larger scale variance; it likely needs group whitening.",
        "- Windowed future mean, e.g. mean of the next 3 trials minus current, is a middle ground between sig4 and full return.",
        "",
    ]
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default="examples/alchemy/logs")
    ap.add_argument("--oracle", default="examples/alchemy/eval/oracle_cache.json")
    ap.add_argument("--out-json", default="examples/alchemy/logs/reward_noise_analysis.json")
    ap.add_argument("--out-md", default="examples/alchemy/logs/reward_noise_analysis.md")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.5, 0.8, 0.95])
    args = ap.parse_args()

    oracle = _oracle_table(Path(args.oracle))
    results = [analyze_run(Path(args.log_root), run, oracle, args.gammas) for run in args.runs]
    Path(args.out_json).write_text(json.dumps(results, indent=2, sort_keys=True))
    write_markdown(results, Path(args.out_md))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
