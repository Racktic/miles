#!/usr/bin/env python3
"""Analyze memory evolution from saved Alchemy training trajectories.

The script is intentionally read-only. It summarizes memory length over rollouts
and writes candidate cases for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev


RUNS = {
    "freeform": "qwen3-4b-curr950-freeform-r120-e10-20260621-022402",
    "sig3_downstream": "qwen3-4b-curr950-sig3down-fixed-r120-e10-20260625",
    "sig4_raw_improve": "qwen3-4b-curr950-sig4improve-fixed-r120-e10-20260625",
    "sig4_norm_improve": "qwen3-4b-curr950-sig4normimprove-userdata-r120-e10-20260625-232325",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def q(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[max(0, min(len(ys) - 1, round((len(ys) - 1) * p)))]


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": median(xs),
        "std": stdev(xs) if len(xs) > 1 else 0.0,
        "p10": q(xs, 0.10),
        "p25": q(xs, 0.25),
        "p75": q(xs, 0.75),
        "p90": q(xs, 0.90),
        "min": min(xs),
        "max": max(xs),
    }


def text_features(s: str) -> dict:
    words = re.findall(r"[A-Za-z0-9_+-]+", s.lower())
    bullets = len(re.findall(r"(?m)^\s*[-*]\s+", s))
    lines = [ln for ln in s.splitlines() if ln.strip()]
    return {
        "chars": len(s),
        "lines": len(lines),
        "bullets": bullets,
        "unique_word_frac": len(set(words)) / len(words) if words else 0.0,
        "contains_table": "|" in s and "---" in s,
        "no_effect_count": len(re.findall(r"\bno effect\b", s, re.I)),
        "highest_reward_mentions": len(re.findall(r"highest reward|reward combination", s, re.I)),
    }


def iter_files(run_dir: Path):
    yield from sorted((run_dir / "traj" / "train").glob("rollout_*/*.json"))


def summarize_run(log_root: Path, run_name: str) -> dict:
    run_dir = log_root / run_name
    by_rollout: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cases = []
    n_files = 0
    n_bad = 0
    for path in iter_files(run_dir):
        try:
            rec = load_json(path)
        except Exception:
            n_bad += 1
            continue
        summaries = rec.get("summaries") or []
        if not summaries:
            continue
        rid = int(rec.get("rollout_id", -1))
        lengths = [len(s) for s in summaries if isinstance(s, str)]
        if not lengths:
            continue
        n_files += 1
        final = summaries[-1]
        final_feat = text_features(final)
        by_rollout[rid]["all_lengths"].extend(lengths)
        by_rollout[rid]["final_lengths"].append(len(final))
        by_rollout[rid]["final_lines"].append(final_feat["lines"])
        by_rollout[rid]["final_bullets"].append(final_feat["bullets"])
        by_rollout[rid]["final_unique_word_frac"].append(final_feat["unique_word_frac"])
        by_rollout[rid]["write_kept_frac"].append(
            sum(1 for w in (rec.get("write_audit") or []) if w.get("kept")) / max(1, len(rec.get("write_audit") or []))
        )
        cases.append(
            {
                "path": str(path),
                "rollout": rid,
                "episode_index": rec.get("episode_index"),
                "episode_id": rec.get("episode_id"),
                "per_trial_scores": rec.get("per_trial_scores"),
                "write_signal": rec.get("write_signal"),
                "summary_lengths": lengths,
                "final_length": len(final),
                "final_features": final_feat,
                "first_summary": summaries[0],
                "final_summary": final,
                "write_audit_brief": [
                    {
                        "rewrite_idx": w.get("rewrite_idx"),
                        "n_fk": w.get("n_fk"),
                        "kept": w.get("kept"),
                        "downstream_reward": w.get("downstream_reward"),
                    }
                    for w in (rec.get("write_audit") or [])
                ],
            }
        )

    rollout_rows = []
    for rid in sorted(by_rollout):
        row = {"rollout": rid}
        for key, vals in by_rollout[rid].items():
            row[key] = stats(vals)
        rollout_rows.append(row)

    final_by_rollout = [(r["rollout"], r["final_lengths"]["mean"]) for r in rollout_rows if r["final_lengths"]["n"]]
    first_window = [x for rid, x in final_by_rollout if rid <= 9]
    last_max = max((rid for rid, _ in final_by_rollout), default=-1)
    last_window = [x for rid, x in final_by_rollout if rid >= last_max - 9]
    trend = None
    if first_window and last_window:
        trend = {
            "early_final_mean": mean(first_window),
            "late_final_mean": mean(last_window),
            "late_minus_early": mean(last_window) - mean(first_window),
            "late_over_early": mean(last_window) / mean(first_window) if mean(first_window) else None,
            "last_rollout": last_max,
        }

    # Candidate cases at early/mid/late, picking closest to rollout median length and extremes.
    candidates = []
    target_rollouts = sorted(set([0, 1, 5, 10, 20, 40, 60, 80, 100, last_max]))
    by_rid_cases: dict[int, list[dict]] = defaultdict(list)
    for c in cases:
        by_rid_cases[c["rollout"]].append(c)
    available = sorted(by_rid_cases)
    for target in target_rollouts:
        if not available:
            continue
        rid = min(available, key=lambda x: abs(x - target))
        cs = by_rid_cases[rid]
        med = median([c["final_length"] for c in cs])
        picked = [
            min(cs, key=lambda c: abs(c["final_length"] - med)),
            min(cs, key=lambda c: c["final_length"]),
            max(cs, key=lambda c: c["final_length"]),
        ]
        seen = set()
        for tag, c in zip(("median", "shortest", "longest"), picked):
            if c["path"] in seen:
                continue
            seen.add(c["path"])
            candidates.append({"target_rollout": target, "case_type": tag, **c})

    return {
        "run": run_name,
        "n_episodes": n_files,
        "n_bad_files": n_bad,
        "rollouts": rollout_rows,
        "trend": trend,
        "cases": candidates,
    }


def concise_excerpt(s: str, max_chars: int = 900) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "\n..."


def write_md(results: dict, out: Path) -> None:
    lines = [
        "# Alchemy Memory Evolution Analysis",
        "",
        "This is an automatically generated first-pass analysis. Case excerpts are included for manual follow-up.",
        "",
        "## Length Trend",
        "",
        "| label | run | episodes | last rollout | early final mean | late final mean | late/early | late - early |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, r in results.items():
        t = r.get("trend") or {}
        lines.append(
            f"| {label} | {r['run']} | {r['n_episodes']} | {t.get('last_rollout')} | "
            f"{t.get('early_final_mean', 0):.1f} | {t.get('late_final_mean', 0):.1f} | "
            f"{t.get('late_over_early', 0):.3f} | {t.get('late_minus_early', 0):+.1f} |"
        )
    lines += ["", "## Rollout Snapshots", ""]
    for label, r in results.items():
        lines += [f"### {label}", "", "| rollout | final mean | final median | final p10 | final p90 | bullets mean | kept frac mean |", "|---:|---:|---:|---:|---:|---:|---:|"]
        wanted = {0, 1, 5, 10, 20, 40, 60, 80, 100, (r.get("trend") or {}).get("last_rollout")}
        for row in r["rollouts"]:
            if row["rollout"] not in wanted:
                continue
            fl = row["final_lengths"]
            bu = row.get("final_bullets", {})
            kf = row.get("write_kept_frac", {})
            lines.append(
                f"| {row['rollout']} | {fl.get('mean', 0):.1f} | {fl.get('median', 0):.1f} | "
                f"{fl.get('p10', 0):.0f} | {fl.get('p90', 0):.0f} | {bu.get('mean', 0):.1f} | {kf.get('mean', 0):.2f} |"
            )
        lines.append("")

    lines += ["## Candidate Case Excerpts", ""]
    for label, r in results.items():
        lines += [f"### {label}", ""]
        for c in r["cases"]:
            if c["case_type"] != "median":
                continue
            if c["target_rollout"] not in (0, 20, 60, 100, (r.get("trend") or {}).get("last_rollout")):
                continue
            lines += [
                f"**rollout {c['rollout']} / episode {c['episode_index']} / final_length {c['final_length']}**",
                "",
                "```text",
                concise_excerpt(c["final_summary"], 900),
                "```",
                "",
            ]
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", default="examples/alchemy/logs")
    ap.add_argument("--out-json", default="examples/alchemy/logs/memory_evolution_analysis.json")
    ap.add_argument("--out-md", default="examples/alchemy/logs/memory_evolution_analysis_auto.md")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    results = {label: summarize_run(log_root, run) for label, run in RUNS.items()}
    Path(args.out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_md(results, Path(args.out_md))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
