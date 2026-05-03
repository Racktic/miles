#!/usr/bin/env python3
"""Post-hoc comparison of TITO vs non-TITO rollout dumps.

Loads .pt files saved by --save-debug-rollout-data from two experiment runs
(one TITO, one non-TITO) and compares aggregate metrics: reward, solve rate,
token counts, response lengths, and status distributions.

Usage:
    python tests/proxy/compare_tito_eval.py \
        --tito-dir /data/ckpts/eval_tito/ \
        --non-tito-dir /data/ckpts/eval_non_tito/
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

import torch


def load_samples(dump_dir: str) -> list[dict]:
    """Load all samples from .pt dump files in a directory."""
    dump_path = Path(dump_dir)
    pt_files = sorted(dump_path.glob("*.pt"))
    if not pt_files:
        print(f"ERROR: No .pt files found in {dump_dir}")
        sys.exit(1)

    samples = []
    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)
        samples.extend(data["samples"])

    print(f"  Loaded {len(samples)} samples from {len(pt_files)} files in {dump_dir}")
    return samples


def report(label: str, samples: list[dict]) -> dict:
    """Print and return aggregate metrics for a set of samples."""
    rewards = [s["reward"] for s in samples if s["reward"] is not None]
    resp_lens = [s["response_length"] for s in samples]
    token_lens = [len(s["tokens"]) for s in samples if s.get("tokens")]
    loss_mask_sums = [
        sum(s["loss_mask"]) for s in samples if s.get("loss_mask") is not None
    ]
    statuses = Counter(s.get("status", "unknown") for s in samples)

    solve_rate = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
    failed = statuses.get("failed", 0) + statuses.get("aborted", 0)

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Total samples:       {len(samples)}")
    print(f"  Solve rate:          {solve_rate:.1%} ({sum(1 for r in rewards if r > 0)}/{len(rewards)})")
    if rewards:
        print(f"  Mean reward:         {mean(rewards):.4f}")
        print(f"  Median reward:       {median(rewards):.4f}")
        print(f"  Stdev reward:        {stdev(rewards):.4f}" if len(rewards) > 1 else "")
    if resp_lens:
        print(f"  Mean response_len:   {mean(resp_lens):.0f}")
        print(f"  Median response_len: {median(resp_lens):.0f}")
    if token_lens:
        print(f"  Mean total_tokens:   {mean(token_lens):.0f}")
    if loss_mask_sums:
        print(f"  Mean loss_mask_sum:  {mean(loss_mask_sums):.0f}")
    print(f"  Status distribution: {dict(statuses)}")
    print(f"  Failed/Aborted:      {failed}")

    return {
        "rewards": rewards,
        "resp_lens": resp_lens,
        "token_lens": token_lens,
        "solve_rate": solve_rate,
        "statuses": statuses,
    }


def compare(tito_metrics: dict, non_tito_metrics: dict):
    """Print side-by-side comparison and statistical test."""
    print(f"\n{'=' * 50}")
    print(f"  COMPARISON")
    print(f"{'=' * 50}")

    # Solve rate diff
    sr_diff = tito_metrics["solve_rate"] - non_tito_metrics["solve_rate"]
    print(f"  Solve rate diff (TITO - NonTITO): {sr_diff:+.1%}")

    # Reward diff
    if tito_metrics["rewards"] and non_tito_metrics["rewards"]:
        r_diff = mean(tito_metrics["rewards"]) - mean(non_tito_metrics["rewards"])
        print(f"  Mean reward diff:                 {r_diff:+.4f}")

    # Response length diff
    if tito_metrics["resp_lens"] and non_tito_metrics["resp_lens"]:
        rl_diff = mean(tito_metrics["resp_lens"]) - mean(non_tito_metrics["resp_lens"])
        rl_pct = rl_diff / mean(non_tito_metrics["resp_lens"]) * 100 if mean(non_tito_metrics["resp_lens"]) > 0 else 0
        print(f"  Mean resp_len diff:               {rl_diff:+.0f} ({rl_pct:+.1f}%)")

    # Token count diff
    if tito_metrics["token_lens"] and non_tito_metrics["token_lens"]:
        tl_diff = mean(tito_metrics["token_lens"]) - mean(non_tito_metrics["token_lens"])
        tl_pct = tl_diff / mean(non_tito_metrics["token_lens"]) * 100 if mean(non_tito_metrics["token_lens"]) > 0 else 0
        print(f"  Mean total_tokens diff:           {tl_diff:+.0f} ({tl_pct:+.1f}%)")

    # Statistical test
    if tito_metrics["rewards"] and non_tito_metrics["rewards"]:
        try:
            from scipy.stats import ttest_ind

            t_stat, p_value = ttest_ind(
                tito_metrics["rewards"], non_tito_metrics["rewards"]
            )
            print(f"\n  T-test (rewards): t={t_stat:.4f}, p={p_value:.4f}")
            if p_value > 0.05:
                print(f"  VERDICT: EQUIVALENT (p={p_value:.4f} > 0.05)")
            else:
                print(f"  VERDICT: DIFFERENT (p={p_value:.4f} < 0.05)")
        except ImportError:
            print("\n  (scipy not available — skipping t-test)")
            if abs(sr_diff) < 0.03:
                print(f"  VERDICT: LIKELY EQUIVALENT (solve rate diff {sr_diff:+.1%} < 3%)")
            else:
                print(f"  VERDICT: POSSIBLY DIFFERENT (solve rate diff {sr_diff:+.1%} >= 3%)")


def compare_diagnostics(tito_samples: list[dict], non_tito_samples: list[dict]):
    """Compare per-task tool call counts between TITO and non-TITO."""
    tito_map = {}
    for s in tito_samples:
        iid = s.get("metadata", {}).get("instance_id")
        if iid:
            tito_map[iid] = s

    non_tito_map = {}
    for s in non_tito_samples:
        iid = s.get("metadata", {}).get("instance_id")
        if iid:
            non_tito_map[iid] = s

    common = sorted(set(tito_map) & set(non_tito_map))
    print(f"\n{'=' * 50}")
    print(f"  PER-TASK DIAGNOSTIC COMPARISON ({len(common)} common tasks)")
    print(f"{'=' * 50}")

    if not common:
        print("  No common tasks found (check instance_id in metadata)")
        return

    tc_diffs = []
    reward_diffs = []
    for iid in common:
        t = tito_map[iid]
        n = non_tito_map[iid]
        t_diag = t.get("metadata", {}).get("diag", {})
        n_diag = n.get("metadata", {}).get("diag", {})
        t_tc = t_diag.get("tool_call_msgs", -1)
        n_tc = n_diag.get("tool_call_msgs", -1)
        t_marker = t_diag.get("tc_marker_in_content", -1)
        n_marker = n_diag.get("tc_marker_in_content", -1)
        t_r = t.get("reward", 0) or 0
        n_r = n.get("reward", 0) or 0
        if t_tc != n_tc:
            tc_diffs.append((iid, t_tc, n_tc, t_marker, n_marker, t_r, n_r))
        if t_r != n_r:
            reward_diffs.append((iid, t_tc, n_tc, t_marker, n_marker, t_r, n_r))

    print(f"  Tasks with different tool_call counts: {len(tc_diffs)}")
    print(f"  Tasks with different rewards:          {len(reward_diffs)}")

    # Tasks where TITO failed but non-TITO succeeded
    tito_fail = [x for x in reward_diffs if x[5] == 0 and x[6] > 0]
    nontito_fail = [x for x in reward_diffs if x[5] > 0 and x[6] == 0]
    print(f"  TITO=0 but NonTITO>0:                  {len(tito_fail)}")
    print(f"  TITO>0 but NonTITO=0:                  {len(nontito_fail)}")

    if tito_fail:
        print(f"\n  Top tasks where TITO failed but non-TITO succeeded:")
        for iid, t_tc, n_tc, t_marker, n_marker, t_r, n_r in tito_fail[:20]:
            print(
                f"    {iid}: tito_tc={t_tc} nontito_tc={n_tc} "
                f"tito_marker={t_marker} nontito_marker={n_marker}"
            )

    # Summary: do tasks with fewer tool calls in TITO correlate with failure?
    fewer_tc_in_tito = [x for x in tc_diffs if x[1] < x[2]]
    fewer_tc_and_failed = [x for x in fewer_tc_in_tito if x[5] == 0 and x[6] > 0]
    if fewer_tc_in_tito:
        print(f"\n  Tasks with fewer tool_calls in TITO: {len(fewer_tc_in_tito)}")
        print(f"  Of those, TITO failed & non-TITO passed: {len(fewer_tc_and_failed)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare TITO vs non-TITO rollout evaluation dumps"
    )
    parser.add_argument(
        "--tito-dir",
        required=True,
        help="Directory containing TITO .pt dump files",
    )
    parser.add_argument(
        "--non-tito-dir",
        required=True,
        help="Directory containing non-TITO .pt dump files",
    )
    args = parser.parse_args()

    print("Loading TITO samples...")
    tito_samples = load_samples(args.tito_dir)

    print("Loading non-TITO samples...")
    non_tito_samples = load_samples(args.non_tito_dir)

    tito_metrics = report("TITO", tito_samples)
    non_tito_metrics = report("Non-TITO", non_tito_samples)

    compare(tito_metrics, non_tito_metrics)
    compare_diagnostics(tito_samples, non_tito_samples)


if __name__ == "__main__":
    main()
