"""Merge a cached oracle table into an eval run's raw results to produce normalized-to-oracle scores.

The online Claude eval runs with --oracle-mode skip (oracle is too slow to compute inline), so its
results.jsonl has agent_per_trial but normalized=None. Once compute_oracle.py has populated
oracle_cache.json, run this to fill in per-trial normalization, performance, and I_score, and rewrite
results.jsonl + summary.json in place (plus a normalized.json with the aggregate).

  python examples/alchemy/eval/finalize.py --run examples/alchemy/logs/eval-claude-opus-4-8-XXXX \
      --cache examples/alchemy/eval/oracle_cache.json
"""
from __future__ import annotations

import argparse
import json
import os

N_TRIALS = 10


def normalize(agent, oracle):
    return [None if o <= 0 else float(a) / float(o) for a, o in zip(agent, oracle)]


def _improve(scores):  # robust: mean(last5) - mean(first5), dropping None per half
    first = [x for x in scores[:5] if x is not None]
    last = [x for x in scores[5:] if x is not None]
    return (sum(last) / len(last) - sum(first) / len(first)) if (first and last) else None


def _improve_tmlr(scores):  # TMLR exact: mean(last5) - trial1
    last = [x for x in scores[5:] if x is not None]
    return (sum(last) / len(last) - scores[0]) if (last and scores[0] is not None) else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _stderr(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="eval run dir (contains results.jsonl)")
    ap.add_argument("--cache", required=True, help="oracle_cache.json from compute_oracle.py")
    args = ap.parse_args()

    with open(args.cache) as f:
        oracle_cache = json.load(f)
    results_path = os.path.join(args.run, "results.jsonl")
    recs = [json.loads(line) for line in open(results_path) if line.strip()]

    filled, missing = [], []
    for rec in recs:
        idx = str(rec["episode_index"])
        if idx not in oracle_cache:
            missing.append(rec["episode_index"])
            continue
        oracle = oracle_cache[idx]
        agent = rec["agent_per_trial"]
        norm = normalize(agent, oracle)
        valid = [x for x in norm if x is not None]
        rec["oracle_per_trial"] = oracle
        rec["normalized_per_trial"] = norm
        rec["performance"] = (sum(valid) / len(valid)) if valid else None
        rec["i_score"] = _improve(norm)            # robust: mean(last5) - mean(first5)
        rec["i_score_tmlr"] = _improve_tmlr(norm)  # paper: mean(last5) - trial1
        filled.append(rec)

    with open(results_path, "w") as f:                  # rewrite in place with normalization filled
        for rec in recs:
            f.write(json.dumps(rec) + "\n")

    perf = [r["performance"] for r in filled]
    iss = [r["i_score"] for r in filled]
    out = {"run": os.path.abspath(args.run), "n_normalized": len(filled),
           "missing_oracle_for": missing,
           "performance_mean": _mean(perf), "performance_se": _stderr(perf),
           "i_score_mean": _mean(iss), "i_score_se": _stderr(iss),
           "i_score_tmlr_mean": _mean([r["i_score_tmlr"] for r in filled]),
           "i_score_tmlr_se": _stderr([r["i_score_tmlr"] for r in filled])}
    with open(os.path.join(args.run, "normalized.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"[finalize] normalized {len(filled)}/{len(recs)} episodes "
          f"(missing oracle: {missing})", flush=True)
    print(f"[finalize] performance (normalized-to-oracle): {out['performance_mean']} "
          f"(SE {out['performance_se']})", flush=True)
    print(f"[finalize] I_score: {out['i_score_mean']} (SE {out['i_score_se']})", flush=True)
    for r in filled:
        print(f"  ep {r['episode_index']}: perf={r['performance']} I={r['i_score']} "
              f"agent={[round(x,1) for x in r['agent_per_trial']]} "
              f"oracle={[round(x,1) for x in r['oracle_per_trial']]}", flush=True)


if __name__ == "__main__":
    main()
