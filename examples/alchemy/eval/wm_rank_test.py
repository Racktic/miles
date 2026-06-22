"""Realistic memory-ranking test: does logP rank REAL (varying-quality) memories correctly?

This is the property GRPO actually consumes (absolute level / ∅-baseline cancel in the group advantage).
Per episode we score, on the CHANGED transitions, a spread of real memories (Claude / Qwen-structured /
Qwen-freeform, at M1/M5/M9) plus oracle/∅ anchors, and check:
  (1) tier ranking: oracle > claude > qwen-struct > qwen-freeform > ∅ ?
  (2) Qwen-vs-Qwen: struct_M9 > freeform_M9 ? (what GRPO compares: Qwen vs Qwen)
  (3) does logP track an INDEPENDENT objective quality (# of potion rules the memory states correctly)?

System prompt = v2 (the validated config). Uses wm_logp_experiment.score_logp.
"""
from __future__ import annotations

import json
import os
import sys
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wm_logp_experiment as W  # noqa: E402
from wm_probe import episode_tables, oracle_memory, potion_effects  # noqa: E402

LOGS = "/home/qixinx/miles/examples/alchemy/logs"
RUNS = {
    "claude":   "eval-claude-opus-4-8-sum-replace-noprior-20260614-234739",
    "qwenStruct": "eval-qwen3.5-4b-sum-replace-noprior-20260614-231736",
    "qwenFree": "eval-qwen3.5-4b-sum-replace-noprior-20260614-010136",
}
MIDX = [1, 5, 9]  # which M_k (1-based) to sample from each run


def _summaries(run, ep):
    p = os.path.join(LOGS, run, "traj", f"ep{ep}.json")
    return json.load(open(p))["summaries"]


def rules_correct(mem: str, truth: dict) -> int:
    """# of potion colours whose TRUE effect (frm->to on some attr) is stated in the memory text."""
    m = mem.lower()
    n = 0
    for col, (attr, frm, to) in truth.items():
        if attr is None:
            continue
        if col in m and frm in m and to in m:   # lenient: colour + from-value + to-value all present
            n += 1
    return n


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j < len(v) and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                r[order[k]] = (i + j - 1) / 2 + 1
            i = j
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def main():
    W.WM_SYSTEM = W.WM_SYSTEMS["v2"]
    eps = list(json.load(open("/home/qixinx/miles/examples/alchemy/data/hard_set_20.json"))["episodes"])
    out = open(os.path.join(LOGS, "wm_rank_test.jsonl"), "w")
    rows = []
    for ep in eps:
        T = episode_tables(ep)
        changed = [(f"{r[0]} with reward {r[1]:+d}", r[2], f"{r[3]} with reward {r[4]:+d}")
                   for r in T["transitions"] if r[5] and not r[6]]
        if not changed:
            continue
        truth = potion_effects(ep)
        conds = {"oracle": (oracle_memory(ep), None), "empty": ("", 0)}
        for rl, run in RUNS.items():
            sm = _summaries(run, ep)
            for k in MIDX:
                if k - 1 < len(sm):
                    mem = sm[k - 1]
                    conds[f"{rl}_M{k}"] = (mem, rules_correct(mem, truth))
        for label, (mem, corr) in conds.items():
            lps = [W.score_logp(mem, s, a, d)[0] for (s, a, d) in changed]
            rec = {"ep": ep, "label": label, "logp": sum(lps) / len(lps),
                   "correct": corr, "n_changed": len(changed)}
            rows.append(rec)
            out.write(json.dumps(rec) + "\n")
            out.flush()
        print(f"[rank] ep{ep} done ({len(changed)} changed x {len(conds)} memories)", flush=True)
    out.close()

    # ---- analysis ----
    def tier(lbl):
        return lbl if lbl in ("oracle", "empty") else lbl.split("_")[0]
    print("\n=== (1) mean logP by tier (CHANGED), aggregated over episodes ===")
    tiers = {}
    for r in rows:
        tiers.setdefault(tier(r["label"]), []).append(r["logp"])
    for t in ["oracle", "claude", "qwenStruct", "qwenFree", "empty"]:
        if t in tiers:
            print(f"  {t:11s}: {st.mean(tiers[t]):+.3f}  (n={len(tiers[t])})")

    print("\n=== (2) Qwen-vs-Qwen paired: structured_M9 vs freeform_M9 (per episode) ===")
    byep = {}
    for r in rows:
        byep.setdefault(r["ep"], {})[r["label"]] = r["logp"]
    d = [(v["qwenStruct_M9"] - v["qwenFree_M9"]) for v in byep.values()
         if "qwenStruct_M9" in v and "qwenFree_M9" in v]
    if d:
        m = st.mean(d); se = (st.pvariance(d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else 0
        print(f"  Δ(struct−free) M9: {m:+.3f} ± {se:.3f}  t={m/se if se else 0:+.2f}  "
              f"{sum(1 for x in d if x > 0)}/{len(d)} >0")

    print("\n=== (3) Spearman(logP, #rules-correct) over REAL memories ===")
    real = [r for r in rows if r["label"] not in ("oracle", "empty")]
    rho = spearman([r["logp"] for r in real], [r["correct"] for r in real])
    print(f"  rho = {rho:+.3f}  (n={len(real)} real memories)")
    # correctness buckets
    bk = {}
    for r in real:
        bk.setdefault(r["correct"], []).append(r["logp"])
    print("  mean logP by #rules-correct:")
    for c in sorted(bk):
        print(f"    {c} correct: {st.mean(bk[c]):+.3f}  (n={len(bk[c])})")


if __name__ == "__main__":
    main()
