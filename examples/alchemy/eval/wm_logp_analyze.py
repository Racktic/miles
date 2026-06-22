"""Analyze the L_WM logP control experiment (wm_logp_experiment.py output).

Answers: (1) does oracle memory raise logP vs ∅ (signal)?  (2) does mismatched memory fall ≤ ∅?
(3) is ∅ a sound baseline (does subtracting it cut variance / track difficulty)?
"""
import argparse
import json
import math
import statistics as st


def stats(xs):
    xs = list(xs)
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, (var / len(xs)) ** 0.5


def auc(pos, neg):
    """P(a random pos > a random neg) via Mann-Whitney; pos should score higher if signal exists."""
    pos, neg = list(pos), list(neg)
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j - 1) / 2 + 1
        for k in range(i, j):
            rank[k] = r
        i = j
    rsum = sum(rank[k] for k, (v, lbl) in enumerate(allv) if lbl == 1)
    n1, n2 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n2)


def cohend(a, b):
    a, b = list(a), list(b)
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.pvariance(a), st.pvariance(b)
    sp = math.sqrt((va + vb) / 2) or 1e-9
    return (ma - mb) / sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="/home/qixinx/miles/examples/alchemy/logs/wm_logp_control.jsonl")
    ap.add_argument("--sum", action="store_true", help="use sum-logP (mean*n) instead of per-token mean")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]

    def val(r, cond):
        return r[cond] * r[cond + "_n"] if args.sum else r[cond]

    def report(sub, label):
        if not sub:
            print(f"\n=== {label}: (none) ==="); return
        emp = [val(r, "empty") for r in sub]
        ora = [val(r, "oracle") for r in sub]
        mis = [val(r, "mismatched") for r in sub]
        d_o = [o - e for o, e in zip(ora, emp)]
        d_m = [m - e for m, e in zip(mis, emp)]
        print(f"\n=== {label}  (n={len(sub)}) {'[sum-logP]' if args.sum else '[mean-logP]'} ===")
        print(f"  logP:    empty {stats(emp)[0]:+.3f}   oracle {stats(ora)[0]:+.3f}   mismatched {stats(mis)[0]:+.3f}")
        mo, so = stats(d_o); mm, sm = stats(d_m)
        print(f"  Δoracle (oracle−empty):     {mo:+.3f} ± {so:.3f}   t={mo/so if so else 0:+.2f}   "
              f"{sum(1 for x in d_o if x>0)}/{len(d_o)} >0")
        print(f"  Δmismatched (mis−empty):    {mm:+.3f} ± {sm:.3f}   t={mm/sm if sm else 0:+.2f}   "
              f"{sum(1 for x in d_m if x>0)}/{len(d_m)} >0")
        print(f"  AUC(oracle>empty)={auc(ora, emp):.3f}   AUC(oracle>mismatched)={auc(ora, mis):.3f}   "
              f"Cohen d(oracle,empty)={cohend(ora, emp):+.2f}")
        # baseline soundness: does subtracting empty cut variance vs raw oracle?
        vo = st.pvariance(ora); vd = st.pvariance(d_o)
        print(f"  Var(oracle)={vo:.3f}  Var(oracle−empty)={vd:.3f}  -> baseline {'CUTS' if vd<vo else 'RAISES'} variance "
              f"({100*(1-vd/vo):+.0f}%)")

    report(rows, "ALL transitions")
    report([r for r in rows if r["changed"] and not r["blocked"]], "CHANGED (potion acts)")
    report([r for r in rows if not r["changed"]], "NO-OP (incl. blocked + already-at-target)")
    report([r for r in rows if r["blocked"]], "BLOCKED (would act but bottleneck)")


if __name__ == "__main__":
    main()
