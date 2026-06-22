"""Faithful within-group test (L_WM_DESIGN §7.6): sample G memories from the CURRENT policy at ONE
(episode, trial-k) context, and check whether the logP reward (Qwen scorer) ranks them in line with an
INDEPENDENT ground-truth quality (Claude judge, given the true chemistry). This is exactly the group GRPO
consumes. Small scale to save API.

  python wm_group_test.py --episodes 1,3,6,8 --G 8 --trial 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wm_logp_experiment as W  # noqa: E402
from wm_probe import episode_tables, potion_effects  # noqa: E402
from llm_client import make_client  # noqa: E402
from prompts_eval import SUMMARY_SYSTEM, SUMMARY_INSTRUCTION  # noqa: E402

LOGS = "/home/qixinx/miles/examples/alchemy/logs"
QWEN_RUN = "eval-qwen3.5-4b-sum-replace-noprior-20260614-231736"   # source of the fixed context
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def _context(ep, t):
    """Reconstruct (trial_messages, prev_summary) at the REWRITE after trial index t, from the run."""
    tj = json.load(open(os.path.join(LOGS, QWEN_RUN, "traj", f"ep{ep}.json")))
    turns = [m for m in tj["transcript"] if m["trial"] == t]
    tmsgs = []
    for m in turns:
        tmsgs.append({"role": "user", "content": m["user"]})
        tmsgs.append({"role": "assistant", "content": m["assistant"]})
    prev = tj["summaries"][t - 1] if t - 1 < len(tj["summaries"]) and t >= 1 else None
    return tmsgs, prev


def _truth_text(ep):
    eff = potion_effects(ep)
    lines = ["TRUE chemistry of this episode:"]
    for col, (attr, frm, to) in sorted(eff.items()):
        if attr is None:
            lines.append(f"- {col}: no effect on any stone (inert).")
        else:
            lines.append(f"- {col}: changes a stone's {attr} from {frm} to {to} (only when {frm}; "
                          f"may be blocked on some stones by the bottleneck).")
    T = episode_tables(ep)
    best = max(T["surf"].items(), key=lambda kv: kv[1][1])
    lines.append(f"- Highest-reward stone: {best[1][0]} ({best[1][1]:+d}).")
    return "\n".join(lines)


JUDGE_SYS = ("You grade an agent's free-text notes about a hidden chemistry against the TRUE chemistry. "
             "Score how well the notes would let someone act optimally: reward correct potion effects and "
             "the correct best-stone; penalize wrong/contradictory claims, drift ('never transform', "
             "'potions don't help'), and clutter. Output ONLY JSON: {\"score\": <integer 0-10>}.")


def judge(claude, ep_truth, memory):
    user = f"{ep_truth}\n\nAGENT NOTES:\n{memory}\n\nScore the notes 0-10. Output ONLY {{\"score\": N}}."
    out = claude.complete(JUDGE_SYS, [{"role": "user", "content": user}]) or ""
    m = re.search(r'"score"\s*:\s*(-?\d+)', out)
    return int(m.group(1)) if m else None


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j < len(v) and v[o[j]] == v[o[i]]:
                j += 1
            for k in range(i, j):
                r[o[k]] = (i + j - 1) / 2 + 1
            i = j
        return r
    rx, ry = rk(xs), rk(ys); n = len(xs); mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5; sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="1,3,6,8")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--trial", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--out", default=os.path.join(LOGS, "wm_group_test.jsonl"))
    args = ap.parse_args()
    W.WM_SYSTEM = W.WM_SYSTEMS["v2"]
    eps = [int(x) for x in args.episodes.split(",")]

    qwen = make_client("openai", "qwen3.5-4b", base_url="http://127.0.0.1:30000/v1",
                       temperature=args.temp, extra_body=NOTHINK)
    claude = make_client("anthropic", "claude-opus-4-8")
    fout = open(args.out, "w")
    all_groups = []
    for ep in eps:
        T = episode_tables(ep)
        changed = [(f"{r[0]} with reward {r[1]:+d}", r[2], f"{r[3]} with reward {r[4]:+d}")
                   for r in T["transitions"] if r[5] and not r[6]]
        tmsgs, prev = _context(ep, args.trial)
        if not tmsgs:
            print(f"[group] ep{ep}: no turns at trial {args.trial}, skip", flush=True)
            continue
        truth = _truth_text(ep)
        instr = (f"Your previous summary:\n{prev}\n\n{SUMMARY_INSTRUCTION}" if prev else SUMMARY_INSTRUCTION)
        group = []
        for g in range(args.G):
            mem = qwen.complete(SUMMARY_SYSTEM, tmsgs + [{"role": "user", "content": instr}]) or ""
            lp = st.mean([W.score_logp(mem, s, a, d)[0] for (s, a, d) in changed])
            q = judge(claude, truth, mem)
            rec = {"ep": ep, "g": g, "logp": lp, "judge": q, "mem_len": len(mem)}
            group.append(rec); fout.write(json.dumps(rec) + "\n"); fout.flush()
        rho = spearman([r["logp"] for r in group], [r["judge"] for r in group])
        lp_sd = st.pstdev([r["logp"] for r in group])
        jr = [r["judge"] for r in group if r["judge"] is not None]
        print(f"[group] ep{ep}: G={len(group)}  logP sd={lp_sd:.3f}  judge range={min(jr)}-{max(jr)}  "
              f"within-group rho(logP,judge)={rho}", flush=True)
        all_groups.append((ep, rho, group))
    fout.close()

    print("\n=== summary ===")
    rhos = [r for _, r, _ in all_groups if r is not None]
    if rhos:
        print(f"per-group rho(logP,judge): {[round(r,2) for r in rhos]}  mean={st.mean(rhos):+.3f}")
    pooled = [(r["logp"], r["judge"]) for _, _, grp in all_groups for r in grp]
    pr = spearman([a for a, b in pooled], [b for a, b in pooled])
    print(f"pooled rho={pr}  (n={len(pooled)})")


if __name__ == "__main__":
    main()
