"""Large-scale validation of the ACCURACY reward, on the training-faithful test set.

For each (episode, trial-k): the test set F_k = the real CHANGED potion-application transitions OBSERVED in
trials > k of the run (NOT the full table). We compute generation accuracy (exact = colour/size/shape/reward
all match; feature = the 3 stone features match) for several memories on that F_k:
  - oracle (perfect, ground truth)      -> ceiling
  - empty (no memory)                   -> floor
  - real M_k from the run               -> the actual mediocre memory
  - G memories sampled from the current policy at that context  -> the GRPO group
Question: does accuracy SEPARATE these (oracle >> empty; spread across qualities; within-group spread)?
No judge — accuracy IS the quality measure. All local Qwen, greedy temp=0.

  python wm_group_test2.py --episodes-file ../data/hard_set_20.json --trials 1,2,3,4 --G 8
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

from wm_probe import episode_tables, oracle_memory  # noqa: E402
from wm_group_test import _context  # noqa: E402
from wm_acc_test import _parse_stone, NOTHINK, BASE  # noqa: E402
from llm_client import make_client  # noqa: E402
from prompts_eval import SUMMARY_SYSTEM, SUMMARY_INSTRUCTION  # noqa: E402
from wm_logp_experiment import WM_SYSTEMS  # noqa: E402

LOGS = "/home/qixinx/miles/examples/alchemy/logs"
QWEN_RUN = "eval-qwen3.5-4b-sum-replace-noprior-20260614-231736"
GEN_SYS = WM_SYSTEMS["v4"]   # reason (REASONING/ANSWER) then answer, enable_thinking=False


def _observed_future(ep, k):
    """F_k = unique (s_desc, colour, d_desc, d_rwd, changed) observed in trials > k of the run.
    INCLUDES no-ops (d==s): predicting a no-op is itself a real test of memory (does it know what is
    blocked/inert?). s/colour from the turn's rendered state (by index); d looked up from ground truth."""
    T = episode_tables(ep)
    lut = {(tuple(r[0].split()), r[2]): (r[3], r[4], r[5]) for r in T["transitions"]}
    tj = json.load(open(os.path.join(LOGS, QWEN_RUN, "traj", f"ep{ep}.json")))
    seen, out = set(), []
    for m in tj["transcript"]:
        if m["trial"] <= k or m["parsed"].get("kind") != "potion":
            continue
        si, pi = m["parsed"].get("stone"), m["parsed"].get("potion")
        stones, potions, mode = {}, {}, None
        for line in m["user"].split("\n"):
            if "following stones" in line: mode = "s"; continue
            if "following potions" in line: mode = "p"; continue
            mm = re.match(r"\s*(\d+):\s*(.+)", line)
            if mm:
                (stones if mode == "s" else potions)[int(mm.group(1))] = mm.group(2).strip()
        if si not in stones or pi not in potions:
            continue
        feats = tuple(stones[si].split()[:3]); col = potions[pi]
        key = (feats, col)
        if key in lut and key not in seen:                   # changed AND no-op, dedup
            seen.add(key)
            rwd = stones[si].split("reward")[-1].strip()
            out.append((f"{feats[0]} {feats[1]} {feats[2]} with reward {rwd}", col,
                        lut[key][0], lut[key][1], lut[key][2]))
    return out


def score(gen, memory, Fk):
    """(exact_acc, feature_acc) over F_k (5-tuples; 5th = changed flag, ignored here)."""
    notes = memory.strip() or "(no notes yet)"
    if not Fk:
        return None, None
    ex = fe = 0
    for (s, a, d_desc, d_rwd, _ch) in Fk:
        user = f"Notes:\n{notes}\n\nYou place {s} into the {a} potion.\nResulting stone:"
        p = _parse_stone(gen.complete(GEN_SYS, [{"role": "user", "content": user}]) or "")
        if p and p[0] == tuple(d_desc.split()):
            fe += 1
            if p[1] == d_rwd:
                ex += 1
    return ex / len(Fk), fe / len(Fk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-file", default="/home/qixinx/miles/examples/alchemy/data/hard_set_20.json")
    ap.add_argument("--episodes", default=None)
    ap.add_argument("--trials", default="1,2,3,4")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--out", default=os.path.join(LOGS, "wm_group_test2.jsonl"))
    args = ap.parse_args()
    eps = ([int(x) for x in args.episodes.split(",")] if args.episodes
           else list(json.load(open(args.episodes_file))["episodes"]))
    ks = [int(x) for x in args.trials.split(",")]

    sampler = make_client("openai", "qwen3.5-4b", base_url=BASE, temperature=args.temp, extra_body=NOTHINK)
    gen = make_client("openai", "qwen3.5-4b", max_tokens=384, base_url=BASE, temperature=0.0, extra_body=NOTHINK)
    fout = open(args.out, "w")
    groups = []
    for ep in eps:
        om = oracle_memory(ep)
        tj = json.load(open(os.path.join(LOGS, QWEN_RUN, "traj", f"ep{ep}.json")))
        for k in ks:
            Fk = _observed_future(ep, k)
            tmsgs, prev = _context(ep, k)
            if len(Fk) < 3 or not tmsgs:
                print(f"[g2] ep{ep} k{k}: |F_k|={len(Fk)} skip", flush=True)
                continue
            instr = (f"Your previous summary:\n{prev}\n\n{SUMMARY_INSTRUCTION}" if prev else SUMMARY_INSTRUCTION)
            real_mk = tj["summaries"][k] if k < len(tj["summaries"]) else ""   # the run's own M_k
            chg = [t for t in Fk if t[4]]; nop = [t for t in Fk if not t[4]]
            anchors = {}
            for lbl, mem in [("oracle", om), ("empty", ""), ("realMk", real_mk)]:
                ex, fe = score(gen, mem, Fk)
                cx = score(gen, mem, chg)[0] if chg else None      # exact on changed subset
                nx = score(gen, mem, nop)[0] if nop else None      # exact on no-op subset
                anchors[lbl] = (ex, fe, cx, nx)
                fout.write(json.dumps({"ep": ep, "k": k, "label": lbl, "exact": ex, "feature": fe,
                                       "exact_changed": cx, "exact_noop": nx, "nFk": len(Fk),
                                       "n_changed": len(chg), "n_noop": len(nop)}) + "\n")
            gacc = []
            for g in range(args.G):
                mem = sampler.complete(SUMMARY_SYSTEM, tmsgs + [{"role": "user", "content": instr}]) or ""
                ex, fe = score(gen, mem, Fk); gacc.append((ex, fe))
                fout.write(json.dumps({"ep": ep, "k": k, "label": f"g{g}", "exact": ex, "feature": fe,
                                       "nFk": len(Fk), "mem": mem}) + "\n")
                fout.flush()
            gex = [a for a, b in gacc]
            print(f"[g2] ep{ep} k{k} |F|={len(Fk)}(chg {len(chg)}/nop {len(nop)}): "
                  f"oracle ex={anchors['oracle'][0]:.2f}(chg {anchors['oracle'][2]}/nop {anchors['oracle'][3]})  "
                  f"empty ex={anchors['empty'][0]:.2f}(chg {anchors['empty'][2]}/nop {anchors['empty'][3]})  "
                  f"realMk={anchors['realMk'][0]:.2f} | G ex[{min(gex):.2f}-{max(gex):.2f}] sd={st.pstdev(gex):.3f}", flush=True)
            groups.append((ep, k, anchors, gacc))
    fout.close()

    print("\n=== SUMMARY (n groups = %d) ===" % len(groups))
    if groups:
        o = st.mean([g[2]['oracle'][0] for g in groups]); e = st.mean([g[2]['empty'][0] for g in groups])
        rm = st.mean([g[2]['realMk'][0] for g in groups]); gm = st.mean([st.mean([a for a, b in g[3]]) for g in groups])
        of = st.mean([g[2]['oracle'][1] for g in groups]); ef = st.mean([g[2]['empty'][1] for g in groups])
        rmf = st.mean([g[2]['realMk'][1] for g in groups]); gmf = st.mean([st.mean([b for a, b in g[3]]) for g in groups])
        print(f"EXACT   mean acc(full F_k): oracle={o:.3f}  realMk={rm:.3f}  G-sampled={gm:.3f}  empty={e:.3f}")
        print(f"FEATURE mean acc(full F_k): oracle={of:.3f}  realMk={rmf:.3f}  G-sampled={gmf:.3f}  empty={ef:.3f}")

        def _m(lbl, idx):
            v = [g[2][lbl][idx] for g in groups if g[2][lbl][idx] is not None]
            return st.mean(v) if v else float("nan")
        print("--- exact split (does empty trivially get no-ops? does memory help on changed?) ---")
        print(f"  CHANGED subset: oracle={_m('oracle',2):.3f}  realMk={_m('realMk',2):.3f}  empty={_m('empty',2):.3f}")
        print(f"  NO-OP   subset: oracle={_m('oracle',3):.3f}  realMk={_m('realMk',3):.3f}  empty={_m('empty',3):.3f}")
        print(f"mean within-group sd(full): exact={st.mean([st.pstdev([a for a,b in g[3]]) for g in groups]):.3f}  "
              f"feature={st.mean([st.pstdev([b for a,b in g[3]]) for g in groups]):.3f}")


if __name__ == "__main__":
    main()
