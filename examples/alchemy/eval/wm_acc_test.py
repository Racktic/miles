"""Memory quality via GENERATION ACCURACY (more robust than teacher-forced logP).

For each candidate memory M, the policy greedily generates d_hat given (M, s, a) over the episode's CHANGED
transitions; we score two accuracies:
  - exact  : d_hat == d  (full stone incl. reward number)
  - feature: stone features (colour/size/shape) match, ignoring the reward number
            (+ a per-feature fraction, for reference)
Within-group test: G memories sampled from the CURRENT policy at one (episode, trial-k); does accuracy
PULL APART the memories (spread)? oracle / ∅ included as ceiling / floor anchors. No Claude, no logP here.

  python wm_acc_test.py --episodes 1,3,6,8 --G 8 --trial 3
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
from wm_group_test import _context  # noqa: E402  (reuse fixed-context reconstruction)
from llm_client import make_client  # noqa: E402
from prompts_eval import SUMMARY_SYSTEM, SUMMARY_INSTRUCTION  # noqa: E402
from wm_logp_experiment import WM_SYSTEMS  # noqa: E402

LOGS = "/home/qixinx/miles/examples/alchemy/logs"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
BASE = "http://127.0.0.1:30000/v1"
_STONE_RE = re.compile(r"(purple|blue)\s+(small|large)\s+(round|pointy)", re.I)
_RWD_RE = re.compile(r"reward\s*([+-]?\d+)", re.I)


def _parse_stone(text, last=False):
    text = re.sub(r"</?[A-Za-z_]+>", " ", text or "")   # strip any XML-ish tags the model may echo
    am = list(re.finditer(r"ANSWER\s*:", text, re.I))   # reason-then-answer format -> parse after ANSWER:
    if am:
        text = text[am[-1].end():]
    ms = list(_STONE_RE.finditer(text))
    if not ms:
        return None
    m = ms[-1] if last else ms[0]
    feats = (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
    rm = _RWD_RE.search(text[m.start():])
    rwd = int(rm.group(1)) if rm else None
    return feats, rwd


def score_acc(gen_client, gen_system, memory, changed):
    """Return (exact_acc, feature_acc, rwd_present_rate) over the changed transitions.
    exact = all 4 match (colour/size/shape/reward); feature = the 3 stone features match;
    rwd_present = fraction of generations that actually stated a reward number (format-compliance check)."""
    notes = memory.strip() or "(no notes yet)"
    ex = fe = rp = 0
    for (s, a, d_desc, d_rwd) in changed:
        user = f"Notes:\n{notes}\n\nYou place {s} into the {a} potion.\nResulting stone:"
        out = gen_client.complete(gen_system, [{"role": "user", "content": user}]) or ""
        p = _parse_stone(out)
        true_feats = tuple(d_desc.split())
        if p is None:
            continue
        hat_feats, hat_rwd = p
        if hat_rwd is not None:
            rp += 1
        if hat_feats == true_feats:
            fe += 1
            if hat_rwd == d_rwd:
                ex += 1
    n = len(changed)
    return ex / n, fe / n, rp / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="1,3,6,8")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--trial", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--gen-system", default="v3", help="WM system prompt for d_hat generation")
    ap.add_argument("--out", default=os.path.join(LOGS, "wm_acc_test.jsonl"))
    args = ap.parse_args()
    eps = [int(x) for x in args.episodes.split(",")]
    gen_system = WM_SYSTEMS[args.gen_system]
    print(f"[acc] gen-system={args.gen_system}", flush=True)

    sampler = make_client("openai", "qwen3.5-4b", base_url=BASE, temperature=args.temp, extra_body=NOTHINK)
    gen = make_client("openai", "qwen3.5-4b", max_tokens=32, base_url=BASE, temperature=0.0, extra_body=NOTHINK)
    fout = open(args.out, "w")
    summary = []
    for ep in eps:
        T = episode_tables(ep)
        changed = [(f"{r[0]} with reward {r[1]:+d}", r[2], r[3], r[4])
                   for r in T["transitions"] if r[5] and not r[6]]
        tmsgs, prev = _context(ep, args.trial)
        if not tmsgs:
            print(f"[acc] ep{ep}: no turns at trial {args.trial}, skip", flush=True)
            continue
        instr = (f"Your previous summary:\n{prev}\n\n{SUMMARY_INSTRUCTION}" if prev else SUMMARY_INSTRUCTION)
        # anchors
        for label, mem in [("oracle", oracle_memory(ep)), ("empty", "")]:
            ex, fe, rp = score_acc(gen, gen_system, mem, changed)
            fout.write(json.dumps({"ep": ep, "label": label, "exact": ex, "feature": fe, "rwd_present": rp}) + "\n")
            fout.flush()
            if label == "oracle":
                anchor_o = (ex, fe, rp)
            else:
                anchor_e = (ex, fe, rp)
        # G sampled memories
        gex, gfe = [], []
        for g in range(args.G):
            mem = sampler.complete(SUMMARY_SYSTEM, tmsgs + [{"role": "user", "content": instr}]) or ""
            ex, fe, rp = score_acc(gen, gen_system, mem, changed)
            gex.append(ex); gfe.append(fe)
            fout.write(json.dumps({"ep": ep, "label": f"g{g}", "exact": ex, "feature": fe,
                                   "rwd_present": rp, "mem": mem}) + "\n")
            fout.flush()
        print(f"[acc] ep{ep} ({len(changed)} changed): "
              f"oracle exact/feat={anchor_o[0]:.2f}/{anchor_o[1]:.2f} (rwd_present {anchor_o[2]:.0%})  "
              f"empty={anchor_e[0]:.2f}/{anchor_e[1]:.2f}  "
              f"| G exact[{min(gex):.2f}-{max(gex):.2f}] sd={st.pstdev(gex):.3f}  "
              f"feat[{min(gfe):.2f}-{max(gfe):.2f}] sd={st.pstdev(gfe):.3f}", flush=True)
        summary.append((ep, gex, gfe, anchor_o, anchor_e))
    fout.close()
    print("\n=== within-group spread (the GRPO-relevant question) ===")
    for ep, gex, gfe, ao, ae in summary:
        print(f"  ep{ep}: exact sd={st.pstdev(gex):.3f} range {max(gex)-min(gex):.2f} | "
              f"feature sd={st.pstdev(gfe):.3f} range {max(gfe)-min(gfe):.2f}")
    if summary:
        print(f"\nmean within-group sd: exact={st.mean([st.pstdev(s[1]) for s in summary]):.3f}  "
              f"feature={st.mean([st.pstdev(s[2]) for s in summary]):.3f}")


if __name__ == "__main__":
    main()
