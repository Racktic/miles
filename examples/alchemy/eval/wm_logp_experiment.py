"""L_WM signal-validation: does logP(d | memory, s, a) separate good/bad memory, and is logP(d|∅) a
sound baseline? Control conditions only: ∅ / oracle-memory / mismatched-memory.

Scorer = the online policy itself (the served Qwen3.5-4B), teacher-forced: we build [system,user] via the
model's chat template, append the gold continuation d, and read sglang's input_token_logprobs over the d
tokens (max_new_tokens=0). Per-token mean logP.

Usage:
  # alignment self-test on one transition (no full run):
  python wm_logp_experiment.py --test
  # full control run over hard-20:
  python wm_logp_experiment.py --episodes-file ../data/hard_set_20.json --out /tmp/wm_logp.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wm_probe import episode_tables, oracle_memory, oracle_memory_table  # noqa: E402

SERVER = os.environ.get("WM_SERVER", "http://127.0.0.1:30000")
CKPT = os.environ.get("WM_CKPT", "/data/user_data/qixinx/Qwen3.5-4B")

WM_SYSTEMS = {
    # v1: original — mentions "or have no effect" (may inflate the no-op prior under ∅)
    "v1": (
        "You are predicting outcomes in the game Alchemy. Stones have a colour, size, and shape, and a "
        "reward value. A potion may change some of a stone's features, or have no effect. Using ONLY the "
        "notes below about THIS episode's chemistry, predict the resulting stone."
    ),
    # v2: drop "or have no effect"; add an explicit output-format constraint
    "v2": (
        "You are predicting the outcome in the game Alchemy. Each stone is written as "
        "'<colour> <size> <shape> with reward <N>'. Using ONLY the notes below about this episode's "
        "chemistry, predict the stone that results when the given potion is applied to the given stone. "
        "Answer with exactly one stone, in that same format."
    ),
    # v4: enable_thinking=False (like the whole pipeline), but ALLOW free-text reasoning via an explicit
    # REASONING/ANSWER format — mirrors the actor's OBSERVATION/REASONING/ACTION rather than answering cold.
    "v4": (
        "You are predicting the outcome in the game Alchemy. A stone is described by its colour, then size, "
        "then shape, then the words 'with reward' followed by a signed integer — the same way stones appear "
        "in the notes and the prompt. Using ONLY the notes below about this episode's chemistry, work out "
        "what stone results when the given potion is applied to the given stone. Reply in EXACTLY this "
        "format:\nREASONING: <your brief reasoning>\nANSWER: <colour> <size> <shape> with reward <N>\n"
        "The ANSWER line must be the resulting stone alone in plain words, and must include 'with reward <N>'."
    ),
    # v3: like v2 but HARD-require the reward part (the model often drops it under v2).
    # NOTE: do NOT use angle-bracket placeholders here — the model echoes them as literal XML tags.
    "v3": (
        "You are predicting the outcome in the game Alchemy. A stone is described by its colour, then size, "
        "then shape, then the words 'with reward' followed by a signed integer — exactly the way stones "
        "appear in the notes and the prompt. Using ONLY the notes below about this episode's chemistry, "
        "predict the resulting stone when the potion is applied to the given stone. Output ONLY one line in "
        "plain words (no tags or placeholders), and you MUST end with 'with reward' and the integer — never "
        "omit the reward."
    ),
}
WM_SYSTEM = WM_SYSTEMS["v1"]   # default; overridden in main() by --system

_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    return _TOK


def _build_ids(memory: str, s: str, a: str, d: str):
    """Return (full_ids, prompt_len, target_ids). Use the model's real chat template (incl. the
    no-thinking <think></think> scaffold — on-distribution), but tokenize prompt+target as ONE string
    so d is tokenized in context (no spurious leading-space token, clean boundary). target = d tokens."""
    tok = _tok()
    notes = memory.strip() if memory.strip() else "(no notes yet)"
    user = f"Notes:\n{notes}\n\nYou place {s} into the {a} potion.\nResulting stone:"
    msgs = [{"role": "system", "content": WM_SYSTEM}, {"role": "user", "content": user}]
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                          enable_thinking=False)
    full_text = prompt_text + d
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(full_text, add_special_tokens=False)
    plen = len(prompt_ids)
    if full_ids[:plen] != prompt_ids:               # safety: fall back to longest common prefix
        plen = next((i for i in range(min(len(prompt_ids), len(full_ids)))
                     if prompt_ids[i] != full_ids[i]), min(len(prompt_ids), len(full_ids)))
    return full_ids, plen, full_ids[plen:]


def score_logp(memory: str, s: str, a: str, d: str):
    """Per-token mean logP of d under (memory, s, a). Returns (mean_logp, n_tokens, aligned_ok)."""
    full_ids, plen, target_ids = _build_ids(memory, s, a, d)
    r = requests.post(f"{SERVER}/generate", json={
        "input_ids": full_ids,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": plen - 1,     # so the first returned logprob is for token at position plen
    }, timeout=120)
    r.raise_for_status()
    meta = r.json()["meta_info"]
    itl = meta["input_token_logprobs"]     # list of [logprob, token_id, token_text]
    # entries correspond to positions [logprob_start_len+1 .. end] = the target tokens
    tail = itl[-len(target_ids):]
    got_ids = [e[1] for e in tail]
    aligned = (got_ids == target_ids)
    lps = [e[0] for e in tail]
    return sum(lps) / len(lps), len(lps), aligned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--episodes-file", default=None)
    ap.add_argument("--episodes", default=None, help="comma list, overrides file")
    ap.add_argument("--system", choices=list(WM_SYSTEMS), default="v1")
    ap.add_argument("--oracle-style", choices=["rules", "table"], default="rules")
    ap.add_argument("--out", default="/tmp/wm_logp.jsonl")
    args = ap.parse_args()

    global WM_SYSTEM
    WM_SYSTEM = WM_SYSTEMS[args.system]
    om_fn = oracle_memory_table if args.oracle_style == "table" else oracle_memory
    print(f"[wm] system={args.system} oracle-style={args.oracle_style} -> {args.out}", flush=True)

    if args.test:
        T = episode_tables(55)
        s_desc, s_rwd, col, d_desc, d_rwd, changed, blocked = next(r for r in T["transitions"] if r[5])
        s = f"{s_desc} with reward {s_rwd:+d}"
        d = f"{d_desc} with reward {d_rwd:+d}"
        om = oracle_memory(55)
        print(f"TEST transition: {s}  +  {col}  ->  {d}")
        full_ids, plen, tgt = _build_ids(om, s, col, d)
        print(f"prompt_len={plen}  target_tokens={len(tgt)}  target_ids={tgt}")
        print("target decoded:", repr(_tok().decode(tgt)))
        for cond, mem in [("oracle", om), ("empty", ""), ("mismatched", oracle_memory(1))]:
            lp, n, ok = score_logp(mem, s, col, d)
            print(f"  {cond:11s}: mean_logP={lp:+.4f}  n={n}  aligned={ok}")
        return

    eps = ([int(x) for x in args.episodes.split(",")] if args.episodes
           else list(json.load(open(args.episodes_file))["episodes"]))
    others = {e: eps[(i + 1) % len(eps)] for i, e in enumerate(eps)}  # mismatched source
    fout = open(args.out, "w")
    n_done = n_misalign = 0
    for ep in eps:
        T = episode_tables(ep)
        om = om_fn(ep)
        mm = om_fn(others[ep])
        for s_desc, s_rwd, col, d_desc, d_rwd, changed, blocked in T["transitions"]:
            s = f"{s_desc} with reward {s_rwd:+d}"
            d = f"{d_desc} with reward {d_rwd:+d}"
            rec = {"ep": ep, "s": s, "a": col, "d": d, "changed": changed, "blocked": blocked}
            for cond, mem in [("empty", ""), ("oracle", om), ("mismatched", mm)]:
                lp, n, ok = score_logp(mem, s, col, d)
                rec[cond] = lp
                rec[cond + "_n"] = n
                if not ok:
                    n_misalign += 1
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1
        print(f"[wm] ep{ep} done ({len(T['transitions'])} transitions)", flush=True)
    fout.close()
    print(f"[wm] DONE {n_done} transitions -> {args.out}  (misaligned scorings: {n_misalign})", flush=True)


if __name__ == "__main__":
    main()
