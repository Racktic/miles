"""Live (real-model) checks for CODEBASE_ACT_THINKING against a running sglang server.

Replays the user side of a real recorded ACT trial (turn-0 prompt, then
FEEDBACK/observation + next-command prompts) while the model generates each
turn with thinking enabled, exactly through the same `_ActTokenCtx` /
`_split_think` / `_pack_act_token_sample` code the rollout uses. Then verifies:

  [A] context: for every turn t>=2 the decoded prompt fed to the engine contains
      the think text of ALL earlier turns (nothing stripped) and ends with the
      thinking-mode generation prompt.
  [B] canonical form: decode(ctx.ids) == qwen3.5_fixed.jinja(clear_thinking=False)
      rendering of the same conversation (think blocks kept everywhere).
  [C] logprob replay: re-scoring the full context with the engine
      (return_logprob + logprob_start_len) reproduces the per-token logprobs
      that were returned at generation time for every earlier turn, i.e. the
      training sequence IS the sequence the model was sampled from (no
      hidden context drift). Reported as max |delta|.
  [D] sample: loss_mask==1 tokens are exactly the concatenated model outputs
      (think + answer + <|im_end|>, all trained); mask==0 tokens are only the
      injected user/feedback ChatML and contain no </think>.
  [E] think-length stats vs the 2500-token per-turn cap (finish reasons).

Run inside the miles SIF on the node hosting the server:
  apptainer exec --nv --bind /project/flame,/home/qixinx,/tmp/qixinx <miles.sif> \
    python3 examples/codebase_adaption/scripts/validate_act_thinking_live.py \
      --url http://127.0.0.1:30001 --traj <trajectory.json> --turns 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "/home/qixinx/miles")
os.environ.setdefault("CODEBASE_ACT_THINKING", "1")

import requests  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from examples.codebase_adaption.codebase_rollout import (  # noqa: E402
    _ActTokenCtx,
    _encode_prompt,
    _pack_act_token_sample,
    _split_think,
    _strip_special,
)

FIXED_TEMPLATE = "/home/qixinx/miles/miles/utils/chat_template_utils/templates/qwen3.5_fixed.jinja"


def gen(url, ids, max_new_tokens, temperature):
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": 1.0},
        "return_logprob": True,
    }
    out = requests.post(url + "/generate", json=payload, timeout=1800).json()
    meta = out["meta_info"]
    resp_ids = [t[1] for t in meta["output_token_logprobs"]]
    resp_lps = [t[0] for t in meta["output_token_logprobs"]]
    return out["text"], resp_ids, resp_lps, meta.get("finish_reason", {}).get("type")


def rescore(url, ids, start_len):
    """Prompt-token logprobs of ids[start_len:] under the prefix ids[:start_len]."""
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": start_len,
    }
    out = requests.post(url + "/generate", json=payload, timeout=1800).json()
    lps = out["meta_info"]["input_token_logprobs"]
    # sglang returns (logprob, token_id, text) per position from start_len (first may be None)
    return [(t[0], t[1]) for t in lps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30001")
    ap.add_argument("--model", default="/project/flame/qixinx/models/Qwen3.5-4B")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=2500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    state = SimpleNamespace(tokenizer=tok, args=SimpleNamespace(arc_enable_thinking=False))
    seed = SimpleNamespace(group_index=0, index=0)
    fixed_template = open(FIXED_TEMPLATE).read()

    trial = json.load(open(a.traj))["trials"][a.trial]
    turns = trial["turns"]
    n_turns = min(a.turns, len(turns))

    messages: list[dict] = []          # what codebase_rollout keeps (raw assistant text, think kept)
    ctx: _ActTokenCtx | None = None
    gen_log = []                       # per turn: dict(prompt_len, resp_ids, resp_lps, think, visible, finish)
    ok = True

    for t in range(n_turns):
        pending = []
        if t > 0:
            fb = f"FEEDBACK: {str(turns[t-1].get('observation') or '').strip()}"
            messages.append({"role": "user", "content": fb}); pending.append(fb)
        u = turns[t]["user"]
        messages.append({"role": "user", "content": u}); pending.append(u)
        if ctx is None:
            ctx = _ActTokenCtx(tok, _encode_prompt(state, messages, enable_thinking=True))
        else:
            ctx.add_user(pending)
        prompt_ids = list(ctx.ids)
        prompt_txt = tok.decode(prompt_ids)

        # [A] all earlier think blocks must be present verbatim in this turn's prompt
        assert prompt_txt.endswith("<|im_start|>assistant\n<think>\n"), "gen prompt tail wrong"
        for j, g in enumerate(gen_log):
            if g["think"] and g["think"] not in prompt_txt:
                print(f"[A] FAIL turn {t}: think of turn {j} missing from prompt"); ok = False
        text, resp_ids, resp_lps, finish = gen(a.url, prompt_ids, a.max_new_tokens, a.temperature)
        ctx.add_model(resp_ids)
        think, visible = _split_think(_strip_special(text))
        messages.append({"role": "assistant", "content": "<think>\n" + _strip_special(text)})
        gen_log.append(dict(prompt_len=len(prompt_ids), resp_ids=resp_ids, resp_lps=resp_lps,
                            think=think, visible=visible, finish=finish))
        print(f"turn {t}: prompt={len(prompt_ids)} tok, out={len(resp_ids)} tok, finish={finish}, "
              f"think_chars={len(think)}, visible_chars={len(visible)}, has_bash={'```bash' in visible}")

    print(f"[A] earlier think blocks present in later prompts: {'OK' if ok else 'FAIL'}")

    # [B] canonical: decode(ctx.ids) == fixed template rendering (thinking kept), modulo trailing "\n"
    ref = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                  chat_template=fixed_template, enable_thinking=True, clear_thinking=False)
    ours = tok.decode(ctx.ids) + "\n"
    if ours == ref:
        print("[B] decode(ctx.ids) == qwen3.5_fixed.jinja(clear_thinking=False): OK")
    else:
        k = next((i for i, (x, y) in enumerate(zip(ours, ref)) if x != y), min(len(ours), len(ref)))
        print(f"[B] MISMATCH at char {k}: ours={ours[max(0,k-80):k+80]!r}\n    ref={ref[max(0,k-80):k+80]!r}")
        ok = False

    # [C] logprob replay for every earlier turn's output under the final full context
    if len(gen_log) >= 2:
        first = gen_log[0]["prompt_len"]
        scored = rescore(a.url, ctx.ids, first)          # covers ids[first:]
        pos = 0
        worst = 0.0
        for j, g in enumerate(gen_log):
            # each turn: user injection (mask 0) then model ids (mask 1); walk ctx.mask
            pass
        # map: ctx.mask aligns with ctx.ids[first:] one-to-one; scored aligns the same way
        m = ctx.mask
        gen_lps = []
        for g in gen_log:
            gen_lps.extend(g["resp_lps"] + ([None] if len(g["resp_ids"]) == 0 or g["resp_ids"][-1] != ctx._im_end else []))
        # rebuild per-position generation logprobs by walking mask
        gi = 0
        deltas = []
        n_cmp = 0
        for i, mk in enumerate(m):
            if mk == 1:
                if gi < len(gen_lps) and gen_lps[gi] is not None and i < len(scored) and scored[i][0] is not None:
                    assert scored[i][1] == ctx.ids[first + i], "rescore token id misaligned"
                    deltas.append(abs(scored[i][0] - gen_lps[gi])); n_cmp += 1
                gi += 1
        worst = max(deltas) if deltas else float("nan")
        mean = sum(deltas) / len(deltas) if deltas else float("nan")
        print(f"[C] logprob replay over {n_cmp} generated tokens: max|delta|={worst:.4f} mean={mean:.5f}")

    # [D] sample packing
    sample = _pack_act_token_sample(seed, ctx, trial_pos=0)
    sample.validate()
    resp = sample.tokens[-sample.response_length:]
    ones = [x for x, mk in zip(resp, sample.loss_mask) if mk == 1]
    zeros = [x for x, mk in zip(resp, sample.loss_mask) if mk == 0]
    exp = []
    for g in gen_log:
        ids = list(g["resp_ids"])
        if not ids or ids[-1] != ctx._im_end:
            ids.append(ctx._im_end)
        exp.extend(ids)
    d_ok = ones == exp
    ones_txt, zeros_txt = tok.decode(ones), tok.decode(zeros)
    n_think_trained = ones_txt.count("</think>")
    print(f"[D] mask=1 == concat(model outputs): {'OK' if d_ok else 'FAIL'}; "
          f"</think> in trained region={n_think_trained}/{len(gen_log)}; </think> in mask=0 region={zeros_txt.count('</think>')}; "
          f"trained_tokens={len(ones)} env_tokens={len(zeros)} total={len(sample.tokens)}")
    ok = ok and d_ok and n_think_trained == sum(1 for g in gen_log if g["think"]) and zeros_txt.count("</think>") == 0

    # [E] think length stats
    for j, g in enumerate(gen_log):
        print(f"[E] turn {j}: out_tokens={len(g['resp_ids'])} finish={g['finish']} "
              f"think_tokens~{len(tok(g['think'], add_special_tokens=False)['input_ids'])} visible_empty={not g['visible']}")

    if a.dump:
        json.dump({"turns": [{k: v for k, v in g.items() if k != 'resp_lps'} for g in gen_log],
                   "prompt_last": tok.decode(ctx.ids[:gen_log[-1]['prompt_len']])[-3000:],
                   "trained_text_head": ones_txt[:3000]}, open(a.dump, "w"), indent=1)
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
