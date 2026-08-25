"""Offline checks for CODEBASE_ACT_THINKING (no engine, no GPU).

1. The append-only token context built by `_ActTokenCtx` must equal, byte for
   byte, what the Qwen3.5 chat template renders for the same conversation with
   thinking kept on every assistant turn (miles' `qwen3.5_fixed.jinja` with
   clear_thinking=False). That template is what miles' own TITO path uses, so
   agreeing with it means our hand-rendered ChatML is the canonical form.
2. It must differ from the *native* template only by the stripped historical
   think blocks — i.e. the native template is exactly what we are avoiding.
3. `_split_think` handles closed / unclosed / absent think blocks.
4. `_pack_act_token_sample` produces a mask whose 1-region decodes back to the
   model outputs and whose 0-region decodes back to the injected user turns.

Run inside the miles container (needs transformers + the model tokenizer):
  apptainer exec --bind /project/flame,/home/qixinx <miles.sif> \
    python3 examples/codebase_adaption/scripts/validate_act_thinking.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "/home/qixinx/miles")
os.environ.setdefault("CODEBASE_ACT_THINKING", "1")

from transformers import AutoTokenizer  # noqa: E402

from examples.codebase_adaption.codebase_rollout import (  # noqa: E402
    _ActTokenCtx,
    _encode_prompt,
    _pack_act_token_sample,
    _split_think,
)

MODEL = os.environ.get("VALIDATE_MODEL", "/project/flame/qixinx/models/Qwen3.5-4B")
FIXED_TEMPLATE = "/home/qixinx/miles/miles/utils/chat_template_utils/templates/qwen3.5_fixed.jinja"

tok = AutoTokenizer.from_pretrained(MODEL)
state = SimpleNamespace(tokenizer=tok, args=SimpleNamespace(arc_enable_thinking=False))
seed = SimpleNamespace(group_index=0, index=0)
fixed_template = open(FIXED_TEMPLATE).read()

# A 3-turn ACT trial: user → assistant(think+answer) → [FEEDBACK, user] → assistant → ...
turns = [
    ("Fix the bug in foo.py\n\n[Budget] You have 40 of 40 turns remaining for this issue.",
     "Let me look at foo.py first.\n</think>\n\nI'll inspect the file.\n\n```bash\ncat foo.py\n```"),
    ("FEEDBACK: <returncode>0</returncode>\n<output>\ndef foo():\n    return None\n</output>",
     "The function returns None; should return 42.\n</think>\n\nPatch it.\n\n```bash\nsed -i 's/None/42/' foo.py\n```"),
    ("FEEDBACK: <returncode>0</returncode>\n<output>\n</output>",
     "Done, submit.\n</think>\n\n```bash\necho submit\n```"),
]
# `messages` mirrors what codebase_rollout keeps: user(s) then raw assistant text (think kept)
messages: list[dict] = []
ctx: _ActTokenCtx | None = None
model_out_ids: list[list[int]] = []
for i, (user, assistant) in enumerate(turns):
    pending = []
    if i > 0:
        fb, u = user.split("\n<output>", 1)[0], user  # keep it simple: one user msg per turn after the first
        messages.append({"role": "user", "content": user})
        pending.append(user)
        # second user message per turn (the query + budget), like the real loop
        q = f"Repository: swesmith/x\nWhat's your next command?\n\n[Budget] You have {40-i} of 40 turns remaining for this issue."
        messages.append({"role": "user", "content": q})
        pending.append(q)
    else:
        messages.append({"role": "user", "content": user})
    if ctx is None:
        ctx = _ActTokenCtx(tok, _encode_prompt(state, messages, enable_thinking=True))
    else:
        ctx.add_user(pending)
    # what the engine would return: the assistant text + <|im_end|>
    resp_ids = tok(assistant + "<|im_end|>", add_special_tokens=False)["input_ids"]
    model_out_ids.append(resp_ids)
    ctx.add_model(resp_ids)
    messages.append({"role": "assistant", "content": "<think>\n" + assistant})

# ---- 1. equality with the fixed template (thinking kept everywhere), no gen prompt at the end
ref_text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False,
    chat_template=fixed_template, enable_thinking=True, clear_thinking=False,
)
ref_ids = tok(ref_text, add_special_tokens=False)["input_ids"]
# our ctx ends right after the last <|im_end|>; the template appends "\n" after it
ours = ctx.ids + tok("\n", add_special_tokens=False)["input_ids"]
if ours != ref_ids:
    # locate first divergence for debugging
    k = next((j for j, (a, b) in enumerate(zip(ours, ref_ids)) if a != b), min(len(ours), len(ref_ids)))
    print("MISMATCH at", k, "ours len", len(ours), "ref len", len(ref_ids))
    print("ours ...", repr(tok.decode(ours[max(0, k-30):k+30])))
    print("ref  ...", repr(tok.decode(ref_ids[max(0, k-30):k+30])))
    sys.exit(1)
print("[1] token ctx == qwen3.5_fixed.jinja(clear_thinking=False): OK  (%d tokens)" % len(ours))

# also: the mid-trial generation prompt we append equals what the template would emit
mid_msgs = messages[:-1]  # everything up to the last user turn (assistant of turn 3 removed)
ref_mid = tok(tok.apply_chat_template(mid_msgs, tokenize=False, add_generation_prompt=True,
                                      chat_template=fixed_template, enable_thinking=True, clear_thinking=False),
              add_special_tokens=False)["input_ids"]
ours_mid = ctx.ids[: len(ctx.ids) - len(model_out_ids[-1])]
assert ours_mid == ref_mid, "generation-prompt boundary mismatch"
print("[1b] mid-trial generation prompt (<|im_start|>assistant\\n<think>\\n) matches template: OK")

# ---- 2. native template strips historical think — confirm that is the thing we avoid
native_ids = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True),
                 add_special_tokens=False)["input_ids"]
native_txt = tok.decode(native_ids)
n_think_native = native_txt.count("</think>")
n_think_ours = tok.decode(ctx.ids).count("</think>")
print(f"[2] </think> blocks kept: native template={n_think_native}, ours={n_think_ours}  (native drops history: {'yes' if n_think_native < n_think_ours else 'NO?!'})")

# ---- 3. _split_think
assert _split_think("abc\n</think>\n\nvisible ```bash\nls\n```") == ("abc", "visible ```bash\nls\n```")
assert _split_think("<think>\nunclosed ```bash\nls\n```")[1] == ""
assert _split_think("no think at all") == ("", "no think at all")
print("[3] _split_think: OK")

# ---- 4. sample packing / loss mask
sample = _pack_act_token_sample(seed, ctx, trial_pos=0)
sample.validate()
resp = sample.tokens[-sample.response_length:]
ones = [t for t, m in zip(resp, sample.loss_mask) if m == 1]
zeros = [t for t, m in zip(resp, sample.loss_mask) if m == 0]
assert ones == [t for r in model_out_ids for t in r], "mask=1 region != concatenated model outputs"
zero_txt = tok.decode(zeros)
assert "<|im_start|>user" in zero_txt and "<|im_start|>assistant\n<think>\n" in zero_txt and "</think>" not in zero_txt
print(f"[4] sample: len={len(sample.tokens)} prompt={ctx.prompt_len} response={sample.response_length} "
      f"trained_tokens={sum(sample.loss_mask)} env_tokens={sample.response_length - sum(sample.loss_mask)}: OK")
print("ALL OK")
