"""Symbolic Alchemy two-phase (ACT -> REWRITE) TEXT rollout for the miles GRPO loop.

Same control flow as ``examples/arc_agi3/arc_rollout_objects.py`` (ACT -> env.step -> REWRITE; one
episode -> list[Sample]; episode-level GRPO), but over ``AlchemyTextEnv``: the state is a symbolic
text dump (``obs_to_text``) and the transition an exact symbolic diff (``obs_diff_text``). No images,
no tracker — the obs is already symbolic and the transition d_t is noise-free, which is the whole
point of moving here from ARC. Selected via
``--custom-generate-function-path examples.alchemy.alchemy_rollout.generate``.

Episode reward = cumulative env reward (dense: cauldron drops pay the stone's value). All samples in
an episode share that reward and an ``episode_id`` so the GRPO post-processor broadcasts one
advantage across the episode's ACT+REWRITE samples (same contract as the ARC path).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from collections import defaultdict
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.mask_utils import MultiTurnLossMaskGenerator
from miles.utils.types import Sample

from examples.alchemy.alchemy_judge import judge_explore
from examples.alchemy.env_alchemy import (
    AlchemyTextEnv,
    _potion_desc,
    _stone_desc,
    decode_obs,
)
# train == eval: ONE source of truth for the no-prior + summary prompts/rendering/parsing. prompts_eval
# only depends on env_alchemy (no llm_client/oracle), so it is safe to import in the ray rollout worker.
from examples.alchemy.eval.prompts_eval import (
    SUMMARY_INSTRUCTION,
    SUMMARY_INSTRUCTION_FREEFORM,
    SUMMARY_SYSTEM,
    build_training_system,
    parse_eval_action,
    render_game_state,
)

EVAL_SYSTEM = build_training_system(prior_info=False)   # no-prior eval prompt + Qwen3-4B exploration hint


class _AlchemyMaskGen(MultiTurnLossMaskGenerator):
    """Standard multi-turn loss mask (one trial = one packed sequence, loss only on assistant tokens),
    with one Qwen3.5 fix: the template injects an empty <think>\\n\\n</think>\\n\\n into each assistant
    turn that the model did NOT generate; the parent marks it loss=1, so we zero those tokens. Verified
    on real trials: the loss=1 spans then decode exactly to the model's generated OBSERVATION/REASONING/
    ACTION text."""

    _think_seqs = None

    def get_loss_mask(self, messages, tools=None):
        ids, mask = super().get_loss_mask(messages, tools=tools)
        if self._think_seqs is None:
            self._think_seqs = [self.tokenizer(s, add_special_tokens=False)["input_ids"]
                                for s in ("<think>\n\n</think>\n\n", "<think>\n\n</think>")]
        for sub in self._think_seqs:
            L = len(sub)
            if not L:
                continue
            i = 0
            while i <= len(ids) - L:
                if ids[i:i + L] == sub:
                    for j in range(i, i + L):
                        mask[j] = 0
                    i += L
                else:
                    i += 1
        return ids, mask


_ACT_MASK_GEN = None
_PACK_CHECKED = False


def _strip_special(t: str) -> str:
    """sglang returns generated text WITH the trailing <|im_end|> (unlike the eval API, which is clean).
    Strip special end tokens before putting text into the chat history / summary, so (a) re-rendering
    doesn't double the <|im_end|>, and (b) history matches the eval setting (clean assistant content)."""
    return (t or "").replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def _pack_act_sample(state, seed, system, trial_messages, trial_pos):
    """Pack one trial's whole chat into a SINGLE ACT Sample: tokens = full trial conversation, loss_mask
    = 1 only on the model's generated assistant tokens (think-block injection zeroed). rollout_log_probs
    is None (use_rollout_logprobs=False -> training recomputes log_probs). Returns None if no assistant."""
    global _ACT_MASK_GEN
    if not any(m["role"] == "assistant" for m in trial_messages):
        return None
    if _ACT_MASK_GEN is None:
        _ACT_MASK_GEN = _AlchemyMaskGen(state.tokenizer, tokenizer_type="qwen3")
    msgs = [{"role": "system", "content": system}] + trial_messages
    ids, mask = _ACT_MASK_GEN.get_loss_mask(msgs)
    rl = _ACT_MASK_GEN.get_response_lengths([mask])[0]
    if rl == 0:
        return None
    global _PACK_CHECKED
    if not _PACK_CHECKED:        # one-time self-check: loss=1 tokens must decode to the generated text
        _PACK_CHECKED = True
        got = "".join(_ACT_MASK_GEN.get_text_from_loss_mask(ids, mask)).replace("<|im_end|>", "")
        want = "".join(m["content"] for m in trial_messages if m["role"] == "assistant")
        ok = "".join(got.split()) == "".join(want.split())
        print(f"[alchemy pack] one-time loss-mask check: {'PASS' if ok else 'FAIL'} "
              f"(masked_tokens={int(sum(mask))}, assistant_turns={sum(1 for m in trial_messages if m['role']=='assistant')})",
              flush=True)
        if not ok:
            print(f"[alchemy pack]   GOT : {got[:160]!r}\n[alchemy pack]   WANT: {want[:160]!r}", flush=True)
    return Sample(
        group_index=seed.group_index, index=seed.index, prompt="(packed trial chat)",
        tokens=ids, response="(packed trial)", response_length=rl, loss_mask=mask[-rl:],
        rollout_log_probs=None, status=Sample.Status.COMPLETED,
        metadata={"phase": "act", "trial_pos": trial_pos})


def _copy_messages(messages: list) -> list:
    return [{"role": m.get("role"), "content": m.get("content", "")} for m in messages]


def _build_rewrite_messages(
    *,
    freeform: bool,
    summary_text: str | None,
    memory_history: list[dict],
    trial_history: list[dict],
    current_trial_idx: int,
    current_trial_messages: list,
    memory_window_size: int,
) -> tuple[list, str]:
    """Build the WRITE prompt.

    `memory_window_size=1` intentionally preserves the old prompt exactly: current trial transcript plus
    one previous summary in the final instruction. Larger windows expose the last K trial transcripts and
    last K memory snapshots so the writer can recover from lossy previous summaries.
    """
    base_instr = SUMMARY_INSTRUCTION_FREEFORM if freeform else SUMMARY_INSTRUCTION
    if memory_window_size <= 1:
        instr = base_instr
        if summary_text:
            instr = f"Your previous summary:\n{summary_text}\n\n{instr}"
        return ([{"role": "system", "content": SUMMARY_SYSTEM}] + current_trial_messages
                + [{"role": "user", "content": instr}]), instr

    window = max(1, int(memory_window_size))
    recent_memories = memory_history[-window:]
    recent_trials = (trial_history + [{
        "trial_idx": current_trial_idx,
        "messages": _copy_messages(current_trial_messages),
    }])[-window:]

    context_messages = [{"role": "system", "content": SUMMARY_SYSTEM}]
    for t in recent_trials:
        context_messages.append({
            "role": "user",
            "content": f"[Recent trial transcript: trial {t['trial_idx']}]\n"
                       f"Use this as evidence when updating the episode memory.",
        })
        context_messages.extend(_copy_messages(t["messages"]))

    mem_block = ""
    if recent_memories:
        parts = []
        for m in recent_memories:
            parts.append(f"[Memory after trial {m['trial_idx']}]\n{m['summary']}")
        mem_block = "Recent memory snapshots (oldest to newest):\n\n" + "\n\n".join(parts) + "\n\n"

    instr = (
        f"{mem_block}"
        f"Update the running memory using the recent memory snapshots and recent trial transcripts above. "
        f"Carry forward confirmed facts, recover any useful details that were lost, and incorporate the "
        f"newest trial evidence.\n\n"
        f"{base_instr}"
    )
    return context_messages + [{"role": "user", "content": instr}], instr


def _classify_action(desc: str) -> str:
    text = (desc or "").lower()
    if "potion" in text:
        return "potion"
    if "cauldron" in text:
        return "cauldron"
    if "end" in text and "trial" in text:
        return "end_trial"
    return "other"


def _episode_stats(traj: list, per_trial_scores: list, write_audit: list) -> dict:
    attempted = 0
    invalid = 0
    valid_total = 0
    action_counts = {"potion": 0, "cauldron": 0, "end_trial": 0, "other": 0}
    turns_per_trial: dict = {}                       # trial idx -> #turns (all turns, matches offline turn plot)
    for rec in traj:
        attempted += 1
        _tk = int(rec.get("trial", 0))
        turns_per_trial[_tk] = turns_per_trial.get(_tk, 0) + 1
        if not rec.get("valid"):
            invalid += 1
            continue
        valid_total += 1
        action_counts[_classify_action(rec.get("action"))] += 1

    fk_counts = [int(w.get("n_fk", 0) or 0) for w in write_audit]
    accs = [float(a) for w in write_audit for a in (w.get("accs") or []) if a is not None]
    kept = sum(1 for w in write_audit if w.get("kept"))

    return {
        "actions": {
            **action_counts,
            "valid_total": valid_total,
            "attempted": attempted,
            "invalid": invalid,
            "zero_potion": action_counts["potion"] == 0,
        },
        "per_trial_scores": [float(x) for x in per_trial_scores],
        "turns_per_trial": {str(k): int(v) for k, v in sorted(turns_per_trial.items())},
        "write": {
            "total": len(write_audit),
            "kept": kept,
            "fk_counts": fk_counts,
            "accs": accs,
        },
    }

# --- writing-memory reward: score a memory by GENERATING the next stone (REASONING/ANSWER) and
# checking exact match against the real transition. Copied (not imported) from eval/wm_*.py to avoid
# dragging llm_client/requests/oracle deps into ray workers. enable_thinking stays False (pipeline-consistent).
WM_V4_SYSTEM = (
    "You are predicting the outcome in the game Alchemy. A stone is described by its colour, then size, "
    "then shape, then the words 'with reward' followed by a signed integer — the same way stones appear "
    "in the notes and the prompt. Using ONLY the notes below about this episode's chemistry, work out "
    "what stone results when the given potion is applied to the given stone. Reply in EXACTLY this "
    "format:\nREASONING: <your brief reasoning>\nANSWER: <colour> <size> <shape> with reward <N>\n"
    "The ANSWER line must be the resulting stone alone in plain words, and must include 'with reward <N>'."
)
_STONE_RE = re.compile(r"(purple|blue)\s+(small|large)\s+(round|pointy)", re.I)
_RWD_RE = re.compile(r"reward\s*([+-]?\d+)", re.I)


def _parse_stone(text):
    """Parse the predicted stone from a REASONING/ANSWER generation -> ((colour,size,shape), reward|None)."""
    text = re.sub(r"</?[A-Za-z_]+>", " ", text or "")
    am = list(re.finditer(r"ANSWER\s*:", text, re.I))
    if am:
        text = text[am[-1].end():]
    ms = list(_STONE_RE.finditer(text))
    if not ms:
        return None
    m = ms[0]
    feats = (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
    rm = _RWD_RE.search(text[m.start():])
    return feats, (int(rm.group(1)) if rm else None)

DEFAULT_LEVEL = "alchemy/perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
_ORACLE_CACHE: dict[str, list[float]] | None = None


def _oracle_cache_path() -> str:
    env_path = os.environ.get("ALCHEMY_ORACLE_CACHE")
    if env_path:
        return env_path
    return os.path.join(os.path.dirname(__file__), "eval", "oracle_cache.json")


def _load_oracle_cache() -> dict[str, list[float]]:
    global _ORACLE_CACHE
    if _ORACLE_CACHE is None:
        with open(_oracle_cache_path()) as f:
            _ORACLE_CACHE = json.load(f)
    return _ORACLE_CACHE


def _oracle_norm_score(episode_index, trial_pos: int, raw_score: float) -> float | None:
    if episode_index is None:
        raise ValueError("downstream_norm_improve requires fixed episode_index for oracle normalization")
    oracle_scores = _load_oracle_cache().get(str(int(episode_index)))
    if not oracle_scores:
        raise ValueError(f"Missing oracle scores for episode_index={episode_index}")
    trial_pos = int(trial_pos)
    if trial_pos < 0 or trial_pos >= len(oracle_scores):
        raise ValueError(
            f"Oracle trial index out of range: episode_index={episode_index}, trial_pos={trial_pos}, "
            f"n_oracle_trials={len(oracle_scores)}"
        )
    denom = float(oracle_scores[trial_pos])
    if denom <= 0:
        return None
    return float(raw_score) / denom


def _safe_path_part(value) -> str:
    text = str(value if value is not None else "unknown")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def _write_trajectory(record: dict) -> None:
    """Dump a human-readable per-episode trajectory to ALCHEMY_TRAJ_DIR (if set). Never raises."""
    root = os.environ.get("ALCHEMY_TRAJ_DIR")
    if not root:
        return
    try:
        rollout_id = _safe_path_part(record.get("rollout_id"))
        episode_index = _safe_path_part(record.get("episode_index", record.get("alchemy_seed")))
        sample_index = _safe_path_part(record.get("index"))
        if record.get("evaluation"):
            dataset = _safe_path_part(record.get("eval_dataset_name"))
            traj_dir = os.path.join(root, "eval", dataset, f"rollout_{rollout_id}")
            filename = f"ep_{sample_index}_episode_{episode_index}.json"
        else:
            traj_dir = os.path.join(root, "train", f"rollout_{rollout_id}")
            filename = f"ep_{sample_index}_episode_{episode_index}.json"
        os.makedirs(traj_dir, exist_ok=True)
        path = os.path.join(traj_dir, filename)
        with open(path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _build_messages(system_text: str, user_text: str) -> list[dict]:
    return [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]


def _bool_flag(args, name: str, default: bool = True) -> bool:
    """Resolve an Alchemy bool knob from env first, then custom config/default args."""
    env_name = f"ALCHEMY_{name.upper()}"
    raw = os.environ.get(env_name)
    if raw is None:
        raw = getattr(args, name, default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _encode_prompt(state, messages):
    """Render chat text then tokenize -> flat list[int] (same approach as the ARC object path; avoids
    the BatchEncoding-from-VL-tokenizer HTTP 422 trap)."""
    tokenizer = state.tokenizer
    enable_thinking = bool(getattr(state.args, "arc_enable_thinking", False))
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
    return [int(t) for t in enc["input_ids"]]


async def _infer(url: str, input_ids: list[int], sampling_params: dict):
    payload = {"input_ids": input_ids, "sampling_params": sampling_params, "return_logprob": True}
    out = await post(url, payload)
    meta = out["meta_info"]
    if meta.get("output_token_logprobs"):
        resp_ids = [item[1] for item in meta["output_token_logprobs"]]
        resp_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        resp_ids, resp_logprobs = [], []
    finish = meta.get("finish_reason", {}).get("type", "stop")
    return out["text"], resp_ids, resp_logprobs, finish


_STATUS_BY_FINISH = {"length": Sample.Status.TRUNCATED, "abort": Sample.Status.ABORTED}


def _collect_transition(before_vec, after_vec, parsed):
    """A real observed potion-application transition (s, a, d) for F_k, or None. d = real env outcome."""
    if not parsed.get("valid") or parsed.get("kind") != "potion":
        return None
    si, pi = parsed.get("stone"), parsed.get("potion")
    db, da = decode_obs(before_vec), decode_obs(after_vec)
    s_desc, s_rwd = _stone_desc(db["stones"][si]["axes"], db["stones"][si]["value"])
    colour = _potion_desc(db["potions"][pi]["type"])
    d_desc, d_rwd = _stone_desc(da["stones"][si]["axes"], da["stones"][si]["value"])
    changed = (d_desc != s_desc) or (d_rwd != s_rwd)
    return (f"{s_desc} with reward {s_rwd:+d}", colour, d_desc, d_rwd, changed)


def _build_fk(all_transitions, k, cap):
    """F_k = dedup (s,a,d) observed at trials > k (single rollout's future); cap by truncation."""
    seen, out = set(), []
    for (t, s_str, colour, d_desc, d_rwd, changed) in all_transitions:
        if t <= k:
            continue
        key = (s_str, colour)
        if key in seen:
            continue
        seen.add(key)
        out.append((s_str, colour, d_desc, d_rwd, changed))
        if len(out) >= cap:
            break
    return out


def _build_last(parsed, new_obs, reward):
    """Eval-style echo for the NEXT turn's GAME STATE UPDATE (copied from eval_alchemy._build_last, which
    can't be imported here because that module pulls in llm_client)."""
    last = {"kind": parsed.get("kind"), "stone": parsed.get("stone"),
            "potion": parsed.get("potion"), "outcome_desc": None, "note": None}
    if not parsed.get("valid"):
        last["kind"] = None
        last["note"] = (f"Your last response was not a valid action "
                        f"({parsed.get('reason') or 'could not parse an ACTION line'}); "
                        f"it was skipped as a no-op. Reply with exactly one ACTION.")
        return last
    if parsed["kind"] == "potion":
        s = decode_obs(new_obs["vec"])["stones"][parsed["stone"]]
        if s["exists"]:
            desc, rwd = _stone_desc(s["axes"], s["value"])
            last["outcome_desc"] = f"{desc} with reward {rwd:+d}"
        else:
            last["outcome_desc"] = "(the stone is gone)"
    elif parsed["kind"] == "cauldron":
        last["note"] = (f"You placed stone {parsed['stone']} in the cauldron and "
                        f"scored {int(round(reward)):+d}.")
    elif parsed["kind"] == "end_trial":
        last["note"] = "You ended the trial."
    return last


async def _score_memory_accuracy(state, url, greedy_params, memory, fk):
    """Reward for one candidate memory: exact-match generation accuracy on F_k (reason-then-answer)."""
    notes = (memory or "").strip() or "(no notes yet)"

    async def _one(s, a, d_desc, d_rwd):
        user = f"Notes:\n{notes}\n\nYou place {s} into the {a} potion.\nResulting stone:"
        ids = _encode_prompt(state, _build_messages(WM_V4_SYSTEM, user))
        text, _ids, _lp, _f = await _infer(url, ids, greedy_params)
        p = _parse_stone(text)
        return 1 if (p and p[0] == tuple(d_desc.split()) and p[1] == d_rwd) else 0

    hits = await asyncio.gather(*[_one(s, a, dd, dr) for (s, a, dd, dr, _ch) in fk])
    return (sum(hits) / len(hits)) if hits else 0.0


def _pack_nomemory_sample(state, seed, system, messages, trial_end_msgidx, per_trial_scores, episode_id):
    """Pack the WHOLE episode chat into ONE Sample (full-history, no memory). loss_mask=1 only on the
    model's assistant tokens (think-injection zeroed). Also compute per-trial token spans in RESPONSE
    space ([start,end,trial_pos)) so the advantage hook can give each trial segment its own advantage.
    msgs_upto_k is a PREFIX of the full chat (Qwen3 template is append-only) -> response lengths are
    monotonic and spans tile [0, rl_total). Returns None if no assistant turn."""
    global _ACT_MASK_GEN
    if not any(m["role"] == "assistant" for m in messages):
        return None
    if _ACT_MASK_GEN is None:
        _ACT_MASK_GEN = _AlchemyMaskGen(state.tokenizer, tokenizer_type="qwen3")
    full = [{"role": "system", "content": system}] + messages
    ids, mask = _ACT_MASK_GEN.get_loss_mask(full)
    rl_total = _ACT_MASK_GEN.get_response_lengths([mask])[0]
    if rl_total == 0:
        return None
    spans = []
    prev = 0
    for k in sorted(trial_end_msgidx):
        upto = [{"role": "system", "content": system}] + messages[:trial_end_msgidx[k]]
        _ids_k, mask_k = _ACT_MASK_GEN.get_loss_mask(upto)
        rl_k = min(_ACT_MASK_GEN.get_response_lengths([mask_k])[0], rl_total)
        if rl_k > prev:
            spans.append([prev, rl_k, int(k)])
            prev = rl_k
    if spans and prev < rl_total:    # tail safety: leftover response tokens -> last trial segment
        spans[-1][1] = rl_total
    return Sample(
        group_index=seed.group_index, index=seed.index, prompt="(packed episode chat)",
        tokens=ids, response="(packed episode)", response_length=rl_total, loss_mask=mask[-rl_total:],
        rollout_log_probs=None, status=Sample.Status.COMPLETED,
        metadata={"no_memory": True, "episode_id": episode_id, "trial_spans": spans,
                  "per_trial_scores": [float(x) for x in per_trial_scores]})


async def _generate_nomemory(input: GenerateFnInput) -> GenerateFnOutput:
    """no-memory (full-history) branch: the whole episode is ONE continuous chat (no summary, no history
    reset); later trials see all earlier trials. Emits ONE Sample/episode with per-trial token spans;
    reward = per-trial score, normalized per (group, trial_pos) into per-token advantages by the hook.
    Mirrors the offline full-history eval (eval-q34b-v2-baseline: nosum-noprior)."""
    args, state, seed = input.args, input.state, input.sample
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = deepcopy(input.sampling_params)

    meta = seed.metadata or {}
    rollout_id = meta.get("_eval_rollout_id", getattr(args, "alchemy_current_rollout_id", None))
    eval_dataset_name = meta.get("_eval_dataset_name")
    level_name = meta.get("level_name")
    max_steps_per_trial = int(meta.get("max_steps_per_trial", 20))
    num_trials_cap = int(os.environ.get("ALCHEMY_NUM_TRIALS_CAP", "0")) or None
    system = EVAL_SYSTEM

    episode_index = meta.get("episode_index")
    if episode_index is None and meta.get("alchemy_seed") is None:
        episode_index = seed.group_index
    if episode_index is not None:
        env = AlchemyTextEnv(episode_index=int(episode_index), level_name=level_name,
                             max_steps_per_trial=max_steps_per_trial, end_trial_action=True)
        env_tag = {"episode_index": int(episode_index)}
    else:
        env = AlchemyTextEnv(level_name=level_name, num_trials=int(meta.get("num_trials", 10)),
                             max_steps_per_trial=max_steps_per_trial, seed=int(meta["alchemy_seed"]),
                             end_trial_action=True)
        env_tag = {"alchemy_seed": int(meta["alchemy_seed"])}
    max_turns = int(getattr(args, "max_turns", env.num_trials * max_steps_per_trial)
                    or env.num_trials * max_steps_per_trial)
    episode_id = f"{seed.group_index}:{seed.index}"

    messages: list = []          # full-history chat: NEVER reset (the whole point of no-memory)
    traj: list = []
    trial_end_msgidx: dict = {}  # trial k -> len(messages) at end of trial k
    last = None
    first = True
    cur_k = 0
    try:
        obs = await asyncio.to_thread(env.reset)
        for _turn in range(max_turns):
            k = int(obs.get("trial", 0))
            user_text = render_game_state(obs, last=last, first_turn=first)
            messages.append({"role": "user", "content": user_text})
            prompt_ids = _encode_prompt(state, [{"role": "system", "content": system}] + messages)
            act_text, resp_ids, resp_lp, act_finish = await _infer(url, prompt_ids, sampling_params)
            if not resp_ids:
                break
            messages.append({"role": "assistant", "content": _strip_special(act_text)})
            cur_k = k
            parsed = parse_eval_action(act_text, obs=obs)
            rec = {"turn": _turn, "trial": k, "step": obs.get("step"), "user": user_text,
                   "raw_act": act_text, "action": parsed["desc"], "action_int": parsed["action"],
                   "valid": parsed.get("valid"), "act_finish": act_finish}
            if act_finish == "abort":
                rec["note"] = "act generation aborted"
                traj.append(rec)
                break
            action_int = parsed["action"] if parsed.get("valid") else (env.num_special - 1)
            obs_next, step_reward, done, _info = await asyncio.to_thread(env.step, action_int)
            rec.update(reward=float(step_reward), new_trial=obs_next["new_trial"])
            traj.append(rec)
            last = _build_last(parsed, obs_next, step_reward)
            first = False
            if obs_next["new_trial"] and not done:
                trial_end_msgidx[k] = len(messages)
                if num_trials_cap is not None and (k + 1) >= num_trials_cap:
                    break
            obs = obs_next
            if done:
                break
    finally:
        env.close()
    trial_end_msgidx[cur_k] = len(messages)   # final trial's end (no boundary fired)

    per_trial_scores = list(env.per_trial_scores)
    sample = _pack_nomemory_sample(state, seed, system, messages, trial_end_msgidx,
                                   per_trial_scores, episode_id)
    samples = [sample] if sample is not None else []

    write_audit = []
    ep_stats = _episode_stats(traj, per_trial_scores, write_audit)
    for s in samples:
        s.reward = float(sum(per_trial_scores))   # episode total (logging/eval metrics; train uses per-token adv)
        s.metadata = {**meta, **(s.metadata or {}), "episode_id": episode_id,
                      "alchemy_episode_stats": ep_stats}
    _write_trajectory({
        "level_name": level_name, "episode_id": episode_id, **env_tag,
        "group_index": seed.group_index, "index": seed.index, "granularity": "nomemory",
        "rollout_id": rollout_id, "eval_dataset_name": eval_dataset_name,
        "evaluation": bool(input.evaluation), "summary_mode": "nomemory", "system": system,
        "per_trial_scores": per_trial_scores, "num_turns": len(traj),
        "trial_spans": (samples[0].metadata.get("trial_spans") if samples else None),
        "summaries": [], "write_audit": write_audit, "turns": traj,
    })
    return GenerateFnOutput(samples=samples)


async def _generate_trial(input: GenerateFnInput) -> GenerateFnOutput:
    """Per-trial branch, ALIGNED WITH eval (no-prior + summary, replace mode): a multi-turn chat where
    each ACT turn conditions on the carried summary + this trial's history; one summary REWRITE per trial
    boundary samples G' candidates. Two reward streams: ACT = trial score; WRITE = summary accuracy on F_k.
    Mirrors eval/eval_alchemy.run_episode so step-0 behaviour == the offline eval."""
    args, state, seed = input.args, input.state, input.sample
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = deepcopy(input.sampling_params)
    greedy_params = deepcopy(input.sampling_params)
    greedy_params["temperature"] = 0.0
    greedy_params["max_new_tokens"] = int(getattr(args, "wm_gen_max_tokens", 384))

    meta = seed.metadata or {}
    rollout_id = meta.get("_eval_rollout_id", getattr(args, "alchemy_current_rollout_id", None))
    eval_dataset_name = meta.get("_eval_dataset_name")
    level_name = meta.get("level_name")
    max_steps_per_trial = int(meta.get("max_steps_per_trial", 20))
    num_trials_cap = int(os.environ.get("ALCHEMY_NUM_TRIALS_CAP", "0")) or None  # ablation: truncate episode to N trials (fixed-episode num_trials is baked in env, so cap in the loop)
    gprime = int(getattr(args, "wm_gprime", 4))
    fk_cap = int(getattr(args, "wm_fk_cap", 12))
    min_fk = int(getattr(args, "wm_min_fk", 3))
    keep_k = int(getattr(args, "wm_rewrite_keep", 3))   # DEFER: train WRITE on only K random valid boundaries
    summary_mode = getattr(args, "wm_summary_mode", "replace")
    memory_window_size = max(1, int(os.environ.get("ALCHEMY_MEMORY_WINDOW_SIZE")
                                    or getattr(args, "wm_memory_window_size", 1)))
    freeform = os.environ.get("ALCHEMY_FREEFORM", "0") == "1"   # free-form memory: drop fixed two-section structure
    train_act = _bool_flag(args, "train_act", True)
    train_write = _bool_flag(args, "train_write", True)
    write_signal = os.environ.get("ALCHEMY_WRITE_SIGNAL") or getattr(args, "write_reward_signal", "transition_acc")
    # ACT exploration shaping: explore_reward_k = Judge(M_{k-1}, M_k) added at the ADVANTAGE level in
    # alchemy_advantage (act_adv += beta * explore_adv). beta=0 (default) -> no judge calls, no behavior change.
    act_explore_beta = float(os.environ.get("ALCHEMY_ACT_EXPLORE_BETA")
                             or getattr(args, "act_explore_beta", 0.0) or 0.0)
    explore_enabled = (not input.evaluation) and act_explore_beta > 0
    # downstream_norm_improve trial window: R(M_k) = mean(norm[k+1 .. k+K]) - mean(norm[k-K+1 .. k])
    # (windows clipped to the episode; oracle<=0 trials dropped from the mean). K=1 (default) is
    # byte-identical to the original single-trial difference norm[k+1] - norm[k]. Only affects
    # write_signal == "downstream_norm_improve"; whitening key (downstream_trial_pos=k+1) unchanged.
    write_improve_k = max(1, int(os.environ.get("ALCHEMY_WRITE_IMPROVE_K")
                                 or getattr(args, "write_improve_k", 1) or 1))
    # k=0(第一条记忆 M_0)的 WRITE 奖励模式,仅作用于 write_signal == "downstream_norm_improve":
    #   improve(默认)  R(M_0) = mean(norm[1..K]) - norm[0]      —— 现行为,逐字节不变
    #   downstream      R(M_0) = mean(norm[1..K])                —— 去掉 -norm0 项(不再为烂第一局付费)
    #   skip            k=0 不训 WRITE(不产生样本,audit 记 k0_skipped);其余 k 一律不变
    write_k0_mode = (os.environ.get("ALCHEMY_WRITE_K0_MODE")
                     or getattr(args, "write_k0_mode", "improve") or "improve").strip().lower()
    if write_k0_mode not in ("improve", "downstream", "skip"):
        raise ValueError(f"ALCHEMY_WRITE_K0_MODE must be improve|downstream|skip, got {write_k0_mode!r}")
    if not input.evaluation and not train_act and not train_write:
        raise ValueError("At least one Alchemy training stream must be enabled: train_act or train_write")
    system = EVAL_SYSTEM   # no-prior eval system prompt (MAIN_PROMPT + FORMAT_BLOCK)

    episode_index = meta.get("episode_index")
    if episode_index is None and meta.get("alchemy_seed") is None:
        episode_index = seed.group_index
    if episode_index is not None:
        env = AlchemyTextEnv(episode_index=int(episode_index), level_name=level_name,
                             max_steps_per_trial=max_steps_per_trial, end_trial_action=True)
        env_tag = {"episode_index": int(episode_index)}
    else:
        env = AlchemyTextEnv(level_name=level_name, num_trials=int(meta.get("num_trials", 10)),
                             max_steps_per_trial=max_steps_per_trial, seed=int(meta["alchemy_seed"]),
                             end_trial_action=True)
        env_tag = {"alchemy_seed": int(meta["alchemy_seed"])}
    max_turns = int(getattr(args, "max_turns", env.num_trials * max_steps_per_trial)
                    or env.num_trials * max_steps_per_trial)
    episode_id = f"{seed.group_index}:{seed.index}"

    act_by_trial: dict = defaultdict(list)         # trial_pos -> list[Sample]
    write_points: list = []                         # {rewrite_idx, samples, memories}
    all_transitions: list = []                      # (trial_at_step, s_str, colour, d_desc, d_rwd, changed)
    traj: list = []
    messages: list = []          # the model's chat (reset each trial in replace mode), mirrors eval
    trial_messages: list = []    # current trial's turns only -> input to the summary REWRITE
    summary_text = None          # carried memory/summary (None before trial 0's first rewrite)
    summaries: list = []         # carried summary after each boundary (mirrors eval traj)
    memory_history: list = []    # previous carry-forward summaries: {trial_idx, summary}
    trial_history: list = []     # completed trial transcripts: {trial_idx, messages}
    last = None
    first = True
    first_of_trial = True
    cur_k = 0                    # trial index of the turns currently accumulating in trial_messages
    try:
        obs = await asyncio.to_thread(env.reset)
        for _turn in range(max_turns):
            k = int(obs.get("trial", 0))
            user_text = render_game_state(obs, last=last, first_turn=first)
            if first_of_trial and summary_text:    # fold the carried summary into the trial's 1st turn
                user_text = (f"[Summary of what you've learned so far this episode]\n{summary_text}\n\n"
                             f"{user_text}")
            messages.append({"role": "user", "content": user_text})
            trial_messages.append({"role": "user", "content": user_text})

            prompt_ids = _encode_prompt(state, [{"role": "system", "content": system}] + messages)
            act_text, resp_ids, resp_lp, act_finish = await _infer(url, prompt_ids, sampling_params)
            if not resp_ids:
                break
            act_clean = _strip_special(act_text)   # history content must be clean (no <|im_end|>), like eval
            messages.append({"role": "assistant", "content": act_clean})
            trial_messages.append({"role": "assistant", "content": act_clean})
            cur_k = k    # trial these accumulating turns belong to (packed into ONE ACT sample below)

            parsed = parse_eval_action(act_text, obs=obs)
            rec = {"turn": _turn, "trial": k, "step": obs.get("step"), "summary_in": summary_text,
                   "user": user_text, "raw_act": act_text, "action": parsed["desc"],
                   "action_int": parsed["action"], "valid": parsed.get("valid"), "act_finish": act_finish}
            if act_finish == "abort":          # system interruption (e.g. mid-rollout) -> end the episode
                rec["note"] = "act generation aborted"
                traj.append(rec)
                break
            # length (truncated) or stop: parse as usual. No valid ACTION (incl. truncated-before-action)
            # -> charged as a no-op that wastes the step (env budget bounds the loop), exactly like eval.
            before_vec = obs["vec"]
            action_int = parsed["action"] if parsed.get("valid") else (env.num_special - 1)  # invalid -> no-op
            obs_next, step_reward, done, _info = await asyncio.to_thread(env.step, action_int)
            if parsed.get("valid"):
                tr = _collect_transition(before_vec, obs_next["vec"], parsed)
                if tr is not None:
                    all_transitions.append((k, *tr))
            rec.update(reward=float(step_reward), new_trial=obs_next["new_trial"])
            traj.append(rec)
            last = _build_last(parsed, obs_next, step_reward)
            first = False
            first_of_trial = False

            # trial boundary -> produce ONE carry-forward summary now (temp=1; needed to ACT the next
            # trial). The expensive WRITE reward (G'-1 extra candidates + F_k scoring) is DEFERRED to
            # post-episode and only done for K randomly-kept VALID boundaries -> big rollout speedup. The
            # carry-forward is reused as candidate 0 of a kept point's G' group (same prompt, same temp).
            if obs_next["new_trial"] and not done:
                # pack the just-finished trial k into ONE ACT training sample (loss on its assistant tokens)
                if input.evaluation or train_act:
                    act_s = _pack_act_sample(state, seed, system, trial_messages, k)
                    if act_s is not None:
                        act_by_trial[k].append(act_s)
                rw_msgs, instr = _build_rewrite_messages(
                    freeform=freeform,
                    summary_text=summary_text,
                    memory_history=memory_history,
                    trial_history=trial_history,
                    current_trial_idx=k,
                    current_trial_messages=trial_messages,
                    memory_window_size=memory_window_size,
                )
                rw_ids = _encode_prompt(state, rw_msgs)
                cf_text, cf_ids, cf_lp, cf_fin = await _infer(url, rw_ids, sampling_params)
                if cf_ids:
                    cf_sample = Sample(group_index=seed.group_index, index=seed.index, prompt=instr,
                                       tokens=list(rw_ids) + cf_ids, response=cf_text,
                                       response_length=len(cf_ids), loss_mask=[1] * len(cf_ids),
                                       rollout_log_probs=cf_lp,
                                       status=_STATUS_BY_FINISH.get(cf_fin, Sample.Status.COMPLETED),
                                       metadata={"phase": "write", "rewrite_idx": k,
                                                 "memory_window_size": memory_window_size})
                    cf_memory = _strip_special(cf_text) or (summary_text or "")
                    write_points.append({"rewrite_idx": k, "rw_ids": rw_ids, "instr": instr,
                                         "input_messages": rw_msgs, "carry_sample": cf_sample,
                                         "carry_memory": cf_memory,
                                         "memory_window_size": memory_window_size})
                    summary_text = cf_memory      # carry forward to the next trial's ACT
                    summaries.append(summary_text)
                    memory_history.append({"trial_idx": k, "summary": summary_text})
                trial_history.append({"trial_idx": k, "messages": _copy_messages(trial_messages)})
                trial_messages = []
                if summary_mode == "replace":              # memory is the sole cross-trial carrier
                    messages = []
                first_of_trial = True
                # ablation: truncate the episode after `num_trials_cap` trials. Trial k just finished and
                # was packed above; trial_messages is already reset, so no partial/double pack. (k is 0-based)
                if num_trials_cap is not None and (k + 1) >= num_trials_cap:
                    break

            obs = obs_next
            if done:
                break
    finally:
        env.close()

    # the final trial ends via done/max_turns (no boundary fires) -> pack its ACT sample now
    if trial_messages and (input.evaluation or train_act):
        act_s = _pack_act_sample(state, seed, system, trial_messages, cur_k)
        if act_s is not None:
            act_by_trial[cur_k].append(act_s)

    per_trial_scores = list(env.per_trial_scores)
    samples: list = []
    write_audit: list = []
    n_trials_seen = max(act_by_trial) if act_by_trial else 0

    # ACT exploration reward: judge M_{k-1}->M_k for each boundary (training only, beta>0). summaries[k]=M_k;
    # M_{k-1}=summaries[k-1] ("" for k=0). Fire all judge calls concurrently; None on any failure -> trial k
    # simply gets no explore_score (advantage hook falls back to task-only for that sample). Never blocks.
    explore_by_trial: dict = {}    # trial k -> judge dict {new_discoveries,...,explore_score,brief_reason}
    if explore_enabled and summaries:
        ks = list(range(len(summaries)))
        prevs = ["" if k == 0 else summaries[k - 1] for k in ks]
        results = await asyncio.gather(*[judge_explore(prevs[i], summaries[k]) for i, k in enumerate(ks)])
        explore_by_trial = {k: r for k, r in zip(ks, results) if r is not None}

    # ACT stream: reward = that trial's score (raw; oracle-norm cancels in the (episode,trial_pos) group)
    if input.evaluation or train_act:
        for k in range(n_trials_seen + 1):
            r_act = float(per_trial_scores[k]) if k < len(per_trial_scores) else 0.0
            for s in act_by_trial.get(k, []):
                s.reward = r_act
                s.metadata = {**meta, **s.metadata, "episode_id": episode_id}
                if k in explore_by_trial:
                    r = explore_by_trial[k]
                    s.metadata["explore_score"] = r["explore_score"]   # used by the advantage hook
                    s.metadata["explore_dims"] = {d: r[d] for d in    # monitored per-dimension in metrics
                                                  ("new_discoveries", "error_correction",
                                                   "verification_targets", "non_redundant_change")}
                samples.append(s)

    if input.evaluation:
        ep_stats = _episode_stats(traj, per_trial_scores, write_audit)
        for s in samples:
            s.metadata = {**(s.metadata or {}), "alchemy_episode_stats": ep_stats}
        _write_trajectory({
            "level_name": level_name, "episode_id": episode_id, **env_tag,
            "group_index": seed.group_index, "index": seed.index, "granularity": "trial",
            "rollout_id": rollout_id, "eval_dataset_name": eval_dataset_name,
            "evaluation": True, "summary_mode": summary_mode, "system": system,
            "memory_window_size": memory_window_size,
            "per_trial_scores": per_trial_scores, "num_turns": len(traj),
            "summaries": summaries, "write_audit": write_audit, "turns": traj,
        })
        return GenerateFnOutput(samples=samples)

    # WRITE stream (DEFER + K): filter boundaries to those with a usable F_k, randomly KEEP K of them,
    # and only there pay the WRITE cost (reuse the online carry-forward as candidate 0, sample G'-1 more
    # from the same prompt, score all G' on F_k). Non-kept boundaries: the carry-forward was already used
    # to ACT, but it is NOT a training sample -> no WRITE gradient there.
    boundary_fk = {wp["rewrite_idx"]: _build_fk(all_transitions, wp["rewrite_idx"], fk_cap)
                   for wp in write_points}
    if not train_write:
        for wp in write_points:
            k = wp["rewrite_idx"]
            fk = boundary_fk[k]
            write_audit.append({"rewrite_idx": k, "n_fk": len(fk), "kept": False,
                                "summary_input_messages": wp["input_messages"],
                                "carried_output": wp["carry_sample"].response,
                                "memory_window_size": wp.get("memory_window_size", memory_window_size),
                                "disabled": True})
        ep_stats = _episode_stats(traj, per_trial_scores, write_audit)
        for s in samples:
            s.metadata = {**(s.metadata or {}), "alchemy_episode_stats": ep_stats}

        _write_trajectory({
            "level_name": level_name, "episode_id": episode_id, **env_tag,
            "group_index": seed.group_index, "index": seed.index, "granularity": "trial",
            "rollout_id": rollout_id,
            "summary_mode": summary_mode, "system": system, "per_trial_scores": per_trial_scores,
            "memory_window_size": memory_window_size,
            "num_turns": len(traj), "summaries": summaries, "write_audit": write_audit,
            "train_act": train_act, "train_write": train_write, "act_explore": explore_by_trial, "turns": traj,
        })
        return GenerateFnOutput(samples=samples)

    # ---- downstream WRITE signals ----
    # carry-forward IS the only WRITE sample (no G' candidates). The reward is derived from the next trial
    # score and whitened across siblings by (group_index, k+1) in the advantage hook via downstream_trial_pos.
    # transition-acc is computed on the carry-forward ONLY as a monitor.
    if write_signal in ("downstream", "downstream_improve", "downstream_norm_improve"):
        for wp in write_points:
            k = wp["rewrite_idx"]
            fk = boundary_fk[k]
            cf = wp["carry_sample"]
            if k == 0 and write_k0_mode == "skip" and write_signal == "downstream_norm_improve":
                # k=0 不训 WRITE:M_0 照常写入/使用,只是不产生训练样本
                write_audit.append({"rewrite_idx": k, "n_fk": len(fk), "kept": False,
                                    "summary_input_messages": wp["input_messages"],
                                    "carried_output": cf.response,
                                    "memory_window_size": wp.get("memory_window_size", memory_window_size),
                                    "k0_skipped": True})
                continue
            r_next = per_trial_scores[k + 1] if (k + 1) < len(per_trial_scores) else None
            if r_next is None:                       # last trial's memory has no downstream trial -> not trained
                write_audit.append({"rewrite_idx": k, "n_fk": len(fk), "kept": False,
                                    "summary_input_messages": wp["input_messages"],
                                    "carried_output": cf.response,
                                    "memory_window_size": wp.get("memory_window_size", memory_window_size),
                                    "no_downstream": True})
                continue
            if write_signal == "downstream_improve":  # signal 4: r_{k+1} - r_k (per-rollout temporal baseline)
                r_prev = per_trial_scores[k] if k < len(per_trial_scores) else 0.0
                r_down = r_next - r_prev
            elif write_signal == "downstream_norm_improve":
                # window K (write_improve_k): mean norm of the K trials after M_k vs the K trials up to k.
                # K=1 reduces exactly to the original norm[k+1] - norm[k].
                nxt = [_oracle_norm_score(episode_index, t, per_trial_scores[t])
                       for t in range(k + 1, min(k + write_improve_k, len(per_trial_scores) - 1) + 1)]
                prv = [_oracle_norm_score(episode_index, t, per_trial_scores[t])
                       for t in range(max(0, k - write_improve_k + 1), k + 1)]
                nxt = [x for x in nxt if x is not None]
                prv = [x for x in prv if x is not None]
                if k == 0 and write_k0_mode == "downstream":
                    # k=0 特例:R(M_0) = mean(norm[1..K]),不减 norm0(低基数付费通道拆除)
                    if not nxt:
                        write_audit.append({
                            "rewrite_idx": k, "n_fk": len(fk), "kept": False,
                            "summary_input_messages": wp["input_messages"],
                            "carried_output": cf.response,
                            "memory_window_size": wp.get("memory_window_size", memory_window_size),
                            "oracle_norm_skipped": True,
                            "oracle_norm_next_valid": False,
                            "oracle_norm_prev_valid": bool(prv),
                        })
                        continue
                    r_down = sum(nxt) / len(nxt)
                else:
                    if not nxt or not prv:
                        write_audit.append({
                            "rewrite_idx": k, "n_fk": len(fk), "kept": False,
                            "summary_input_messages": wp["input_messages"],
                            "carried_output": cf.response,
                            "memory_window_size": wp.get("memory_window_size", memory_window_size),
                            "oracle_norm_skipped": True,
                            "oracle_norm_next_valid": bool(nxt),
                            "oracle_norm_prev_valid": bool(prv),
                        })
                        continue
                    r_down = sum(nxt) / len(nxt) - sum(prv) / len(prv)
            else:                                     # signal 3: r_{k+1} (absolute downstream score)
                r_down = r_next
            cf.reward = float(r_down)
            cf.metadata = {**meta, **cf.metadata, "episode_id": episode_id, "downstream_trial_pos": k + 1}
            samples.append(cf)
            mon_acc = (await _score_memory_accuracy(state, url, greedy_params, wp["carry_memory"], fk)
                       if len(fk) >= 1 else None)     # transition-acc as MONITOR only (carry-forward, not G')
            write_audit.append({
                "rewrite_idx": k, "n_fk": len(fk), "kept": True,
                "fk": [{"stone": s, "potion": a, "result": dd, "reward": dr} for (s, a, dd, dr, _ch) in fk],
                "summary_input_messages": wp["input_messages"],
                "candidates": [cf.response], "carried_idx": 0,
                "memory_window_size": wp.get("memory_window_size", memory_window_size),
                "accs": ([mon_acc] if mon_acc is not None else []),
                "downstream_reward": float(r_down),
                **({"write_improve_k": write_improve_k} if write_improve_k != 1 else {}),
                **({"write_k0_mode": write_k0_mode} if (k == 0 and write_k0_mode != "improve") else {}),
            })
        ep_stats = _episode_stats(traj, per_trial_scores, write_audit)
        for s in samples:
            s.metadata = {**(s.metadata or {}), "alchemy_episode_stats": ep_stats}
        _write_trajectory({
            "level_name": level_name, "episode_id": episode_id, **env_tag,
            "group_index": seed.group_index, "index": seed.index, "granularity": "trial",
            "rollout_id": rollout_id,
            "summary_mode": summary_mode, "system": system, "per_trial_scores": per_trial_scores,
            "memory_window_size": memory_window_size,
            "num_turns": len(traj), "summaries": summaries, "write_audit": write_audit,
            "train_act": train_act, "train_write": train_write, "write_signal": write_signal,
            "act_explore": explore_by_trial, "turns": traj,
        })
        return GenerateFnOutput(samples=samples)

    valid_ks = [k for k, fk in boundary_fk.items() if len(fk) >= min_fk]
    kept = set(random.sample(valid_ks, min(keep_k, len(valid_ks)))) if valid_ks else set()
    for wp in write_points:
        k = wp["rewrite_idx"]
        fk = boundary_fk[k]
        if k not in kept:
            # not trained on, but the carry-forward summary (input + output) IS the summary-generation
            # process at this boundary -> always save it so every boundary is fully inspectable.
            write_audit.append({"rewrite_idx": k, "n_fk": len(fk), "kept": False,
                                "summary_input_messages": wp["input_messages"],
                                "carried_output": wp["carry_sample"].response,
                                "memory_window_size": wp.get("memory_window_size", memory_window_size)})
            continue
        extra = await asyncio.gather(*[_infer(url, wp["rw_ids"], sampling_params)
                                       for _ in range(max(0, gprime - 1))])
        cand_samples = [wp["carry_sample"]]
        cand_memories = [wp["carry_memory"]]
        for (rw_text, rw_ids2, rw_lp, rw_fin) in extra:
            if not rw_ids2:
                continue
            s_rw = Sample(group_index=seed.group_index, index=seed.index, prompt=wp["instr"],
                          tokens=list(wp["rw_ids"]) + rw_ids2, response=rw_text,
                          response_length=len(rw_ids2), loss_mask=[1] * len(rw_ids2),
                          rollout_log_probs=rw_lp,
                          status=_STATUS_BY_FINISH.get(rw_fin, Sample.Status.COMPLETED),
                          metadata={"phase": "write", "rewrite_idx": k,
                                    "memory_window_size": wp.get("memory_window_size", memory_window_size)})
            cand_samples.append(s_rw)
            cand_memories.append(_strip_special(rw_text) or wp["carry_memory"])
        accs = []
        for s, m in zip(cand_samples, cand_memories):
            acc = await _score_memory_accuracy(state, url, greedy_params, m, fk)
            s.reward = float(acc)
            accs.append(acc)
            s.metadata = {**meta, **s.metadata, "episode_id": episode_id}
            samples.append(s)
        write_audit.append({
            "rewrite_idx": k, "n_fk": len(fk), "kept": True,
            "fk": [{"stone": s, "potion": a, "result": dd, "reward": dr} for (s, a, dd, dr, _ch) in fk],
            "summary_input_messages": wp["input_messages"],
            "candidates": [s.response for s in cand_samples],
            "memory_window_size": wp.get("memory_window_size", memory_window_size),
            "carried_idx": 0, "accs": accs,
        })

    ep_stats = _episode_stats(traj, per_trial_scores, write_audit)
    for s in samples:
        s.metadata = {**(s.metadata or {}), "alchemy_episode_stats": ep_stats}

    _write_trajectory({
        "level_name": level_name, "episode_id": episode_id, **env_tag,
        "group_index": seed.group_index, "index": seed.index, "granularity": "trial",
        "rollout_id": rollout_id,
        "summary_mode": summary_mode, "system": system, "per_trial_scores": per_trial_scores,
        "memory_window_size": memory_window_size,
        "num_turns": len(traj), "summaries": summaries, "write_audit": write_audit,
        "train_act": train_act, "train_write": train_write, "act_explore": explore_by_trial, "turns": traj,
    })
    return GenerateFnOutput(samples=samples)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Entry point (--custom-generate-function-path). Default = per-trial memory path (_generate_trial,
    eval-aligned summary-replace). ALCHEMY_NO_MEMORY=1 selects the no-memory full-history path
    (_generate_nomemory): the whole episode is ONE continuous sequence with per-trial token-level
    advantages (no summary/memory)."""
    if os.environ.get("ALCHEMY_NO_MEMORY", "0") == "1" or (input.sample.metadata or {}).get("no_memory"):
        return await _generate_nomemory(input)
    return await _generate_trial(input)
