"""Object-only (TEXT, no VLM) variant of the ARC-AGI-3 two-phase rollout.

Same control flow as ``arc_rollout.py`` (ACT -> env.step -> REWRITE; one composed episode ->
list[Sample], episode-level GRPO later), but the state is a TEXT object list
(``grid_to_objects_text``) and the transition an object-level diff (``objects_diff_text``) instead of
an image + raw matrix. No processor / image_data / multimodal_train_inputs — the prompt is plain text.
Selected by the run script's ``ARC_OBS_MODE=objects`` switch
(``--custom-generate-function-path examples.arc_agi3.arc_rollout_objects.generate``).
The VLM rollout in ``arc_rollout.py`` is untouched.
"""
from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.types import Sample

from examples.arc_agi3.env_arc import ArcInteractionEnv, clean_memory, parse_action
from examples.arc_agi3.object_tracker import ObjectTracker, changes_to_text, tracked_objects_text
from examples.arc_agi3.prompts import (
    MEMORY_TEMPLATE,
    SYSTEM_ACT_OBJ,
    SYSTEM_REWRITE_OBJ,
    render_act_objects_text,
    render_rewrite_objects_text,
)


def _action_str(parsed: dict) -> str:
    if parsed["action"] == "ACTION6":
        return f"ACTION6 x={parsed['x']} y={parsed['y']}"
    return str(parsed["action"])


def _write_trajectory(record: dict) -> None:
    """Dump a human-readable per-episode trajectory to ARC_TRAJ_DIR (if set). Never raises."""
    traj_dir = os.environ.get("ARC_TRAJ_DIR")
    if not traj_dir:
        return
    try:
        os.makedirs(traj_dir, exist_ok=True)
        path = os.path.join(traj_dir, f"ep_{record.get('index')}.json")
        with open(path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _build_messages(system_text: str, user_text: str) -> list[dict]:
    """Text-only system + user messages (no image)."""
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def _encode_prompt(state, messages):
    """Encode a text-only chat prompt -> prompt_ids as a flat list[int] (matches the VLM path's
    processor output). apply_chat_template(tokenize=True) can return a BatchEncoding on a VL tokenizer,
    whose list() yields key strings -> sglang rejects input_ids with HTTP 422. So we render text first,
    then tokenize."""
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


async def _run_phase(state, url, sampling_params, seed, system_text, user_text, phase):
    """One text-only generation (system + user_text) -> a Sample. Returns (sample|None, text, finish).

    ``phase`` ("act"|"rewrite") is stamped on sample.metadata. Returns sample=None if nothing generated.
    """
    messages = _build_messages(system_text, user_text)
    prompt_ids = _encode_prompt(state, messages)
    text, resp_ids, resp_logprobs, finish = await _infer(url, prompt_ids, sampling_params)
    if not resp_ids:
        return None, text, finish
    s = Sample(
        group_index=seed.group_index,
        index=seed.index,
        prompt=user_text,
        tokens=list(prompt_ids) + resp_ids,
        response=text,
        response_length=len(resp_ids),
        loss_mask=[1] * len(resp_ids),
        rollout_log_probs=resp_logprobs,
        status=_STATUS_BY_FINISH.get(finish, Sample.Status.COMPLETED),
        metadata={"phase": phase},
    )
    return s, text, finish


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    state = input.state
    seed = input.sample
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sampling_params = deepcopy(input.sampling_params)

    max_turns = int(getattr(args, "max_turns", 16) or 16)
    meta = seed.metadata or {}
    game_id = meta.get("game_id")
    assert game_id, "ARC sample.metadata must contain 'game_id' (set via --metadata-key)."
    max_actions = int(meta.get("max_actions", max_turns))
    sys_act = meta.get("system_act") or SYSTEM_ACT_OBJ
    sys_rw = meta.get("system_rewrite") or SYSTEM_REWRITE_OBJ

    env = ArcInteractionEnv(game_id, max_actions=max_actions)
    samples: list[Sample] = []
    traj: list[dict] = []
    memory = MEMORY_TEMPLATE                 # M_{-1} carried into turn 0
    try:
        obs = await asyncio.to_thread(env.reset)
        tracker = ObjectTracker()                       # per-episode: stable ids + occlusion tracking
        tagged, bg, _ = tracker.update(obs["grid"])     # x_0
        for _turn in range(max_turns):
            rec = {
                "turn": _turn,
                "state_before": obs.get("state"),
                "levels_before": obs.get("levels"),
                "available": obs.get("available"),
                "memory_in": memory,
            }

            # ---- ① ACT: decide one action from M_{t-1} + current objects (stable ids) ----
            act_user = render_act_objects_text(memory, obs, tracked_objects_text(tagged, bg))
            s_act, act_out, act_finish = await _run_phase(
                state, url, sampling_params, seed, sys_act, act_user, "act")
            if s_act is None:
                break  # nothing generated; drop this degenerate turn
            samples.append(s_act)
            parsed = parse_action(act_out, available=obs.get("available"))
            rec.update(act_input=act_user, raw_act=act_out, action=parsed["action"],
                       act_finish=act_finish, act_response_len=s_act.response_length)

            if act_finish in ("length", "abort"):
                rec.update(valid=False, note=f"act generation {act_finish}", reward=0.0)
                traj.append(rec)
                break

            # ---- env step (legal action) or reject (illegal: no step, screen unchanged) ----
            if parsed["valid"]:
                prev_grid = obs["grid"]
                obs_next, step_reward, done, _info = await asyncio.to_thread(
                    env.step, parsed["action"], parsed["x"], parsed["y"])
                tagged_next, bg_next, changes = tracker.update(obs_next["grid"])   # x_{t+1}
                diff_text = changes_to_text(changes)
                changed = prev_grid != obs_next["grid"]
                last = {"action": _action_str(parsed), "diff": diff_text, "changed": changed,
                        "state": obs_next.get("state"), "levels": obs_next.get("levels")}
                rec.update(valid=True, reward=float(step_reward),
                           levels_after=obs_next.get("levels"), diff=diff_text, changed=changed)
            else:
                obs_next, done = obs, False   # no step; screen unchanged
                tagged_next, bg_next = tagged, bg   # no tracker update (grid unchanged)
                diff_text = "Your action was rejected (not a legal move); the screen did NOT change."
                last = {"action": f"INVALID ({parsed['reason']})", "diff": diff_text,
                        "changed": False, "state": obs.get("state"), "levels": obs.get("levels")}
                rec.update(valid=False, note=parsed["reason"], reward=0.0, diff=diff_text, changed=False)

            # ---- ② REWRITE: M_t = update(M_{t-1}, a_t, BEFORE objects, AFTER objects) ----
            rw_user = render_rewrite_objects_text(memory, last,
                                                  tracked_objects_text(tagged, bg),
                                                  tracked_objects_text(tagged_next, bg_next))
            s_rw, rw_out, rw_finish = await _run_phase(
                state, url, sampling_params, seed, sys_rw, rw_user, "rewrite")
            if s_rw is not None:
                samples.append(s_rw)
                memory = clean_memory(rw_out) or memory
                rec.update(rw_input=rw_user, raw_rewrite=rw_out, memory=memory,
                           rw_finish=rw_finish, rw_response_len=s_rw.response_length)
            else:
                rw_finish = "empty"
                rec.update(rw_finish=rw_finish, memory=memory)
            traj.append(rec)

            obs = obs_next
            tagged, bg = tagged_next, bg_next
            if done or rw_finish in ("length", "abort"):
                break
    finally:
        env.close()

    levels = float(env.total_levels)
    episode_id = f"{seed.group_index}:{seed.index}"
    for s in samples:
        s.reward = levels
        s.metadata = {**meta, "arc_levels": levels, "episode_id": episode_id,
                      "arc_turns": len(traj), "phase": (s.metadata or {}).get("phase")}

    _write_trajectory({
        "game_id": game_id,
        "episode_id": episode_id,
        "group_index": seed.group_index,
        "index": seed.index,
        "final_levels": levels,
        "num_turns": len(traj),
        "system_act": sys_act,
        "system_rewrite": sys_rw,
        "turns": traj,
    })
    return GenerateFnOutput(samples=samples)
