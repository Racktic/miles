"""Custom multi-turn rollout for ARC-AGI-3 as a memory-based agent (miles ``--custom-generate-function-path``).

Per the design in ``ARC-AGI-3-Agents/docs/world_model_memory_rl.md``: at each turn the policy sees
ONLY ``[system + previous memory M_{t-1} + last transition + current state x_t (image)]`` — never the
past-turn history — and emits, in one composed generation, the updated memory ``M_t`` followed by the
action ``a_t``. The environment steps; the loop repeats.

Because each turn's context is self-contained, **every turn is an ordinary single-turn (prompt, response)
RL sample** (``loss_mask = 1`` over the whole generation = memory + action). One ``generate`` call runs
one full episode and returns ``GenerateFnOutput(samples=[turn_0, turn_1, ...])``. The episode's sparse
reward (= final ``levels_completed``) is stamped on every turn-sample; episode-level GRPO grouping is
done later by ``arc_advantage.reward_post_process``.
"""
from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample

from examples.arc_agi3.env_arc import ArcInteractionEnv, clean_memory, parse_action
from examples.arc_agi3.grid_render import grid_diff_text, grid_matrix_text, render_grid
from examples.arc_agi3.prompts import (
    MEMORY_TEMPLATE,
    SYSTEM_ACT,
    SYSTEM_REWRITE,
    render_act_text,
    render_rewrite_text,
)


def _action_str(parsed: dict) -> str:
    if parsed["action"] == "ACTION6":
        return f"ACTION6 x={parsed['x']} y={parsed['y']}"
    return str(parsed["action"])


def _write_trajectory(record: dict) -> None:
    """Dump a human-readable per-episode trajectory to ARC_TRAJ_DIR (if set). Never raises.

    One JSON file per episode (``seed.index`` is globally unique across rollouts, so no write
    races between the ~32 concurrent episodes). Stores memory/action/reward/diff per turn — NOT
    the images — so it stays tiny and replayable for inspecting how memory evolves.
    """
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


def _build_messages(system_text: str, user_text: str, images) -> list[dict]:
    """system + user(one or more images, then text). ``images`` = a single PIL image or a list."""
    if not isinstance(images, (list, tuple)):
        images = [images]
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]


def _encode_prompt(state, messages):
    """Encode a fresh chat prompt (one image) -> (prompt_ids, image_data, multimodal_train_inputs)."""
    tokenizer, processor = state.tokenizer, state.processor
    # Qwen3.5 defaults to enable_thinking=True (long CoT). For this strict turn format we keep thinking
    # OFF by default and let reasoning go into <memory>; toggle via --arc-enable-thinking / config.
    enable_thinking = bool(getattr(state.args, "arc_enable_thinking", False))
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert processor is not None, (
        "ARC rollout needs a VLM processor for image observations; Qwen3.5-4B should provide one."
    )
    from qwen_vl_utils import process_vision_info

    images, videos = process_vision_info(messages)
    proc_kwargs = {"text": [text], "images": images, "return_tensors": "pt"}
    if videos:
        proc_kwargs["videos"] = videos
    proc_out = processor(**proc_kwargs)
    prompt_ids = proc_out["input_ids"][0].tolist()
    # Exclude input_ids/attention_mask AND mm_token_type_ids: the latter is a Qwen3.5-VL per-token
    # tensor of shape (1, prompt_len) that VARIES with prompt length across turns. miles concatenates
    # every multimodal_train_inputs key along dim 0, which fails on a seq-length-varying tensor.
    # The megatron VLM forward locates image tokens from input_ids, so this is safe to drop.
    _MM_EXCLUDE = ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids")
    mm_train = {k: v for k, v in proc_out.items() if k not in _MM_EXCLUDE} or None
    image_data = [encode_image_for_rollout_engine(im) for im in (images or [])]
    return prompt_ids, image_data, mm_train, images


async def _infer(url: str, input_ids: list[int], sampling_params: dict, image_data):
    payload = {"input_ids": input_ids, "sampling_params": sampling_params, "return_logprob": True}
    if image_data:
        payload["image_data"] = image_data
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


async def _run_phase(state, url, sampling_params, seed, system_text, user_text, images, phase):
    """One generation (system + images + user_text) -> a Sample. Returns (sample|None, text, finish).

    ``phase`` ("act"|"rewrite") is stamped on sample.metadata so the trainer / L_WM can tell the two
    kinds of sample apart. Returns ``sample=None`` if nothing was generated (degenerate turn).
    """
    messages = _build_messages(system_text, user_text, images)
    prompt_ids, image_data, mm_train, imgs = _encode_prompt(state, messages)
    text, resp_ids, resp_logprobs, finish = await _infer(url, prompt_ids, sampling_params, image_data)
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
        multimodal_inputs={"images": imgs},
        multimodal_train_inputs=mm_train,
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
    # Rules live in code; a dataset row may override either phase's system prompt via metadata.
    sys_act = meta.get("system_act") or SYSTEM_ACT
    sys_rw = meta.get("system_rewrite") or SYSTEM_REWRITE

    env = ArcInteractionEnv(game_id, max_actions=max_actions)
    samples: list[Sample] = []
    traj: list[dict] = []
    memory = MEMORY_TEMPLATE                 # M_{-1} carried into turn 0
    try:
        obs = await asyncio.to_thread(env.reset)
        for _turn in range(max_turns):
            img = render_grid(obs["grid"])          # current frame x_t (also REWRITE's "before")
            rec = {
                "turn": _turn,
                "state_before": obs.get("state"),
                "levels_before": obs.get("levels"),
                "available": obs.get("available"),
                "memory_in": memory,
            }

            # ---- ① ACT: decide one action from M_{t-1} + current screen ----
            s_act, act_out, act_finish = await _run_phase(
                state, url, sampling_params, seed, sys_act,
                render_act_text(memory, obs, grid_matrix_text(obs["grid"])), img, "act")
            if s_act is None:
                break  # nothing generated; drop this degenerate turn
            samples.append(s_act)
            parsed = parse_action(act_out, available=obs.get("available"))
            rec.update(raw_act=act_out, act_finish=act_finish,
                       act_response_len=s_act.response_length, action=parsed["action"])

            if act_finish in ("length", "abort"):
                rec.update(valid=False, note=f"act generation {act_finish}", reward=0.0)
                traj.append(rec)
                break

            # ---- env step (legal action) or reject (illegal: no step, screen unchanged) ----
            if parsed["valid"]:
                prev_grid = obs["grid"]
                obs_next, step_reward, done, _info = await asyncio.to_thread(
                    env.step, parsed["action"], parsed["x"], parsed["y"])
                diff_text = grid_diff_text(prev_grid, obs_next["grid"])
                changed = prev_grid != obs_next["grid"]
                after_img = render_grid(obs_next["grid"])
                last = {"action": _action_str(parsed), "diff": diff_text, "changed": changed,
                        "state": obs_next.get("state"), "levels": obs_next.get("levels")}
                rec.update(valid=True, reward=float(step_reward),
                           levels_after=obs_next.get("levels"), diff=diff_text, changed=changed)
            else:
                obs_next, done, after_img = obs, False, img   # no step; screen unchanged
                diff_text = "Your action was rejected (not a legal move); the screen did NOT change."
                last = {"action": f"INVALID ({parsed['reason']})", "diff": diff_text,
                        "changed": False, "state": obs.get("state"), "levels": obs.get("levels")}
                rec.update(valid=False, note=parsed["reason"], reward=0.0, diff=diff_text, changed=False)

            # ---- ② REWRITE: M_t = update(M_{t-1}, a_t, before x_t, after x_{t+1}, diff d_t) ----
            s_rw, rw_out, rw_finish = await _run_phase(
                state, url, sampling_params, seed, sys_rw,
                render_rewrite_text(memory, last,
                                    grid_matrix_text(obs["grid"]),
                                    grid_matrix_text(obs_next["grid"])),
                [img, after_img], "rewrite")
            if s_rw is not None:
                samples.append(s_rw)
                memory = clean_memory(rw_out) or memory
                rec.update(raw_rewrite=rw_out, rw_finish=rw_finish,
                           rw_response_len=s_rw.response_length, memory=memory)
            else:
                rw_finish = "empty"
                rec.update(rw_finish=rw_finish, memory=memory)
            traj.append(rec)

            obs = obs_next
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
        "turns": traj,
    })
    return GenerateFnOutput(samples=samples)
