"""Episode-wise Frontier-CS rollout for clean delayed memory credit.

One Miles generation input is one complete problem-group episode.  The model
weights stay frozen while all ``S`` memory rounds are sampled.  The returned
training unit contains every ACT sample and the ``S - 1`` WRITE samples whose
rewards become observable in the following round.  Miles updates the actor only
after a configurable batch of complete group episodes has returned.

The older ``frontiercs_rollout.generate`` entry point remains available for the
round-wise/update-between-rounds formulation.  This module is deliberately a
separate entry point so experiments cannot silently change semantics.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample
from qwen_eval.frontiercs_ttt.judge import FrontierAlgorithmJudge
from qwen_eval.frontiercs_ttt.prompts import build_write_prompt, clean_memory
from qwen_eval.frontiercs_ttt.trace import TraceStore
from qwen_eval.frontiercs_ttt.types import ModelReply

from .frontiercs_rollout import (
    _SAFE_ID,
    _atomic_json,
    _atomic_text,
    _bool_setting,
    _candidate,
    _encode_prompt,
    _env_or_arg,
    _group_root,
    _group_score,
    _infer,
    _int_setting,
    _load_json,
    _pack_sample,
    _problem,
    _run_root,
    _sampling_params,
    _update_best,
    _visible_response,
)


_EPISODE_LOCKS: dict[str, asyncio.Lock] = {}


def _plain_token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text or "", add_special_tokens=False, return_tensors=None
    )["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def _episode_trace_group_id(group_id: str, episode_index: int) -> str:
    value = f"{group_id}.episode-{episode_index:08d}"
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe Frontier-CS episode trace ID: {value!r}")
    return value


def _initial_episode_state(
    *,
    group_id: str,
    trace_group_id: str,
    episode_index: int,
    problem_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "training_unit": "complete_group_episode",
        "group_id": group_id,
        "trace_group_id": trace_group_id,
        "episode_index": episode_index,
        "problem_ids": problem_ids,
        "next_round": 0,
        "memory": "",
        "previous_group_score": None,
        "pending_write_sample": None,
        "best_candidates": {},
        "train_samples": [],
    }


def _ensure_episode_manifest(
    path: Path,
    *,
    group_id: str,
    trace_group_id: str,
    episode_index: int,
    problem_ids: list[str],
    group_size: int,
    candidates_per_problem: int,
    memory_rounds: int,
    act_code_context: str,
    semantics: dict[str, Any],
) -> None:
    expected = {
        "schema_version": 1,
        "training_unit": "complete_group_episode",
        "optimizer_updates_inside_episode": 0,
        "group_id": group_id,
        "trace_group_id": trace_group_id,
        "episode_index": episode_index,
        "problem_ids": problem_ids,
        "group_size": group_size,
        "candidates_per_problem": candidates_per_problem,
        "memory_rounds": memory_rounds,
        "act_code_context": act_code_context,
        "act_input_contains_previous_diagnostics": False,
        "write_input_contains_act_reasoning": False,
        "semantics": semantics,
    }
    existing = _load_json(path)
    if existing is not None and existing != expected:
        raise ValueError(
            f"Frontier-CS episode manifest changed at {path}; use a new run ID"
        )
    if existing is None:
        _atomic_json(path, expected)


def _episode_settings(input: GenerateFnInput) -> dict[str, Any]:
    args = input.args
    if input.evaluation:
        raise ValueError(
            "Frontier-CS TTT episode state is not mutated during Miles eval; "
            "use the standalone evaluator for checkpoint evaluation"
        )
    if int(getattr(args, "n_samples_per_prompt", 1)) != 1:
        raise ValueError(
            "set --n-samples-per-prompt 1; Frontier-CS K is generated inside one group episode"
        )

    metadata = dict(input.sample.metadata or {})
    group_id = str(metadata.get("group_id") or "")
    problem_ids = [str(value) for value in (metadata.get("problem_ids") or [])]
    if not group_id or not problem_ids:
        raise ValueError("prompt metadata must contain group_id and non-empty problem_ids")
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError(f"group {group_id!r} contains duplicate problem IDs")

    group_size = _int_setting(
        args,
        "FRONTIERCS_GROUP_SIZE",
        "frontiercs_group_size",
        3,
    )
    if group_size < 1:
        raise ValueError("frontiercs_group_size must be at least 1")
    if len(problem_ids) != group_size:
        raise ValueError(
            f"group {group_id!r} has {len(problem_ids)} problems, but configured G={group_size}"
        )

    candidates_per_problem = _int_setting(
        args,
        "FRONTIERCS_CANDIDATES_PER_PROBLEM",
        "frontiercs_candidates_per_problem",
        1,
    )
    memory_rounds = _int_setting(
        args, "FRONTIERCS_MEMORY_ROUNDS", "frontiercs_memory_rounds", 4
    )
    if candidates_per_problem < 1 or memory_rounds < 2:
        raise ValueError(
            "episode-wise Frontier-CS training requires K>=1 and S>=2"
        )

    act_code_context = str(
        _env_or_arg(
            args,
            "FRONTIERCS_ACT_CODE_CONTEXT",
            "frontiercs_act_code_context",
            "none",
        )
    ).strip().lower()
    if act_code_context not in {"none", "best"}:
        raise ValueError("frontiercs_act_code_context must be none|best")

    episode_index = int(input.sample.index or 0)
    trace_group_id = _episode_trace_group_id(group_id, episode_index)
    return {
        "group_id": group_id,
        "trace_group_id": trace_group_id,
        "episode_index": episode_index,
        "problem_ids": problem_ids,
        "group_size": group_size,
        "candidates_per_problem": candidates_per_problem,
        "memory_rounds": memory_rounds,
        "act_code_context": act_code_context,
    }


async def _generate_episode_locked(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    settings = _episode_settings(input)
    group_id = settings["group_id"]
    trace_group_id = settings["trace_group_id"]
    episode_index = settings["episode_index"]
    problem_ids = settings["problem_ids"]
    candidates_per_problem = settings["candidates_per_problem"]
    memory_rounds = settings["memory_rounds"]
    act_code_context = settings["act_code_context"]

    run_root = _run_root(args)
    episode_root = _group_root(run_root, trace_group_id)
    state_path = episode_root / "episode_state.json"
    commit_path = episode_root / "episode.json"
    committed = _load_json(commit_path)
    if committed is not None:
        return GenerateFnOutput(
            samples=[
                Sample.from_dict(value)
                for value in (committed.get("train_samples") or [])
            ]
        )

    act_max_new_tokens = _int_setting(
        args,
        "FRONTIERCS_ACT_MAX_NEW_TOKENS",
        "frontiercs_act_max_new_tokens",
        25600,
    )
    write_max_new_tokens = _int_setting(
        args,
        "FRONTIERCS_WRITE_MAX_NEW_TOKENS",
        "frontiercs_write_max_new_tokens",
        25600,
    )
    diagnostics_chars = _int_setting(
        args,
        "FRONTIERCS_DIAGNOSTICS_CHARS",
        "frontiercs_diagnostics_chars_per_candidate",
        12000,
    )
    thinking = _bool_setting(
        args, "FRONTIERCS_ENABLE_THINKING", "frontiercs_enable_thinking", True
    )
    reward_mode = str(
        _env_or_arg(
            args,
            "FRONTIERCS_WRITE_REWARD_MODE",
            "frontiercs_write_reward_mode",
            "delta",
        )
    ).strip().lower()
    if reward_mode not in {"delta", "downstream"}:
        raise ValueError("frontiercs_write_reward_mode must be delta|downstream")

    _ensure_episode_manifest(
        episode_root / "episode_manifest.json",
        group_id=group_id,
        trace_group_id=trace_group_id,
        episode_index=episode_index,
        problem_ids=problem_ids,
        group_size=settings["group_size"],
        candidates_per_problem=candidates_per_problem,
        memory_rounds=memory_rounds,
        act_code_context=act_code_context,
        semantics={
            "write_reward_mode": reward_mode,
            "act_max_new_tokens": act_max_new_tokens,
            "write_max_new_tokens": write_max_new_tokens,
            "diagnostics_chars_per_candidate": diagnostics_chars,
            "enable_thinking": thinking,
        },
    )

    state = _load_json(state_path) or _initial_episode_state(
        group_id=group_id,
        trace_group_id=trace_group_id,
        episode_index=episode_index,
        problem_ids=problem_ids,
    )
    if state.get("problem_ids") != problem_ids:
        raise ValueError(
            f"episode state problem IDs {state.get('problem_ids')} do not match prompt {problem_ids}"
        )

    trace = TraceStore(run_root)
    problems = [_problem(problem_id) for problem_id in problem_ids]
    judge = FrontierAlgorithmJudge(
        str(
            _env_or_arg(
                args,
                "FRONTIERCS_JUDGE_URL",
                "frontiercs_judge_url",
                "http://127.0.0.1:8081",
            )
        ),
        timeout_seconds=float(
            _env_or_arg(
                args,
                "FRONTIERCS_JUDGE_TIMEOUT_SECONDS",
                "frontiercs_judge_timeout_seconds",
                1800.0,
            )
        ),
        poll_interval_seconds=float(
            _env_or_arg(
                args,
                "FRONTIERCS_JUDGE_POLL_SECONDS",
                "frontiercs_judge_poll_seconds",
                1.0,
            )
        ),
        diagnostics_limit=diagnostics_chars,
    )
    seq_length = int(getattr(args, "seq_length", 32768))
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    train_samples = [
        Sample.from_dict(value) for value in (state.get("train_samples") or [])
    ]

    for round_index in range(int(state.get("next_round", 0)), memory_rounds):
        round_root = trace.round_root(trace_group_id, round_index)
        memory_in = str(state.get("memory") or "")
        _atomic_text(round_root / "memory_in.md", memory_in)
        best_candidates = dict(state.get("best_candidates") or {})
        # ``sample.index`` changes on every dataset visit, so repeated epochs use
        # different seeds while a retry of the same episode stays deterministic.
        seed_base = (
            int(getattr(args, "rollout_seed", 42))
            + episode_index * 10_000_000
            + round_index * 100_000
        )

        work = []
        for problem_position, problem in enumerate(problems):
            previous = dict(best_candidates.get(problem.problem_id) or {})
            previous_code = (
                str(previous.get("code") or "")
                if act_code_context == "best"
                else None
            )
            for candidate_index in range(candidates_per_problem):
                work.append(
                    _candidate(
                        input=input,
                        trace=trace,
                        judge=judge,
                        url=url,
                        group_id=trace_group_id,
                        round_index=round_index,
                        problem=problem,
                        candidate_index=candidate_index,
                        memory=memory_in,
                        previous_code=previous_code,
                        act_max_new_tokens=act_max_new_tokens,
                        seq_length=seq_length,
                        thinking=thinking,
                        seed_base=seed_base + problem_position * 1000,
                    )
                )

        candidate_pairs = list(await asyncio.gather(*work))
        records = [pair[0] for pair in candidate_pairs]
        act_samples = [pair[1] for pair in candidate_pairs]
        current_group_score = _group_score(records, problem_ids)
        for sample in act_samples:
            sample.metadata = {
                **(sample.metadata or {}),
                "training_unit": "complete_group_episode",
                "group_template_id": group_id,
                "episode_index": episode_index,
            }
        added_samples: list[Sample] = list(act_samples)

        pending = state.get("pending_write_sample")
        previous_group_score = state.get("previous_group_score")
        if pending is not None:
            if previous_group_score is None:
                raise ValueError("pending WRITE exists without previous group score")
            write_sample = Sample.from_dict(dict(pending))
            if reward_mode == "delta":
                write_reward = (
                    current_group_score - float(previous_group_score)
                ) / 100.0
            else:
                write_reward = current_group_score / 100.0
            write_sample.reward = write_reward
            write_sample.metadata = {
                **(write_sample.metadata or {}),
                "downstream_round": round_index,
                "previous_group_score_0_100": float(previous_group_score),
                "downstream_group_score_0_100": current_group_score,
                "write_reward_mode": reward_mode,
            }
            added_samples.append(write_sample)

        train_samples.extend(added_samples)
        _update_best(state, records)

        memory_out = memory_in
        next_pending: dict[str, Any] | None = None
        write_summary: dict[str, Any]
        if round_index + 1 < memory_rounds:
            write_prompt = build_write_prompt(
                previous_memory=memory_in,
                problems=problems,
                candidates=records,
            )
            max_prompt_chars = _int_setting(
                args,
                "FRONTIERCS_WRITER_MAX_PROMPT_CHARS",
                "frontiercs_writer_max_prompt_chars",
                120000,
            )
            if len(write_prompt) > max_prompt_chars:
                raise ValueError(
                    f"WRITE prompt has {len(write_prompt)} chars, above limit {max_prompt_chars}; "
                    "reduce G, K, code length, or diagnostic limits"
                )
            write_prompt_ids = _encode_prompt(input.state, write_prompt, thinking=thinking)
            write_params = _sampling_params(
                input.sampling_params,
                prompt_tokens=len(write_prompt_ids),
                max_new_tokens=write_max_new_tokens,
                seq_length=seq_length,
                sampling_seed=seed_base + 99999,
            )
            (
                write_text,
                write_ids,
                write_logprobs,
                write_finish,
                write_engine_metadata,
            ) = await _infer(url, write_prompt_ids, write_params)
            write_reasoning, write_visible = _visible_response(write_text)
            memory_out = clean_memory(write_visible)
            memory_changed = memory_out.strip() != memory_in.strip()
            memory_empty = not bool(memory_out.strip())
            memory_tokens = _plain_token_count(input.state.tokenizer, memory_out)
            packed_write = _pack_sample(
                seed=input.sample,
                prompt_label=(
                    f"Frontier-CS WRITE {trace_group_id}/r{round_index}"
                ),
                prompt_ids=write_prompt_ids,
                response_text=write_text,
                response_ids=write_ids,
                response_logprobs=write_logprobs,
                finish=write_finish,
                metadata={
                    "phase": "write",
                    "training_unit": "complete_group_episode",
                    "group_id": trace_group_id,
                    "group_template_id": group_id,
                    "episode_index": episode_index,
                    "produced_round": round_index,
                    "memory_round": round_index,
                    "memory_tokens": memory_tokens,
                    "memory_changed": memory_changed,
                    "memory_empty": memory_empty,
                },
                sample_index=(
                    int(input.sample.index or 0) * 100000
                    + round_index * 1000
                    + 999
                ),
                engine_metadata=write_engine_metadata,
            )
            next_pending = packed_write.to_dict()
            trace.save_write(
                trace_group_id,
                round_index,
                write_prompt,
                ModelReply(
                    text=write_text,
                    reasoning=write_reasoning,
                    completion_tokens=len(write_ids),
                    finish_reason=write_finish,
                ),
                memory_out,
            )
            write_summary = {
                "generated": True,
                "reward_available_after_round": round_index + 1,
                "response_tokens": len(write_ids),
                "finish_reason": write_finish,
            }
        else:
            _atomic_text(round_root / "memory_out.md", memory_out)
            write_summary = {
                "generated": False,
                "reason": "final ACT round has no downstream round for WRITE reward",
            }

        state = {
            **state,
            "next_round": round_index + 1,
            "memory": memory_out,
            "previous_group_score": current_group_score,
            "pending_write_sample": next_pending,
            "train_samples": [sample.to_dict() for sample in train_samples],
        }
        round_value = {
            "schema_version": 1,
            "training_unit": "complete_group_episode",
            "optimizer_updates_inside_episode": 0,
            "group_id": group_id,
            "trace_group_id": trace_group_id,
            "episode_index": episode_index,
            "round_index": round_index,
            "problem_ids": problem_ids,
            "group_score_0_100": current_group_score,
            "candidate_scores_0_100": [
                {
                    "problem_id": record.problem_id,
                    "candidate_index": record.candidate_index,
                    "score": float(record.feedback.score),
                }
                for record in records
            ],
            "write": write_summary,
            "memory_in": memory_in,
            "memory_out": memory_out,
            "train_samples_added": [sample.to_dict() for sample in added_samples],
            "state_after": state,
        }
        _atomic_json(round_root / "round.json", round_value)
        _atomic_json(state_path, state)

    if state.get("pending_write_sample") is not None:
        raise RuntimeError("complete episode ended with an uncredited WRITE sample")
    expected_samples = (
        memory_rounds * len(problem_ids) * candidates_per_problem
        + memory_rounds
        - 1
    )
    if len(train_samples) != expected_samples:
        raise RuntimeError(
            f"complete episode produced {len(train_samples)} samples; "
            f"expected {expected_samples}"
        )
    episode_value = {
        "schema_version": 1,
        "training_unit": "complete_group_episode",
        "optimizer_updates_inside_episode": 0,
        "group_id": group_id,
        "trace_group_id": trace_group_id,
        "episode_index": episode_index,
        "problem_ids": problem_ids,
        "candidates_per_problem": candidates_per_problem,
        "memory_rounds": memory_rounds,
        "act_sample_count": memory_rounds
        * len(problem_ids)
        * candidates_per_problem,
        "write_sample_count": memory_rounds - 1,
        "train_samples": [sample.to_dict() for sample in train_samples],
    }
    _atomic_json(commit_path, episode_value)
    return GenerateFnOutput(samples=train_samples)


async def generate_episode(input: GenerateFnInput) -> GenerateFnOutput:
    """Miles custom generation entry point for complete-group episodes."""
    settings = _episode_settings(input)
    run_key = str(_run_root(input.args)) + "::" + settings["trace_group_id"]
    lock = _EPISODE_LOCKS.setdefault(run_key, asyncio.Lock())
    async with lock:
        return await _generate_episode_locked(input)
