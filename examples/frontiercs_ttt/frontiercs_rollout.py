"""Miles rollout implementing synchronous group-shared Frontier-CS memory.

One Miles rollout is one memory round.  For each fixed problem group, the
rollout samples K independent one-shot ACT submissions per problem, evaluates
all G*K submissions, then performs exactly one WRITE after the barrier.  The
WRITE sample is held until the next round, where its reward is computed from
the downstream group score.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any


def _discover_frontiercs_root() -> Path:
    configured = os.environ.get("FRONTIERCS_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    # Portable convenience for the common layout where the two repositories
    # are cloned next to each other. Explicit FRONTIERCS_ROOT always wins.
    miles_root = Path(__file__).resolve().parents[2]
    candidates.append(miles_root.parent / "Frontier-CS")
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "algorithmic" / "problems").is_dir():
            return resolved
    checked = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        "cannot locate the Frontier-CS checkout containing algorithmic/problems; "
        f"set FRONTIERCS_ROOT explicitly (checked: {checked})"
    )


_FRONTIERCS_ROOT = _discover_frontiercs_root()
if str(_FRONTIERCS_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRONTIERCS_ROOT))

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.types import Sample
from qwen_eval.frontiercs_ttt.judge import FrontierAlgorithmJudge
from qwen_eval.frontiercs_ttt.prompts import (
    build_act_prompt,
    build_write_prompt,
    clean_memory,
    extract_cpp,
)
from qwen_eval.frontiercs_ttt.trace import TraceStore
from qwen_eval.frontiercs_ttt.types import (
    CandidateRecord,
    JudgeFeedback,
    ModelReply,
    ProblemSpec,
)


_GROUP_LOCKS: dict[str, asyncio.Lock] = {}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _env_or_arg(args: Any, env: str, arg: str, default: Any) -> Any:
    value = os.environ.get(env)
    if value not in (None, ""):
        return value
    return getattr(args, arg, default)


def _int_setting(args: Any, env: str, arg: str, default: int) -> int:
    return int(_env_or_arg(args, env, arg, default))


def _bool_setting(args: Any, env: str, arg: str, default: bool) -> bool:
    value = _env_or_arg(args, env, arg, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _run_root(args: Any) -> Path:
    default_output_root = (
        _FRONTIERCS_ROOT / "qwen_eval" / "results" / "frontiercs_ttt_rl"
    )
    configured_output_root = _env_or_arg(
        args,
        "FRONTIERCS_OUTPUT_ROOT",
        "frontiercs_output_root",
        "",
    )
    output_root = Path(
        str(configured_output_root).strip() or default_output_root
    ).expanduser().resolve()
    run_id = str(
        _env_or_arg(args, "FRONTIERCS_RUN_ID", "frontiercs_run_id", "development")
    )
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError(f"unsafe FRONTIERCS_RUN_ID: {run_id!r}")
    root = output_root / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _group_root(run_root: Path, group_id: str) -> Path:
    if not _SAFE_ID.fullmatch(group_id):
        raise ValueError(f"unsafe Frontier-CS group_id: {group_id!r}")
    return run_root / "groups" / group_id


def _round_root(run_root: Path, group_id: str, round_index: int) -> Path:
    return _group_root(run_root, group_id) / f"round_{round_index:03d}"


def _sample_status(finish: str) -> Sample.Status:
    return {
        "length": Sample.Status.TRUNCATED,
        "abort": Sample.Status.ABORTED,
        "stop": Sample.Status.COMPLETED,
    }.get(finish, Sample.Status.COMPLETED)


def _visible_response(text: str, *, thinking: bool) -> tuple[str, str]:
    """Split Qwen reasoning from its visible response.

    Qwen's chat template places the opening ``<think>`` token in the prompt,
    while SGLang returns generated tokens only. Therefore an unfinished
    thinking response normally contains neither tag. When thinking is enabled,
    the closing tag is the only reliable reasoning-to-answer transition.
    """
    value = text or ""
    if "</think>" in value:
        reasoning, visible = value.split("</think>", 1)
        return reasoning.replace("<think>", "", 1).strip(), visible.strip()
    if thinking:
        # No closing tag means the model never completed the transition from
        # reasoning to its visible answer. The opening tag may be absent
        # because it was already part of the prompt.
        reasoning = value.split("<think>", 1)[1] if "<think>" in value else value
        return reasoning.strip(), ""
    if "<think>" in value:
        # Preserve safe behavior if a model unexpectedly starts an explicit
        # thought while thinking mode is disabled.
        return value.split("<think>", 1)[1].strip(), ""
    return "", value.strip()


def _encode_prompt(state: Any, user_content: str, *, thinking: bool) -> list[int]:
    messages = [{"role": "user", "content": user_content}]
    tokenizer = state.tokenizer
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    return [int(token) for token in encoded["input_ids"]]


async def _infer(
    url: str,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
) -> tuple[str, list[int], list[float], str, dict[str, Any]]:
    output = await post(
        url,
        {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        },
    )
    metadata = dict(output.get("meta_info") or {})
    token_logprobs = metadata.get("output_token_logprobs") or []
    response_ids = [int(item[1]) for item in token_logprobs]
    response_logprobs = [float(item[0]) for item in token_logprobs]
    if len(response_ids) != len(response_logprobs):
        raise RuntimeError("SGLang response token/logprob lengths disagree")
    if not response_ids:
        raise RuntimeError("SGLang returned no response token logprobs")
    finish = str((metadata.get("finish_reason") or {}).get("type") or "stop")
    text = str(output.get("text") or "").encode("utf-8", errors="replace").decode("utf-8")
    return text, response_ids, response_logprobs, finish, metadata


def _sampling_params(
    base: dict[str, Any],
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    seq_length: int,
    sampling_seed: int,
) -> dict[str, Any]:
    available = seq_length - prompt_tokens
    if available < 1:
        raise ValueError(
            f"prompt has {prompt_tokens} tokens and does not fit seq_length={seq_length}"
        )
    params = deepcopy(base)
    params["max_new_tokens"] = min(int(max_new_tokens), available)
    params["sampling_seed"] = int(sampling_seed)
    return params


def _pack_sample(
    *,
    seed: Sample,
    prompt_label: str,
    prompt_ids: list[int],
    response_text: str,
    response_ids: list[int],
    response_logprobs: list[float],
    finish: str,
    metadata: dict[str, Any],
    sample_index: int,
    engine_metadata: dict[str, Any],
) -> Sample:
    weight_versions: list[str] = []
    if engine_metadata.get("weight_version") is not None:
        weight_versions.append(str(engine_metadata["weight_version"]))
    sample = Sample(
        group_index=seed.group_index,
        index=sample_index,
        prompt=prompt_label,
        tokens=list(prompt_ids) + list(response_ids),
        response=response_text,
        response_length=len(response_ids),
        loss_mask=[1] * len(response_ids),
        rollout_log_probs=list(response_logprobs),
        weight_versions=weight_versions,
        status=_sample_status(finish),
        metadata=metadata,
    )
    sample.validate()
    return sample


def _problem(problem_id: str) -> ProblemSpec:
    path = _FRONTIERCS_ROOT / "algorithmic" / "problems" / problem_id / "statement.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing Frontier-CS statement: {path}")
    statement = path.read_text(encoding="utf-8")
    title = next((line.strip().lstrip("# ") for line in statement.splitlines() if line.strip()), "")
    return ProblemSpec(problem_id=problem_id, statement=statement, title=title)


def _initial_state(group_id: str, problem_ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "group_id": group_id,
        "problem_ids": problem_ids,
        "next_round": 0,
        "memory": "",
        "previous_group_score": None,
        "pending_write_sample": None,
        "best_candidates": {},
    }


def _ensure_group_manifest(
    path: Path,
    *,
    problem_ids: list[str],
    group_size: int,
    candidates_per_problem: int,
    memory_rounds: int,
    act_code_context: str,
    semantics: dict[str, Any],
) -> None:
    expected = {
        "schema_version": 1,
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
            f"Frontier-CS group manifest changed at {path}; use a new run ID"
        )
    if existing is None:
        _atomic_json(path, expected)


def _update_best(state: dict[str, Any], records: list[CandidateRecord]) -> None:
    best = dict(state.get("best_candidates") or {})
    legacy_zero_best = os.environ.get("FRONTIERCS_LEGACY_ZERO_BEST") == "1"
    for record in records:
        score = float(record.feedback.score)
        if score <= 0.0 and not legacy_zero_best:
            continue
        current = dict(best.get(record.problem_id) or {})
        if not current or score > float(current.get("score") or 0.0):
            best[record.problem_id] = {
                "score": score,
                "round_index": int(record.round_index),
                "candidate_index": int(record.candidate_index),
                "code": record.code,
            }
    state["best_candidates"] = best


def _group_score(records: list[CandidateRecord], problem_ids: list[str]) -> float:
    per_problem: list[float] = []
    for problem_id in problem_ids:
        values = [
            float(record.feedback.score)
            for record in records
            if record.problem_id == problem_id
        ]
        if not values:
            raise ValueError(f"no candidate score for Frontier-CS problem {problem_id}")
        per_problem.append(sum(values) / len(values))
    return sum(per_problem) / len(per_problem)


def _candidate_sample_path(
    trace: TraceStore,
    group_id: str,
    round_index: int,
    problem_id: str,
    candidate_index: int,
) -> Path:
    return trace.candidate_root(
        group_id, round_index, problem_id, candidate_index
    ) / "train_sample.json"


async def _candidate(
    *,
    input: GenerateFnInput,
    trace: TraceStore,
    judge: FrontierAlgorithmJudge,
    url: str,
    group_id: str,
    round_index: int,
    problem: ProblemSpec,
    candidate_index: int,
    memory: str,
    previous_code: str | None,
    act_max_new_tokens: int,
    seq_length: int,
    thinking: bool,
    seed_base: int,
) -> tuple[CandidateRecord, Sample]:
    record = trace.load_candidate(
        group_id, round_index, problem.problem_id, candidate_index
    )
    sample_path = _candidate_sample_path(
        trace, group_id, round_index, problem.problem_id, candidate_index
    )
    if record is not None and sample_path.is_file():
        return record, Sample.from_dict(json.loads(sample_path.read_text(encoding="utf-8")))

    prompt = build_act_prompt(problem, memory, previous_code=previous_code)
    prompt_ids = _encode_prompt(input.state, prompt, thinking=thinking)
    params = _sampling_params(
        input.sampling_params,
        prompt_tokens=len(prompt_ids),
        max_new_tokens=act_max_new_tokens,
        seq_length=seq_length,
        sampling_seed=seed_base + candidate_index,
    )
    response, response_ids, logprobs, finish, engine_metadata = await _infer(
        url, prompt_ids, params
    )
    reasoning, visible = _visible_response(response, thinking=thinking)
    code = extract_cpp(visible)
    if not code.strip():
        feedback = JudgeFeedback(
            status="invalid_submission",
            score=0.0,
            error="ACT response did not contain non-empty visible C++ code",
        )
    else:
        feedback = await judge.evaluate(problem.problem_id, code)
    record = CandidateRecord(
        problem_id=problem.problem_id,
        round_index=round_index,
        candidate_index=candidate_index,
        act_prompt=prompt,
        response=response,
        reasoning=reasoning,
        code=code,
        completion_tokens=len(response_ids),
        finish_reason=finish,
        feedback=feedback,
    )
    feedback_status = str(feedback.status or "").strip().lower()
    feedback_error = str(feedback.error or "").lower()
    metadata = {
        "phase": "act",
        "group_id": group_id,
        "memory_round": round_index,
        "problem_id": problem.problem_id,
        "candidate_index": candidate_index,
        "score_0_100": float(feedback.score),
        "evaluation_status": feedback_status,
        "executed": feedback_status in {"done", "completed"},
        "compile_error": "compile failed" in feedback_error,
        "invalid_submission": feedback_status == "invalid_submission",
        "has_diagnostics": bool(str(feedback.diagnostics or "").strip()),
    }
    packed = _pack_sample(
        seed=input.sample,
        prompt_label=f"Frontier-CS ACT {group_id}/{problem.problem_id}/r{round_index}/k{candidate_index}",
        prompt_ids=prompt_ids,
        response_text=response,
        response_ids=response_ids,
        response_logprobs=logprobs,
        finish=finish,
        metadata=metadata,
        sample_index=int(input.sample.index or 0) * 100_000
        + round_index * 1_000
        + int(problem.problem_id) * 10
        + candidate_index,
        engine_metadata=engine_metadata,
    )
    packed.reward = float(feedback.score) / 100.0
    trace.save_candidate(group_id, record)
    _atomic_json(sample_path, packed.to_dict())
    return record, packed


async def _generate_locked(input: GenerateFnInput) -> GenerateFnOutput:
    if input.evaluation:
        raise ValueError(
            "Frontier-CS TTT training state is not mutated during Miles eval; "
            "use the standalone evaluator for checkpoint evaluation"
        )
    args = input.args
    if int(getattr(args, "n_samples_per_prompt", 1)) != 1:
        raise ValueError(
            "set --n-samples-per-prompt 1; Frontier-CS K is generated inside one group rollout"
        )

    metadata = dict(input.sample.metadata or {})
    group_id = str(metadata.get("group_id") or "")
    problem_ids = [str(value) for value in (metadata.get("problem_ids") or [])]
    if not group_id or not problem_ids:
        raise ValueError("prompt metadata must contain group_id and non-empty problem_ids")
    if len(set(problem_ids)) != len(problem_ids):
        raise ValueError(f"group {group_id!r} contains duplicate problem IDs")

    group_size = _int_setting(
        args, "FRONTIERCS_GROUP_SIZE", "frontiercs_group_size", 3
    )
    if group_size < 1:
        raise ValueError("frontiercs_group_size must be at least 1")
    if len(problem_ids) != group_size:
        raise ValueError(
            f"group {group_id!r} has {len(problem_ids)} problems, but configured G={group_size}"
        )
    candidates_per_problem = _int_setting(
        args, "FRONTIERCS_CANDIDATES_PER_PROBLEM", "frontiercs_candidates_per_problem", 1
    )
    memory_rounds = _int_setting(
        args, "FRONTIERCS_MEMORY_ROUNDS", "frontiercs_memory_rounds", 2
    )
    if candidates_per_problem < 1 or memory_rounds < 1:
        raise ValueError("Frontier-CS K and S must both be >= 1")
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

    run_root = _run_root(args)
    group_root = _group_root(run_root, group_id)
    state_path = group_root / "state.json"
    _ensure_group_manifest(
        group_root / "group_manifest.json",
        problem_ids=problem_ids,
        group_size=group_size,
        candidates_per_problem=candidates_per_problem,
        memory_rounds=memory_rounds,
        act_code_context=act_code_context,
        semantics={
            "act_advantage_mode": str(
                _env_or_arg(
                    args,
                    "FRONTIERCS_ACT_ADVANTAGE_MODE",
                    "frontiercs_act_advantage_mode",
                    "raw",
                )
            ),
            "write_reward_mode": str(
                _env_or_arg(
                    args,
                    "FRONTIERCS_WRITE_REWARD_MODE",
                    "frontiercs_write_reward_mode",
                    "delta",
                )
            ),
            "write_advantage_mode": str(
                _env_or_arg(
                    args,
                    "FRONTIERCS_WRITE_ADVANTAGE_MODE",
                    "frontiercs_write_advantage_mode",
                    "direct",
                )
            ),
            "act_max_new_tokens": _int_setting(
                args,
                "FRONTIERCS_ACT_MAX_NEW_TOKENS",
                "frontiercs_act_max_new_tokens",
                25600,
            ),
            "write_max_new_tokens": _int_setting(
                args,
                "FRONTIERCS_WRITE_MAX_NEW_TOKENS",
                "frontiercs_write_max_new_tokens",
                25600,
            ),
            "diagnostics_chars_per_candidate": _int_setting(
                args,
                "FRONTIERCS_DIAGNOSTICS_CHARS",
                "frontiercs_diagnostics_chars_per_candidate",
                12_000,
            ),
            "enable_thinking": _bool_setting(
                args,
                "FRONTIERCS_ENABLE_THINKING",
                "frontiercs_enable_thinking",
                True,
            ),
        },
    )
    trace = TraceStore(run_root)
    state = _load_json(state_path) or _initial_state(group_id, problem_ids)
    if state.get("problem_ids") != problem_ids:
        raise ValueError(
            f"state problem IDs {state.get('problem_ids')} do not match prompt {problem_ids}"
        )

    rollout_id_value = getattr(args, "alchemy_current_rollout_id", None)
    round_index = int(state.get("next_round", 0) if rollout_id_value is None else rollout_id_value)
    round_root = _round_root(run_root, group_id, round_index)
    committed = _load_json(round_root / "round.json")
    if committed is not None:
        state_after = dict(committed.get("state_after") or {})
        if state_after:
            _atomic_json(state_path, state_after)
        samples = [Sample.from_dict(value) for value in committed.get("train_samples") or []]
        return GenerateFnOutput(samples=samples)

    expected_round = int(state.get("next_round", 0))
    if round_index != expected_round:
        raise ValueError(
            f"Miles rollout_id={round_index} but group {group_id!r} expects round "
            f"{expected_round}; use the matching Miles checkpoint or a new FRONTIERCS_RUN_ID"
        )
    if round_index >= memory_rounds:
        raise ValueError(
            f"group {group_id!r} already completed S={memory_rounds} memory rounds"
        )

    memory_in = str(state.get("memory") or "")
    _atomic_text(round_root / "memory_in.md", memory_in)
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
        diagnostics_limit=_int_setting(
            args,
            "FRONTIERCS_DIAGNOSTICS_CHARS",
            "frontiercs_diagnostics_chars_per_candidate",
            12_000,
        ),
    )
    seq_length = int(getattr(args, "seq_length", 32_768))
    act_max_new_tokens = _int_setting(
        args, "FRONTIERCS_ACT_MAX_NEW_TOKENS", "frontiercs_act_max_new_tokens", 25600
    )
    write_max_new_tokens = _int_setting(
        args, "FRONTIERCS_WRITE_MAX_NEW_TOKENS", "frontiercs_write_max_new_tokens", 25600
    )
    thinking = _bool_setting(
        args, "FRONTIERCS_ENABLE_THINKING", "frontiercs_enable_thinking", True
    )
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    seed_base = int(getattr(args, "rollout_seed", 42)) + round_index * 100_000
    best_candidates = dict(state.get("best_candidates") or {})

    work = []
    for problem_position, problem in enumerate(problems):
        previous = dict(best_candidates.get(problem.problem_id) or {})
        previous_code = (
            str(previous.get("code") or "") if act_code_context == "best" else None
        )
        for candidate_index in range(candidates_per_problem):
            work.append(
                _candidate(
                    input=input,
                    trace=trace,
                    judge=judge,
                    url=url,
                    group_id=group_id,
                    round_index=round_index,
                    problem=problem,
                    candidate_index=candidate_index,
                    memory=memory_in,
                    previous_code=previous_code,
                    act_max_new_tokens=act_max_new_tokens,
                    seq_length=seq_length,
                    thinking=thinking,
                    seed_base=seed_base + problem_position * 1_000,
                )
            )

    # Synchronization barrier: WRITE is unreachable until all G*K jobs have
    # generated and received evaluator feedback.
    candidate_pairs = list(await asyncio.gather(*work))
    records = [pair[0] for pair in candidate_pairs]
    act_samples = [pair[1] for pair in candidate_pairs]
    current_group_score = _group_score(records, problem_ids)

    train_samples: list[Sample] = list(act_samples)
    pending = state.get("pending_write_sample")
    previous_group_score = state.get("previous_group_score")
    if pending is not None:
        if previous_group_score is None:
            raise ValueError("pending WRITE exists without a previous group score")
        write_sample = Sample.from_dict(dict(pending))
        reward_mode = str(
            _env_or_arg(
                args,
                "FRONTIERCS_WRITE_REWARD_MODE",
                "frontiercs_write_reward_mode",
                "delta",
            )
        ).strip().lower()
        if reward_mode == "delta":
            write_reward = (current_group_score - float(previous_group_score)) / 100.0
        elif reward_mode == "downstream":
            write_reward = current_group_score / 100.0
        else:
            raise ValueError("frontiercs_write_reward_mode must be delta|downstream")
        write_sample.reward = write_reward
        write_sample.metadata = {
            **(write_sample.metadata or {}),
            "downstream_round": round_index,
            "previous_group_score_0_100": float(previous_group_score),
            "downstream_group_score_0_100": current_group_score,
            "write_reward_mode": reward_mode,
        }
        train_samples.append(write_sample)

    _update_best(state, records)
    memory_out = memory_in
    next_pending: dict[str, Any] | None = None
    write_summary: dict[str, Any] | None = None
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
            120_000,
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
            sampling_seed=seed_base + 99_999,
        )
        write_text, write_ids, write_logprobs, write_finish, write_engine_metadata = await _infer(
            url, write_prompt_ids, write_params
        )
        write_reasoning, write_visible = _visible_response(
            write_text, thinking=thinking
        )
        memory_out = clean_memory(write_visible)
        memory_changed = memory_out.strip() != memory_in.strip()
        memory_empty = not bool(memory_out.strip())
        packed_write = _pack_sample(
            seed=input.sample,
            prompt_label=f"Frontier-CS WRITE {group_id}/r{round_index}",
            prompt_ids=write_prompt_ids,
            response_text=write_text,
            response_ids=write_ids,
            response_logprobs=write_logprobs,
            finish=write_finish,
            metadata={
                "phase": "write",
                "group_id": group_id,
                "produced_round": round_index,
                "memory_round": round_index,
                "memory_changed": memory_changed,
                "memory_empty": memory_empty,
            },
            sample_index=int(input.sample.index or 0) * 100_000 + round_index * 1_000 + 999,
            engine_metadata=write_engine_metadata,
        )
        next_pending = packed_write.to_dict()
        trace.save_write(
            group_id,
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
            "trained_this_round": False,
            "reward_available_after_round": round_index + 1,
            "response_tokens": len(write_ids),
            "finish_reason": write_finish,
        }
    else:
        _atomic_text(round_root / "memory_out.md", memory_out)
        write_summary = {
            "generated": False,
            "reason": "final configured ACT round has no downstream round for WRITE reward",
        }

    state_after = {
        **state,
        "next_round": round_index + 1,
        "memory": memory_out,
        "previous_group_score": current_group_score,
        "pending_write_sample": next_pending,
    }
    round_value = {
        "schema_version": 1,
        "group_id": group_id,
        "round_index": round_index,
        "problem_ids": problem_ids,
        "candidates_per_problem": candidates_per_problem,
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
        "state_after": state_after,
        "train_samples": [sample.to_dict() for sample in train_samples],
    }
    # round.json is the commit record.  On a retry it repairs state.json and
    # replays these exact samples rather than sampling or judging again.
    _atomic_json(round_root / "round.json", round_value)
    _atomic_json(state_path, state_after)
    return GenerateFnOutput(samples=train_samples)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Miles ``--custom-generate-function-path`` entry point."""
    metadata = dict(input.sample.metadata or {})
    group_id = str(metadata.get("group_id") or "")
    run_key = str(_run_root(input.args)) + "::" + group_id
    lock = _GROUP_LOCKS.setdefault(run_key, asyncio.Lock())
    async with lock:
        return await _generate_locked(input)
