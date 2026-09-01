"""Configurable ACT/WRITE advantages for Frontier-CS memory co-training.

The rollout stores task scores in [0, 1].  ACT and WRITE are deliberately
handled as two streams: ACT may use within-problem group-relative advantages,
a fixed task baseline, or the raw reward; WRITE defaults to its delayed group
score delta without requiring sibling WRITE samples.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


_BASELINE_CACHE: tuple[str, dict[str, float]] | None = None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _env_or_arg(args: Any, env: str, arg: str, default: Any) -> Any:
    value = os.environ.get(env)
    if value not in (None, ""):
        return value
    return getattr(args, arg, default)


def _normalize_score(value: Any) -> float:
    score = float(value or 0.0)
    # Baseline artifacts may be copied from Frontier-CS reports that use 0-100.
    return score / 100.0 if abs(score) > 1.0 else score


def _load_task_baselines(args: Any) -> dict[str, float]:
    global _BASELINE_CACHE
    raw_path = _env_or_arg(
        args,
        "FRONTIERCS_TASK_BASELINE_ARTIFACT",
        "frontiercs_task_baseline_artifact",
        "",
    )
    path = str(raw_path or "")
    if not path:
        return {}
    resolved = str(Path(path).expanduser().resolve())
    if _BASELINE_CACHE is not None and _BASELINE_CACHE[0] == resolved:
        return _BASELINE_CACHE[1]
    payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Frontier-CS task baseline must be a JSON object: {resolved}")
    baselines = {str(key): _normalize_score(value) for key, value in payload.items()}
    _BASELINE_CACHE = (resolved, baselines)
    return baselines


def _standardized(values: list[float], *, use_std: bool) -> list[float]:
    if not values:
        return []
    center = _mean(values)
    centered = [value - center for value in values]
    if not use_std or len(values) == 1:
        return centered
    # Match Miles' default ``torch.std`` GRPO normalization (correction=1).
    variance = sum(value * value for value in centered) / (len(centered) - 1)
    denominator = variance**0.5 + 1e-6
    return [value / denominator for value in centered]


def reward_post_process(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Return ``(raw_rewards, token advantages)`` for Miles.

    ``FRONTIERCS_ACT_ADVANTAGE_MODE`` / ``frontiercs_act_advantage_mode``:
      - ``group_relative``: normalize K candidates for the same problem+round.
      - ``temporal_problem_relative``: normalize all S*K answers for one
        problem across the complete frozen-policy episode.
      - ``task_baseline``: subtract a fixed per-problem baseline (works for K=1).
      - ``raw``: direct task reward (also works for K=1).

    WRITE uses its delayed downstream reward directly by default.  It is never
    silently whitened with unrelated ACT samples.
    """

    raw_rewards = [float(sample.reward or 0.0) for sample in samples]
    advantages = [0.0] * len(samples)
    act_mode = str(
        _env_or_arg(
            args,
            "FRONTIERCS_ACT_ADVANTAGE_MODE",
            "frontiercs_act_advantage_mode",
            "raw",
        )
    ).strip().lower()
    write_mode = str(
        _env_or_arg(
            args,
            "FRONTIERCS_WRITE_ADVANTAGE_MODE",
            "frontiercs_write_advantage_mode",
            "direct",
        )
    ).strip().lower()
    use_std = bool(getattr(args, "grpo_std_normalization", True))

    if act_mode not in {
        "group_relative",
        "temporal_problem_relative",
        "task_baseline",
        "raw",
    }:
        raise ValueError(f"unsupported Frontier-CS ACT advantage mode: {act_mode!r}")
    if write_mode not in {"direct", "positive_only", "center_by_round"}:
        raise ValueError(f"unsupported Frontier-CS WRITE advantage mode: {write_mode!r}")

    act_indices: list[int] = []
    write_indices: list[int] = []
    for index, sample in enumerate(samples):
        phase = str((sample.metadata or {}).get("phase") or "act")
        (write_indices if phase == "write" else act_indices).append(index)

    if act_mode == "raw":
        for index in act_indices:
            advantages[index] = raw_rewards[index]
    elif act_mode == "task_baseline":
        baselines = _load_task_baselines(args)
        missing: set[str] = set()
        for index in act_indices:
            problem_id = str((samples[index].metadata or {}).get("problem_id") or "")
            if problem_id not in baselines:
                missing.add(problem_id)
                continue
            advantages[index] = raw_rewards[index] - baselines[problem_id]
        if missing:
            raise ValueError(
                "task_baseline mode has no baseline for problem IDs: "
                + ", ".join(sorted(missing))
            )
    elif act_mode == "group_relative":
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index in act_indices:
            metadata = samples[index].metadata or {}
            key = (
                metadata.get("group_id"),
                metadata.get("memory_round"),
                metadata.get("problem_id"),
            )
            groups[key].append(index)
        for key, indices in groups.items():
            if len(indices) < 2:
                raise ValueError(
                    "group_relative ACT advantage requires K>=2; "
                    f"group {key!r} contains one sample"
                )
            normalized = _standardized(
                [raw_rewards[index] for index in indices], use_std=use_std
            )
            for index, value in zip(indices, normalized, strict=True):
                advantages[index] = value
    else:
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index in act_indices:
            metadata = samples[index].metadata or {}
            key = (metadata.get("group_id"), metadata.get("problem_id"))
            groups[key].append(index)

        memory_rounds = int(
            _env_or_arg(
                args,
                "FRONTIERCS_MEMORY_ROUNDS",
                "frontiercs_memory_rounds",
                4,
            )
        )
        candidates_per_problem = int(
            _env_or_arg(
                args,
                "FRONTIERCS_CANDIDATES_PER_PROBLEM",
                "frontiercs_candidates_per_problem",
                1,
            )
        )
        expected = memory_rounds * candidates_per_problem
        for key, indices in groups.items():
            if len(indices) != expected:
                raise ValueError(
                    "temporal_problem_relative ACT advantage requires every "
                    f"episode/problem group to contain S*K={expected} samples; "
                    f"group {key!r} contains {len(indices)}"
                )
            normalized = _standardized(
                [raw_rewards[index] for index in indices], use_std=True
            )
            for index, value in zip(indices, normalized, strict=True):
                advantages[index] = value

    if write_mode == "direct":
        for index in write_indices:
            advantages[index] = raw_rewards[index]
    elif write_mode == "positive_only":
        for index in write_indices:
            advantages[index] = max(0.0, raw_rewards[index])
    else:
        groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index in write_indices:
            metadata = samples[index].metadata or {}
            key = (metadata.get("produced_round"), metadata.get("downstream_round"))
            groups[key].append(index)
        for indices in groups.values():
            normalized = _standardized(
                [raw_rewards[index] for index in indices], use_std=use_std
            )
            for index, value in zip(indices, normalized, strict=True):
                advantages[index] = value

    write_scale = float(
        _env_or_arg(
            args,
            "FRONTIERCS_WRITE_ADVANTAGE_SCALE",
            "frontiercs_write_advantage_scale",
            1.0,
        )
        or 0.0
    )
    for index in write_indices:
        advantages[index] *= write_scale

    explore_beta = float(
        _env_or_arg(
            args,
            "FRONTIERCS_ACT_EXPLORE_BETA",
            "frontiercs_act_explore_beta",
            0.0,
        )
        or 0.0
    )
    if explore_beta:
        explore_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for index in act_indices:
            metadata = samples[index].metadata or {}
            if metadata.get("explore_score") is None:
                continue
            key = (
                metadata.get("group_id"),
                metadata.get("memory_round"),
                metadata.get("problem_id"),
            )
            explore_groups[key].append(index)
        for indices in explore_groups.values():
            values = [
                float((samples[index].metadata or {}).get("explore_score"))
                for index in indices
            ]
            shaped = (
                _standardized(values, use_std=use_std)
                if len(indices) > 1
                else values
            )
            for index, value in zip(indices, shaped, strict=True):
                advantages[index] += explore_beta * value

    return raw_rewards, advantages
