"""Minimal scalar rollout metrics for Frontier-CS complete episodes."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.types import Sample

from .frontiercs_advantage import reward_post_process


logger = logging.getLogger(__name__)


def _flatten(values: Iterable[Any]) -> Iterable[Sample]:
    for value in values or []:
        if isinstance(value, list):
            yield from _flatten(value)
        else:
            yield value


def _mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _setting(args: Any, env_name: str, arg_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value not in (None, ""):
        return int(value)
    return int(getattr(args, arg_name, default))


def _response_length(sample: Sample) -> float:
    value = getattr(sample, "effective_response_length", None)
    return float(value if value is not None else sample.response_length)


def compute_frontiercs_metrics(args: Any, samples: list[Sample]) -> dict[str, float]:
    """Compute only the agreed Frontier-CS numeric metrics for one rollout step."""
    flat = list(_flatten(samples))
    act = [
        sample for sample in flat if (sample.metadata or {}).get("phase") != "write"
    ]
    write = [
        sample for sample in flat if (sample.metadata or {}).get("phase") == "write"
    ]
    if not act:
        raise ValueError("Frontier-CS metrics received no ACT samples")

    memory_rounds = _setting(
        args, "FRONTIERCS_MEMORY_ROUNDS", "frontiercs_memory_rounds", 4
    )
    metrics: dict[str, float] = {}

    by_round: dict[int, list[Sample]] = defaultdict(list)
    by_membership: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    memory_events: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in act:
        metadata = sample.metadata or {}
        round_index = int(metadata["memory_round"])
        by_round[round_index].append(sample)
        by_membership[
            (str(metadata.get("group_id") or ""), str(metadata.get("problem_id") or ""))
        ].append(sample)
        if metadata.get("memory_generated_after_round"):
            memory_events.setdefault(
                (str(metadata.get("group_id") or ""), round_index), metadata
            )

    for round_index in range(memory_rounds):
        round_samples = by_round.get(round_index) or []
        if not round_samples:
            raise ValueError(
                "Frontier-CS metrics received no ACT samples for round "
                f"{round_index}"
            )
        metrics[f"score/current_mean_r{round_index}"] = _mean(
            float((sample.metadata or {}).get("score_0_100") or 0.0)
            for sample in round_samples
        )
        metrics[f"score/positive_frac_r{round_index}"] = _mean(
            float((sample.metadata or {}).get("score_0_100") or 0.0) > 0.0
            for sample in round_samples
        )
        metrics[f"act/executed_frac_r{round_index}"] = _mean(
            bool((sample.metadata or {}).get("executed"))
            for sample in round_samples
        )

        best_scores: list[float] = []
        for membership_samples in by_membership.values():
            eligible = [
                float((sample.metadata or {}).get("score_0_100") or 0.0)
                for sample in membership_samples
                if int((sample.metadata or {}).get("memory_round") or 0) <= round_index
            ]
            if not eligible:
                raise ValueError(
                    f"Frontier-CS membership has no score through round {round_index}"
                )
            best_scores.append(max(eligible))
        metrics[f"score/best_mean_r{round_index}"] = _mean(best_scores)

    metrics["act/executed_frac"] = _mean(
        bool((sample.metadata or {}).get("executed")) for sample in act
    )
    metrics["act/compile_error_frac"] = _mean(
        bool((sample.metadata or {}).get("compile_error")) for sample in act
    )
    metrics["act/invalid_submission_frac"] = _mean(
        bool((sample.metadata or {}).get("invalid_submission")) for sample in act
    )
    metrics["act/length_stop_frac"] = _mean(
        sample.status == Sample.Status.TRUNCATED for sample in act
    )
    nonterminal_memory_events = [
        metadata
        for metadata in memory_events.values()
        if not bool(metadata.get("memory_terminal_after_round"))
    ]
    if nonterminal_memory_events:
        metrics["write/length_stop_frac"] = _mean(
            str(metadata.get("memory_finish_reason_after_round") or "") == "length"
            for metadata in nonterminal_memory_events
        )
    else:
        metrics["write/length_stop_frac"] = _mean(
            sample.status == Sample.Status.TRUNCATED for sample in write
        )

    metrics["sample_length/act_mean"] = _mean(
        _response_length(sample) for sample in act
    )
    if nonterminal_memory_events:
        metrics["sample_length/write_mean"] = _mean(
            float(metadata.get("memory_response_tokens_after_round") or 0.0)
            for metadata in nonterminal_memory_events
        )
    else:
        metrics["sample_length/write_mean"] = _mean(
            _response_length(sample) for sample in write
        )
    metrics["diagnostics/nonempty_frac"] = _mean(
        bool((sample.metadata or {}).get("has_diagnostics")) for sample in act
    )
    if nonterminal_memory_events:
        metrics["memory/changed_frac"] = _mean(
            bool(metadata.get("memory_changed_after_round"))
            for metadata in nonterminal_memory_events
        )
        metrics["memory/empty_frac"] = _mean(
            bool(metadata.get("memory_empty_after_round"))
            for metadata in nonterminal_memory_events
        )
    else:
        metrics["memory/changed_frac"] = _mean(
            bool((sample.metadata or {}).get("memory_changed")) for sample in write
        )
        metrics["memory/empty_frac"] = _mean(
            bool((sample.metadata or {}).get("memory_empty")) for sample in write
        )

    for produced_round in range(memory_rounds - 1):
        if nonterminal_memory_events:
            memory_lengths = [
                float(metadata.get("memory_tokens_after_round") or 0.0)
                for metadata in nonterminal_memory_events
                if int(metadata.get("memory_round") or 0) == produced_round
            ]
        else:
            memory_lengths = [
                float((sample.metadata or {}).get("memory_tokens") or 0.0)
                for sample in write
                if int((sample.metadata or {}).get("produced_round") or 0)
                == produced_round
            ]
        if not memory_lengths:
            raise ValueError(
                f"Frontier-CS metrics received no WRITE memory for round {produced_round}"
            )
        metrics[f"memory_length/after_r{produced_round}_mean"] = _mean(
            memory_lengths
        )

    metrics["training_signal/write_reward_mean"] = _mean(
        float(sample.reward or 0.0) for sample in write
    )

    temporal_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for sample in act:
        metadata = sample.metadata or {}
        temporal_groups[
            (str(metadata.get("group_id") or ""), str(metadata.get("problem_id") or ""))
        ].append(float(sample.reward or 0.0))
    zero_std_groups = sum(
        1
        for rewards in temporal_groups.values()
        if not rewards or max(rewards) - min(rewards) <= 1e-12
    )
    metrics["training_signal/grpo_zero_std_group_frac"] = (
        zero_std_groups / len(temporal_groups) if temporal_groups else 0.0
    )

    _, advantages = reward_post_process(args, flat)
    act_advantages = [
        abs(float(advantages[index]))
        for index, sample in enumerate(flat)
        if (sample.metadata or {}).get("phase") != "write"
    ]
    write_advantages = [
        abs(float(advantages[index]))
        for index, sample in enumerate(flat)
        if (sample.metadata or {}).get("phase") == "write"
    ]
    metrics["training_signal/act_advantage_abs_mean"] = _mean(
        act_advantages
    )
    metrics["training_signal/write_advantage_abs_mean"] = _mean(
        write_advantages
    )
    return metrics


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Log one numeric Frontier-CS metric row and suppress default sample metrics."""
    del rollout_time
    flat = list(_flatten(samples))
    metrics = dict(rollout_extra_metrics or {})
    metrics.update(compute_frontiercs_metrics(args, flat))
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    logger.info("Frontier-CS rollout %s metrics: %s", rollout_id, metrics)
    tracking_utils.log(args, metrics, step_key="rollout/step")
    return True
