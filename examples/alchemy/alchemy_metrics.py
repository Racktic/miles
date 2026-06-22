"""W&B rollout metrics for Symbolic Alchemy training.

Hooked through ``--custom-rollout-log-function-path``. The function logs the
standard Miles rollout/perf metrics plus Alchemy-specific action, normalized
score, WRITE-signal, GRPO-group, and response-health metrics.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from miles.ray.rollout.metrics import _compute_metrics_from_samples, _compute_perf_metrics_from_samples
from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step, dict_add_prefix
from miles.utils.types import Sample


logger = logging.getLogger(__name__)

_ORACLE_CACHE: dict[str, list[float]] | None = None


def _flatten(xs):
    for x in xs or []:
        if isinstance(x, list):
            yield from _flatten(x)
        else:
            yield x


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _zero_std(vals: list[float]) -> bool:
    if len(vals) <= 1:
        return True
    first = vals[0]
    return all(abs(v - first) <= 1e-12 for v in vals[1:])


def _oracle_cache_path() -> Path:
    env_path = os.environ.get("ALCHEMY_ORACLE_CACHE")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "eval" / "oracle_cache.json"


def _load_oracle_cache() -> dict[str, list[float]]:
    global _ORACLE_CACHE
    if _ORACLE_CACHE is None:
        with _oracle_cache_path().open() as f:
            _ORACLE_CACHE = json.load(f)
    return _ORACLE_CACHE


def _episode_stats(samples: list[Sample]) -> list[dict[str, Any]]:
    by_episode: dict[Any, dict[str, Any]] = {}
    for s in samples:
        md = s.metadata or {}
        stats = md.get("alchemy_episode_stats")
        if not stats:
            continue
        ep_key = md.get("episode_id", (s.group_index, s.index))
        by_episode.setdefault(ep_key, stats)
    return list(by_episode.values())


def _action_metrics(samples: list[Sample]) -> dict[str, float]:
    episodes = _episode_stats(samples)
    counts = defaultdict(int)
    for ep in episodes:
        for key, value in (ep.get("actions") or {}).items():
            counts[key] += int(value)

    valid = counts["valid_total"]
    attempted = counts["attempted"]
    n_ep = len(episodes)
    return {
        "alchemy_action/potion_frac": counts["potion"] / valid if valid else 0.0,
        "alchemy_action/cauldron_frac": counts["cauldron"] / valid if valid else 0.0,
        "alchemy_action/end_trial_frac": counts["end_trial"] / valid if valid else 0.0,
        "alchemy_action/invalid_frac": counts["invalid"] / attempted if attempted else 0.0,
        "alchemy_action/zero_potion_episode_frac": counts["zero_potion"] / n_ep if n_ep else 0.0,
    }


def _act_score_metrics(samples: list[Sample]) -> dict[str, float]:
    oracle = _load_oracle_cache()
    raw_vals = []
    norm_vals = []
    norm_by_trial: dict[int, list[float]] = defaultdict(list)

    for s in samples:
        md = s.metadata or {}
        if md.get("phase") == "write":
            continue
        if md.get("no_memory"):
            # no-memory: ONE episode sample carries the whole per_trial_scores list. Expand it into
            # per-trial raw/normalized values so act_raw_mean / act_norm_mean / per-trial / improve
            # match the memory path (which has one sample per trial).
            pts = md.get("per_trial_scores") or []
            ep = md.get("episode_index")
            oracle_scores = oracle.get(str(ep)) if ep is not None else None
            for k, rawk in enumerate(pts):
                raw_vals.append(float(rawk))
                if not oracle_scores or k >= len(oracle_scores):
                    continue
                denom = float(oracle_scores[k])
                if denom <= 0:
                    continue
                norm = float(rawk) / denom
                norm_vals.append(norm)
                norm_by_trial[int(k)].append(norm)
            continue
        raw = float(s.reward or 0.0)
        raw_vals.append(raw)
        ep = md.get("episode_index")
        trial = md.get("trial_pos")
        if ep is None or trial is None:
            continue
        oracle_scores = oracle.get(str(ep))
        if not oracle_scores:
            continue
        trial = int(trial)
        if trial >= len(oracle_scores):
            continue
        denom = float(oracle_scores[trial])
        if denom <= 0:
            continue
        norm = raw / denom
        norm_vals.append(norm)
        norm_by_trial[trial].append(norm)

    out = {
        "alchemy_score/act_n": float(len(raw_vals)),
        "alchemy_score/act_raw_mean": _mean(raw_vals),
        "alchemy_score/act_norm_mean": _mean(norm_vals),
    }
    for k in range(10):
        out[f"alchemy_score/norm_trial_{k}_mean"] = _mean(norm_by_trial.get(k, []))
    early = [x for k in range(5) for x in norm_by_trial.get(k, [])]
    late = [x for k in range(5, 10) for x in norm_by_trial.get(k, [])]
    out["alchemy_score/norm_improve"] = _mean(late) - _mean(early) if early and late else 0.0
    return out


def _write_metrics(samples: list[Sample]) -> dict[str, float]:
    write_samples = [s for s in samples if (s.metadata or {}).get("phase") == "write"]
    write_rewards = [float(s.reward or 0.0) for s in write_samples]

    episodes = _episode_stats(samples)
    total = 0
    kept = 0
    fk_counts = []
    accs = []
    for ep in episodes:
        w = ep.get("write") or {}
        total += int(w.get("total", 0) or 0)
        kept += int(w.get("kept", 0) or 0)
        fk_counts.extend(int(x) for x in (w.get("fk_counts") or []))
        accs.extend(float(x) for x in (w.get("accs") or []) if x is not None)

    return {
        "alchemy_write/write_n": float(len(write_samples)),
        "alchemy_write/write_mean": _mean(write_rewards),
        "alchemy_write/kept_frac": kept / total if total else 0.0,
        "alchemy_write/fk_mean": _mean(fk_counts),
        "alchemy_write/fk_zero_frac": sum(1 for x in fk_counts if x == 0) / len(fk_counts) if fk_counts else 0.0,
        "alchemy_write/fk_ge3_frac": sum(1 for x in fk_counts if x >= 3) / len(fk_counts) if fk_counts else 0.0,
        "alchemy_write/acc_mean": _mean(accs),
    }


def _grpo_metrics(samples: list[Sample]) -> dict[str, float]:
    act_groups = defaultdict(list)
    write_groups = defaultdict(list)
    for s in samples:
        md = s.metadata or {}
        if md.get("no_memory"):
            # no-memory: GRPO groups are (group_index, trial_pos); expand per_trial_scores so the
            # zero-std diagnostic reflects per-trial groups (same grouping the advantage hook uses).
            for k, v in enumerate(md.get("per_trial_scores") or []):
                act_groups[(s.group_index, k)].append(float(v))
            continue
        if md.get("phase") == "write":
            key = (md.get("episode_id"), md.get("rewrite_idx"))
            write_groups[key].append(float(s.reward or 0.0))
        else:
            key = (s.group_index, md.get("trial_pos"))
            act_groups[key].append(float(s.reward or 0.0))

    return {
        "alchemy_grpo/act_group_zero_std_frac": (
            sum(1 for vals in act_groups.values() if _zero_std(vals)) / len(act_groups) if act_groups else 0.0
        ),
        "alchemy_grpo/write_group_zero_std_frac": (
            sum(1 for vals in write_groups.values() if _zero_std(vals)) / len(write_groups) if write_groups else 0.0
        ),
    }


def _response_metrics(samples: list[Sample]) -> dict[str, float]:
    act = [s for s in samples if (s.metadata or {}).get("phase") != "write"]
    write = [s for s in samples if (s.metadata or {}).get("phase") == "write"]
    return {
        "alchemy_response/act_len_mean": _mean([float(s.effective_response_length) for s in act]),
        "alchemy_response/write_len_mean": _mean([float(s.effective_response_length) for s in write]),
        "alchemy_response/act_truncated_frac": (
            sum(1 for s in act if s.status == Sample.Status.TRUNCATED) / len(act) if act else 0.0
        ),
        "alchemy_response/write_truncated_frac": (
            sum(1 for s in write if s.status == Sample.Status.TRUNCATED) / len(write) if write else 0.0
        ),
    }


def _alchemy_metrics(samples: list[Sample]) -> dict[str, float]:
    out = {}
    out.update(_action_metrics(samples))
    out.update(_act_score_metrics(samples))
    out.update(_write_metrics(samples))
    out.update(_grpo_metrics(samples))
    out.update(_response_metrics(samples))
    return out


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    if args.load_debug_rollout_data:
        return True

    flat_samples = list(_flatten(samples))
    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, flat_samples), "rollout/")
    log_dict |= dict_add_prefix(_compute_perf_metrics_from_samples(args, flat_samples, rollout_time), "perf/")
    log_dict |= _alchemy_metrics(flat_samples)

    logger.info("alchemy rollout %s: %s", rollout_id, log_dict)
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
    return True


def _alchemy_eval_metrics(data, extra_metrics: dict[str, Any] | None = None) -> dict[str, float]:
    log_dict = {**(extra_metrics or {})}
    for name, rec in (data or {}).items():
        samples = rec.get("samples") or []
        act_metrics = _act_score_metrics([s for s in samples if (s.metadata or {}).get("phase") != "write"])
        prefix = f"eval/{name}/"
        log_dict[prefix + "norm_score"] = act_metrics["alchemy_score/act_norm_mean"]
        log_dict[prefix + "norm_improve"] = act_metrics["alchemy_score/norm_improve"]
        log_dict[prefix + "act_n"] = act_metrics["alchemy_score/act_n"]
        log_dict[prefix + "act_raw_mean"] = act_metrics["alchemy_score/act_raw_mean"]
        for k in range(10):
            log_dict[prefix + f"norm_trial_{k}_mean"] = act_metrics[f"alchemy_score/norm_trial_{k}_mean"]
    return log_dict


def log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None) -> bool:
    log_dict = _alchemy_eval_metrics(data, extra_metrics)
    logger.info("alchemy eval %s: %s", rollout_id, log_dict)
    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking_utils.log(args, log_dict, step_key="eval/step")
    return True
