"""Lightweight rollout metrics for codebase-adaptation co-training."""

from __future__ import annotations

from collections import defaultdict

from miles.ray.rollout.metrics import _compute_metrics_from_samples, _compute_perf_metrics_from_samples
from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step


def _flatten(xs):
    for x in xs or []:
        if isinstance(x, list):
            yield from _flatten(x)
        else:
            yield x


def _mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def _act_samples(samples):
    return [s for s in samples if (s.metadata or {}).get("phase") != "write"]


def _write_samples(samples):
    return [s for s in samples if (s.metadata or {}).get("phase") == "write"]


def _zero_std(vals: list[float]) -> bool:
    if len(vals) <= 1:
        return True
    first = vals[0]
    return all(abs(v - first) <= 1e-12 for v in vals[1:])


def _scalar_by_trial(samples, key: str) -> dict[int, list[float]]:
    by_trial: dict[int, list[float]] = defaultdict(list)
    for sample in samples:
        md = sample.metadata or {}
        trial = md.get("trial_pos")
        value = sample.reward if key == "reward" else md.get(key)
        if trial is None or value is None:
            continue
        by_trial[int(trial)].append(float(value))
    return by_trial


def _add_by_trial_metrics(out: dict[str, float], prefix: str, by_trial: dict[int, list[float]]) -> None:
    for trial in sorted(by_trial):
        out[f"{prefix}{trial}"] = _mean(by_trial[trial])


def _act_curve_metrics(samples) -> dict[str, float]:
    act = _act_samples(samples)
    turns_by_trial = _scalar_by_trial(act, "turns")
    reward_by_trial = _scalar_by_trial(act, "reward")
    gain_by_trial = _scalar_by_trial(act, "gain")

    out = {
        "codebase_turns/act_mean": _mean(v for vals in turns_by_trial.values() for v in vals),
        "codebase_reward/act_mean": _mean(v for vals in reward_by_trial.values() for v in vals),
        "codebase_gain/mean": _mean(v for vals in gain_by_trial.values() for v in vals),
    }
    _add_by_trial_metrics(out, "codebase_turns/act_by_index", turns_by_trial)
    _add_by_trial_metrics(out, "codebase_reward/mean_by_index", reward_by_trial)
    _add_by_trial_metrics(out, "codebase_gain/mean_by_index", gain_by_trial)
    return out


def _group_zero_std_metrics(samples) -> dict[str, float]:
    act_groups = defaultdict(list)
    write_groups = defaultdict(list)
    for sample in samples:
        md = sample.metadata or {}
        if md.get("phase") == "write":
            key = (sample.group_index, md.get("downstream_trial_pos"))
            write_groups[key].append(float(sample.reward or 0.0))
        else:
            key = (sample.group_index, md.get("trial_pos"))
            act_groups[key].append(float(sample.reward or 0.0))

    return {
        "codebase_grpo/act_group_zero_std_frac": (
            sum(1 for vals in act_groups.values() if _zero_std(vals)) / len(act_groups) if act_groups else 0.0
        ),
        "codebase_grpo/write_group_zero_std_frac": (
            sum(1 for vals in write_groups.values() if _zero_std(vals)) / len(write_groups) if write_groups else 0.0
        ),
    }


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    flat = list(_flatten(samples))
    base = {}
    base.update(_compute_metrics_from_samples(args, flat))
    base.update(_compute_perf_metrics_from_samples(args, flat, rollout_time))
    if rollout_extra_metrics:
        base.update(rollout_extra_metrics)

    act = _act_samples(flat)
    write = _write_samples(flat)
    by_phase = {
        "codebase_samples/act_n": float(len(act)),
        "codebase_samples/write_n": float(len(write)),
        "codebase_reward/write_mean": _mean(float(s.reward or 0.0) for s in write),
        "codebase_response/act_len_mean": _mean(float(s.effective_response_length) for s in act),
        "codebase_response/write_len_mean": _mean(float(s.effective_response_length) for s in write),
    }
    by_phase.update(_act_curve_metrics(flat))

    success = []
    seen = set()
    for s in act:
        md = s.metadata or {}
        # 去重键必须带 sample 身份(s.index): 同组 8 个 rollout 的 episode_id 相同,
        # 只按 (episode_id, trial_pos) 会把 8 次尝试当重复丢 7 个, success_frac 失真 8 倍样本量。
        key = (md.get("episode_id"), md.get("trial_pos"), s.index)
        if key in seen:
            continue
        seen.add(key)
        if md.get("success") is not None:
            success.append(1.0 if md.get("success") else 0.0)
    by_phase["codebase_reward/success_frac"] = _mean(success)
    # 超长样本比例(用户指令 2026-07-13): 截断/丢弃前 token 数 > seq_length 的样本占比
    by_phase["codebase_samples/act_overlong_frac"] = _mean(
        1.0 if (s.metadata or {}).get("overlong_pre_truncate") else 0.0 for s in act
    )
    by_phase["codebase_samples/write_overlong_frac"] = _mean(
        1.0 if (s.metadata or {}).get("overlong_pre_truncate") else 0.0 for s in write
    )
    by_phase.update(_group_zero_std_metrics(flat))

    metrics = {**base, **by_phase, "rollout/rollout_time": rollout_time}
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    tracking_utils.log(args, metrics, step_key="rollout/step")
    return True


def log_eval_rollout_data(rollout_id, args, data, extra_metrics=None) -> bool:
    log_dict = {**(extra_metrics or {})}
    for name, rec in (data or {}).items():
        samples = rec.get("samples") or []
        rewards = [float(s.reward or 0.0) for s in samples]
        successes = [
            1.0 if (s.metadata or {}).get("success") else 0.0
            for s in samples
            if (s.metadata or {}).get("success") is not None
        ]
        gains = [
            float((s.metadata or {}).get("gain"))
            for s in samples
            if (s.metadata or {}).get("gain") is not None
        ]
        log_dict[f"eval/{name}"] = _mean(rewards)
        log_dict[f"eval/{name}/codebase_samples/act_n"] = float(len(samples))
        log_dict[f"eval/{name}/codebase_reward/success_frac"] = _mean(successes)
        log_dict[f"eval/{name}/codebase_gain/mean"] = _mean(gains)
        log_dict[f"eval/{name}/codebase_response/act_len_mean"] = _mean(float(s.effective_response_length) for s in samples)
        eval_curve = _act_curve_metrics(samples)
        for key, value in eval_curve.items():
            log_dict[f"eval/{name}/{key}"] = value
    log_dict["eval/step"] = compute_rollout_step(args, rollout_id)
    # 同一行里附带 rollout/step(同值): wandb 图表按"行内共现"配对 x/y,
    # 工作区 x 轴设为 rollout/step 时 eval 曲线才有数据(2026-07-14 用户要求)。
    log_dict["rollout/step"] = log_dict["eval/step"]
    tracking_utils.log(args, log_dict, step_key="eval/step")
    return True
