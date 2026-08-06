"""DAPO-style zero-std group filtering (2026-07-21, for the next experiment; off by default).

Background: with 6p6 episodes, ~40-50% of instance-level sibling groups (8 rollouts
of the same instance) end up all-pass or all-fail (structurally driven by the hard
strategy tiers; see the pass@k difficulty table). These dead groups have constant
advantage 0 and contribute no policy gradient, yet they dilute the effective step
size by inflating the dynamic-global-batch denominator, and their KL term
(kl_loss_coef=0.01) keeps pulling dead-group tokens toward the ref model.

Mechanism: point --rollout-function-path at generate_rollout below. It runs the
stock sglang generate_rollout first, then physically removes samples belonging to
zero-std groups BEFORE flatten/dynamic-gbs computation, using the exact same
grouping key as codebase_advantage._group_key. Downstream
use_dynamic_global_batch_size recomputes gbs from the surviving sample count, so
the training side needs no changes.

Switch: CODEBASE_DROP_ZERO_STD_GROUPS=1 (train path only; eval is never filtered).
Guard: if every group is zero-std, skip filtering to avoid an empty batch
(the trim step would raise) and only log.
"""
import os
from argparse import Namespace
from typing import Any

from examples.codebase_adaption.codebase_advantage import _group_key, explore_beta

_EPS = 1e-8

# Dropped-group fraction of the current rollout, read by
# codebase_metrics.log_rollout_data and reported to wandb. The filter and the
# metrics hook run in the same RolloutManager process, so a module-level
# variable is a sufficient channel. Stays None when filtering is disabled
# (the metric then reports 0).
last_dropped_frac: float | None = None

# Pre-filter snapshot of the current rollout, read by codebase_metrics.log_rollout_data.
# The wandb success_frac is computed AFTER this filter, which silently removes all-pass
# and all-fail groups — exactly the groups a saturation-style reward (e.g. grace12's
# 12-step grace) is designed to produce. These stats capture the unbiased view.
last_prefilter_stats: dict | None = None


def _compute_prefilter_stats(samples: list) -> dict:
    flat = [s for s in _iter_samples(samples) if hasattr(s, "reward")]
    seen: set = set()
    succ: list[float] = []
    act_groups: dict[Any, list[bool]] = {}
    for s in flat:
        md = getattr(s, "metadata", None) or {}
        if md.get("phase") == "write" or md.get("success") is None:
            continue
        # Same dedup key as codebase_metrics' post-filter success_frac.
        key = (md.get("episode_id"), md.get("trial_pos"), getattr(s, "index", None))
        if key not in seen:
            seen.add(key)
            succ.append(1.0 if md.get("success") else 0.0)
        act_groups.setdefault(_group_key(s), []).append(bool(md.get("success")))
    n_groups = len(act_groups)
    return {
        "success_frac_prefilter": sum(succ) / len(succ) if succ else 0.0,
        "act_group_allpass_frac": (
            sum(1 for v in act_groups.values() if all(v)) / n_groups if n_groups else 0.0
        ),
        "act_group_allfail_frac": (
            sum(1 for v in act_groups.values() if not any(v)) / n_groups if n_groups else 0.0
        ),
    }


def _iter_samples(node):
    """samples is an arbitrarily nested list[list[...]]; walk all leaves."""
    if isinstance(node, list):
        for x in node:
            yield from _iter_samples(x)
    else:
        yield node


def _filter_tree(node, drop: set):
    """Remove dropped samples while preserving nesting; prune emptied sublists."""
    if not isinstance(node, list):
        return node if id(node) not in drop else None
    kept = []
    for x in node:
        y = _filter_tree(x, drop)
        if y is None or (isinstance(y, list) and not y):
            continue
        kept.append(y)
    return kept


def _drop_broken_samples(samples: list) -> list:
    """Drop aborted samples whose rollout_log_probs is None.

    A generation aborted mid-flight yields a sample that still carries a reward
    but no rollout_log_probs; slice_log_prob_with_cp then dies with
    `NoneType has no len()` (seen at grace12 r107, 2026-07-27). Only drop when
    such samples are a strict minority — if the whole batch has None the run
    simply does not use rollout log probs and dropping would empty the batch.
    """
    flat = [s for s in _iter_samples(samples) if hasattr(s, "reward")]
    broken = {id(s) for s in flat if getattr(s, "rollout_log_probs", "n/a") is None}
    if not broken or len(broken) * 2 >= len(flat):
        return samples
    print(
        f"[zero_std_filter] WARNING: dropping {len(broken)} aborted samples "
        "with rollout_log_probs=None",
        flush=True,
    )
    return _filter_tree(samples, broken)


def drop_zero_std_groups(samples: list, evaluation: bool, min_keep: int = 0) -> list:
    global last_dropped_frac, last_prefilter_stats
    if evaluation:
        return samples
    samples = _drop_broken_samples(samples)
    last_prefilter_stats = _compute_prefilter_stats(samples)
    if os.environ.get("CODEBASE_DROP_ZERO_STD_GROUPS", "0") != "1":
        return samples

    flat = [s for s in _iter_samples(samples) if hasattr(s, "reward")]
    groups: dict[Any, list] = {}
    for s in flat:
        groups.setdefault(_group_key(s), []).append(s)

    drop_ids: set = set()
    n_drop_act = n_drop_write = 0
    # Filter runs in the RolloutManager before reward_post_process, so with the ACT
    # exploration reward enabled an all-pass/all-fail ACT group can still carry gradient
    # through explore_score spread — dropping it would kill the explore signal's main
    # beneficiary groups (~40-50% of groups per the module docstring). env-only read:
    # the filter has no args object here, and the arm scripts enable beta via env anyway.
    explore_on = explore_beta(None) > 0
    for key, members in groups.items():
        rewards = [float(m.reward or 0.0) for m in members]
        if len(members) < 2 or (max(rewards) - min(rewards)) > _EPS:
            continue
        if explore_on and key[0] == "act":
            evs = [
                float(v)
                for v in ((m.metadata or {}).get("explore_score") for m in members)
                if v is not None
            ]
            if len(evs) >= 2 and (max(evs) - min(evs)) > _EPS:
                continue
        for m in members:
            drop_ids.add(id(m))
        if key[0] == "write":
            n_drop_write += 1
        else:
            n_drop_act += 1

    if not drop_ids:
        last_dropped_frac = 0.0
        print(f"[zero_std_filter] groups={len(groups)} all informative, nothing dropped", flush=True)
        return samples
    if len(drop_ids) >= len(flat):
        # Skip filtering but report truthfully: 100% of groups carry no signal.
        last_dropped_frac = 1.0
        print(
            f"[zero_std_filter] WARNING: all {len(groups)} groups are zero-std, "
            "skipping filter to preserve the batch",
            flush=True,
        )
        return samples
    if min_keep > 0 and len(flat) - len(drop_ids) < min_keep:
        # Survivors would not fill even one global batch and
        # postprocess_rollout_data raises on that; keep the full batch instead.
        last_dropped_frac = (len(groups) - 1) / len(groups) if len(groups) > 1 else 1.0
        print(
            f"[zero_std_filter] WARNING: only {len(flat) - len(drop_ids)} samples "
            f"would survive (< min_keep {min_keep}), skipping filter to preserve the batch",
            flush=True,
        )
        return samples

    filtered = _filter_tree(samples, drop_ids)
    n_groups_dropped = n_drop_act + n_drop_write
    last_dropped_frac = n_groups_dropped / len(groups)
    print(
        f"[zero_std_filter] dropped {n_groups_dropped}/{len(groups)} groups "
        f"(act {n_drop_act}, write {n_drop_write}) | samples {len(flat)} -> {len(flat) - len(drop_ids)}",
        flush=True,
    )
    return filtered


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False):
    # Lazy imports keep this module unit-testable in environments without sglang.
    from miles.rollout.base_types import RolloutFnTrainOutput
    from miles.rollout.sglang_rollout import generate_rollout as base_generate_rollout

    output = base_generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
    if isinstance(output, RolloutFnTrainOutput):
        min_keep = int(getattr(args, "global_batch_size", 0) or 0)
        output = RolloutFnTrainOutput(
            samples=drop_zero_std_groups(output.samples, evaluation, min_keep=min_keep),
            metrics=output.metrics,
        )
    return output
