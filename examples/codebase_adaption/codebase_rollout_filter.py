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

from examples.codebase_adaption.codebase_advantage import _group_key

_EPS = 1e-8

# Dropped-group fraction of the current rollout, read by
# codebase_metrics.log_rollout_data and reported to wandb. The filter and the
# metrics hook run in the same RolloutManager process, so a module-level
# variable is a sufficient channel. Stays None when filtering is disabled
# (the metric then reports 0).
last_dropped_frac: float | None = None


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


def drop_zero_std_groups(samples: list, evaluation: bool) -> list:
    global last_dropped_frac
    if evaluation or os.environ.get("CODEBASE_DROP_ZERO_STD_GROUPS", "0") != "1":
        return samples

    flat = [s for s in _iter_samples(samples) if hasattr(s, "reward")]
    groups: dict[Any, list] = {}
    for s in flat:
        groups.setdefault(_group_key(s), []).append(s)

    drop_ids: set = set()
    n_drop_act = n_drop_write = 0
    for key, members in groups.items():
        rewards = [float(m.reward or 0.0) for m in members]
        if len(members) < 2 or (max(rewards) - min(rewards)) > _EPS:
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
        output = RolloutFnTrainOutput(
            samples=drop_zero_std_groups(output.samples, evaluation), metrics=output.metrics
        )
    return output
