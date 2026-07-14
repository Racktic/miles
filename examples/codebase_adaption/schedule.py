"""Episode construction for codebase-adaptation test-time training.

This module intentionally stores only public benchmark instance ids. It does
not copy problem statements, patches, tests, traces, or other benchmark data.
"""

from __future__ import annotations

import gzip
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_STAGE_IDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "tablib",
        (
            "jazzband__tablib-534",
            "jazzband__tablib-540",
            "jazzband__tablib-547",
            "jazzband__tablib-579",
            "jazzband__tablib-584",
            "jazzband__tablib-594",
            "jazzband__tablib-595",
            "jazzband__tablib-596",
            "jazzband__tablib-613",
        ),
    ),
    (
        "tenacity",
        (
            "jd__tenacity-597",
            "jd__tenacity-603",
            "jd__tenacity-604",
            "jd__tenacity-606",
            "jd__tenacity-609",
            "jd__tenacity-610",
            "jd__tenacity-611",
            "jd__tenacity-614",
            "jd__tenacity-615",
            "jd__tenacity-628",
        ),
    ),
)

DEFAULT_TRAIN_COUNTS = {"tablib": 6, "tenacity": 7}

REPO_BY_STAGE = {
    "tablib": "jazzband/tablib",
    "tenacity": "jd/tenacity",
    # swe_bench_cl 训练池的 8 个 repo(短名 -> 数据集 repo 全名),供训练 episode 的 stage_lookup 用。
    "astropy": "astropy/astropy",
    "pydata": "pydata/xarray",
    "pytest": "pytest-dev/pytest",
    "sphinx": "sphinx-doc/sphinx",
    "scikit": "scikit-learn/scikit-learn",
    "matplotlib": "matplotlib/matplotlib",
    "django": "django/django",
    "sympy": "sympy/sympy",
}


@dataclass(frozen=True)
class EpisodeOrder:
    instance_ids: list[str]
    stage_labels: list[str]

    @property
    def stage_sizes(self) -> list[int]:
        sizes: list[int] = []
        last_label = None
        for label in self.stage_labels:
            if label != last_label:
                sizes.append(1)
                last_label = label
            else:
                sizes[-1] += 1
        return sizes


def split_stage_ids(
    *,
    split: str,
    train_counts: dict[str, int] | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return fixed train or heldout ids by stage."""
    if split not in {"train", "heldout", "eval"}:
        raise ValueError(f"split must be train|heldout|eval, got {split!r}")
    counts = dict(DEFAULT_TRAIN_COUNTS)
    if train_counts:
        counts.update({str(k): int(v) for k, v in train_counts.items()})

    out = []
    for label, ids in DEFAULT_STAGE_IDS:
        n_train = counts.get(label, len(ids))
        if n_train < 0 or n_train > len(ids):
            raise ValueError(f"Invalid train count for {label}: {n_train}")
        selected = ids[:n_train] if split == "train" else ids[n_train:]
        out.append((label, tuple(selected)))
    return tuple(out)


def build_episode_order(
    *,
    split: str = "train",
    shuffle_seed: int = 0,
    order_rank: int | None = None,
    train_counts: dict[str, int] | None = None,
    shuffle_within_stage: bool = True,
) -> EpisodeOrder:
    """Build one stage-aligned episode order.

    Repo blocks are preserved, while ids are shuffled within each repo block.
    This matches the benchmark's blocked tablib-then-tenacity structure without
    letting heldout ids enter training.
    """
    instance_ids: list[str] = []
    stage_labels: list[str] = []

    stages = split_stage_ids(split=split, train_counts=train_counts)
    if order_rank is not None:
        rank = int(order_rank)
        total = 1
        for _label, ids_tuple in stages:
            total *= math.factorial(len(ids_tuple))
        if total <= 0:
            raise ValueError("Cannot rank an empty episode order.")
        rank %= total
        for label, ids_tuple in stages:
            n_orders = math.factorial(len(ids_tuple))
            stage_rank = rank % n_orders
            rank //= n_orders
            ids = _unrank_permutation(list(ids_tuple), stage_rank)
            instance_ids.extend(ids)
            stage_labels.extend([label] * len(ids))
        return EpisodeOrder(instance_ids=instance_ids, stage_labels=stage_labels)

    rng = random.Random(int(shuffle_seed))
    for label, ids_tuple in stages:
        ids = list(ids_tuple)
        if shuffle_within_stage:
            rng.shuffle(ids)
        instance_ids.extend(ids)
        stage_labels.extend([label] * len(ids))
    return EpisodeOrder(instance_ids=instance_ids, stage_labels=stage_labels)


def ranked_order_count(*, split: str = "heldout", train_counts: dict[str, int] | None = None) -> int:
    """Return the exact number of repo-internal orderings for a split."""
    total = 1
    for _label, ids_tuple in split_stage_ids(split=split, train_counts=train_counts):
        total *= math.factorial(len(ids_tuple))
    return total


def _unrank_permutation(items: list[str], rank: int) -> list[str]:
    """Return the rank-th lexicographic permutation without materializing all permutations."""
    remaining = list(items)
    out: list[str] = []
    rank = int(rank)
    for n in range(len(remaining), 0, -1):
        block = math.factorial(n - 1)
        idx = rank // block
        rank %= block
        out.append(remaining.pop(idx))
    return out


def stage_lookup_for(order: EpisodeOrder) -> list[dict[str, object]]:
    """Build the private clbench stage lookup payload for an EpisodeOrder."""
    return [
        {
            "variant": label,
            "label": label.capitalize(),
            "notify_change": label != order.stage_labels[0],
            "repo": REPO_BY_STAGE.get(label),
            "variant_display_name": label.capitalize(),
        }
        for label in order.stage_labels
    ]


def _open_json(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def load_baseline_rewards(path: str | Path | None) -> dict[str, float]:
    """Load baseline rewards by instance_id from a trace or viewer artifact.

    Supported inputs:
    - viewer artifact with summary.baseline.instance_outcomes
    - viewer artifact with summary.runs[*].instance_outcomes baseline_reward
    - raw baseline trace with top-level instance_outcomes
    """
    if not path:
        return {}
    with _open_json(path) as f:
        payload = json.load(f)

    outcomes = (
        (((payload.get("summary") or {}).get("baseline") or {}).get("instance_outcomes"))
        or payload.get("instance_outcomes")
        or ((payload.get("execution") or {}).get("instance_outcomes"))
        or []
    )
    rewards: dict[str, float] = {}
    for outcome in outcomes:
        instance_id = outcome.get("instance_id")
        if instance_id is not None and outcome.get("reward") is not None:
            rewards[str(instance_id)] = float(outcome["reward"])

    if rewards:
        return rewards

    # Some viewer artifacts only attach baseline_reward to each rollout outcome.
    for run in ((payload.get("summary") or {}).get("runs") or []):
        for outcome in run.get("instance_outcomes") or []:
            instance_id = outcome.get("instance_id")
            baseline_reward = outcome.get("baseline_reward")
            if instance_id is not None and baseline_reward is not None:
                rewards.setdefault(str(instance_id), float(baseline_reward))
    return rewards


def write_seed_stream_jsonl(path: str | Path, *, split: str = "train") -> None:
    """Write one metadata row; rollout derives unbounded shuffle seeds from group_index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        row: dict[str, Any] = {
            "prompt": "",
            "metadata": {
                "split": split,
            },
        }
        f.write(json.dumps(row) + "\n")
