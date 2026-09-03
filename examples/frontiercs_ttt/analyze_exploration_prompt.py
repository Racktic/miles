#!/usr/bin/env python3
"""Score offline Frontier-CS memory transitions with the current exploration judge."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from examples.frontiercs_ttt.frontiercs_exploration_judge import (
    EXPLORE_DIMS,
    JUDGE_PROMPT_HASH,
    SYSTEM_PROMPT,
    judge_config,
    judge_memory_delta,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _model_family(run_id: str) -> str:
    prefixes = (
        "qwen36_27b",
        "qwen35_35b_a3b",
        "qwen35_9b",
        "qwen35_4b",
        "gpt55",
        "standalone_qwen35_4b",
    )
    return next((prefix for prefix in prefixes if run_id.startswith(prefix)), "other")


def _pair_hash(previous_memory: str, updated_memory: str) -> str:
    return hashlib.sha256(
        (previous_memory + "\0" + updated_memory).encode("utf-8")
    ).hexdigest()


def _discover(trace_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for memory_in_path in sorted(trace_root.glob("*/groups/*/round_*/memory_in.md")):
        memory_out_path = memory_in_path.with_name("memory_out.md")
        if not memory_out_path.is_file():
            continue
        relative = memory_in_path.relative_to(trace_root)
        run_id, marker, group_id, round_name, _ = relative.parts
        if marker != "groups" or not round_name.startswith("round_"):
            continue
        previous_memory = memory_in_path.read_text(encoding="utf-8")
        updated_memory = memory_out_path.read_text(encoding="utf-8")
        records.append(
            {
                "run_id": run_id,
                "model_family": _model_family(run_id),
                "group_id": group_id,
                "round_index": int(round_name.split("_", 1)[1]),
                "memory_in_path": str(memory_in_path.resolve()),
                "memory_out_path": str(memory_out_path.resolve()),
                "previous_memory": previous_memory,
                "updated_memory": updated_memory,
                "pair_hash": _pair_hash(previous_memory, updated_memory),
                "previous_chars": len(previous_memory),
                "updated_chars": len(updated_memory),
                "changed": previous_memory.strip() != updated_memory.strip(),
            }
        )
    return records


def _balanced_sample(
    records: list[dict[str, Any]],
    *,
    model_families: list[str],
    per_model: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample each requested model equally and balance its sample over rounds."""
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for family in model_families:
        family_records = [
            record for record in records if record["model_family"] == family
        ]
        if len(family_records) < per_model:
            raise ValueError(
                f"model family {family!r} has {len(family_records)} records, "
                f"below requested {per_model}"
            )
        by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in family_records:
            by_round[int(record["round_index"])].append(record)
        round_indices = sorted(by_round)
        base, remainder = divmod(per_model, len(round_indices))
        for position, round_index in enumerate(round_indices):
            quota = base + int(position < remainder)
            candidates = by_round[round_index]
            if len(candidates) < quota:
                raise ValueError(
                    f"model family {family!r} round {round_index} has "
                    f"{len(candidates)} records, below quota {quota}"
                )
            selected.extend(rng.sample(candidates, quota))
    return sorted(
        selected,
        key=lambda record: (
            record["model_family"],
            record["run_id"],
            record["group_id"],
            record["round_index"],
        ),
    )


def _distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [record for record in records if record.get("result") is not None]
    values = [float(record["result"]["explore_score"]) for record in scored]
    score_hist = Counter(f"{value:.3f}" for value in values)
    dimensions: dict[str, Any] = {}
    for name in EXPLORE_DIMS:
        dim_values = [int(record["result"][name]) for record in scored]
        dimensions[name] = {
            "mean": statistics.fmean(dim_values) if dim_values else None,
            "counts": {str(level): dim_values.count(level) for level in range(3)},
            "fractions": {
                str(level): dim_values.count(level) / len(dim_values)
                if dim_values
                else 0.0
                for level in range(3)
            },
        }
    return {
        "total": len(records),
        "scored": len(scored),
        "unavailable": len(records) - len(scored),
        "mean": statistics.fmean(values) if values else None,
        "std_population": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "zero_fraction": values.count(0.0) / len(values) if values else 0.0,
        "positive_fraction": sum(value > 0.0 for value in values) / len(values)
        if values
        else 0.0,
        "score_histogram": dict(sorted(score_hist.items(), key=lambda item: float(item[0]))),
        "dimensions": dimensions,
    }


def _group_distributions(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: _distribution(values) for name, values in sorted(groups.items())}


def _markdown_table(title: str, values: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| Slice | N | Scored | Mean | Std | Zero | Positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in values.items():
        mean = value["mean"]
        std = value["std_population"]
        lines.append(
            f"| {name} | {value['total']} | {value['scored']} | "
            f"{mean:.4f} | {std:.4f} | {100 * value['zero_fraction']:.1f}% | "
            f"{100 * value['positive_fraction']:.1f}% |"
        )
    lines.append("")
    return lines


def _write_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    unique_scores: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    occurrence_records: list[dict[str, Any]] = []
    for record in records:
        occurrence_records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"previous_memory", "updated_memory"}
            }
            | {"result": unique_scores.get(record["pair_hash"])}
        )

    unique_records = [
        {
            "pair_hash": pair_hash,
            "result": result,
        }
        for pair_hash, result in unique_scores.items()
    ]
    summary = {
        "metadata": metadata,
        "occurrence_weighted": _distribution(occurrence_records),
        "unique_pair_weighted": _distribution(unique_records),
        "by_model_family": _group_distributions(occurrence_records, "model_family"),
        "by_round": _group_distributions(occurrence_records, "round_index"),
        "by_run": _group_distributions(occurrence_records, "run_id"),
    }
    _atomic_json(output_dir / "summary.json", summary)
    with (output_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for record in occurrence_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    overall = summary["occurrence_weighted"]
    lines = [
        "# Frontier-CS Exploration Prompt Offline Score Distribution",
        "",
        f"- Prompt hash: `{metadata['prompt_hash']}`",
        f"- Judge model: `{metadata['judge_model']}`",
        f"- Trace root: `{metadata['trace_root']}`",
        f"- Memory-transition occurrences: {metadata['occurrences']}",
        f"- Unique memory pairs: {metadata['unique_pairs']}",
        f"- Successfully scored occurrences: {overall['scored']}",
        f"- Overall mean: {overall['mean']:.4f}",
        f"- Overall population std: {overall['std_population']:.4f}",
        "",
        "## Overall score histogram",
        "",
        "| Score | Count | Fraction |",
        "|---:|---:|---:|",
    ]
    for score, count in overall["score_histogram"].items():
        fraction = count / overall["scored"] if overall["scored"] else 0.0
        lines.append(f"| {score} | {count} | {100 * fraction:.1f}% |")
    lines.extend(["", "## Dimension distributions", ""])
    lines.extend(
        [
            "| Dimension | Mean | Score 0 | Score 1 | Score 2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, value in overall["dimensions"].items():
        counts = value["counts"]
        lines.append(
            f"| {name} | {value['mean']:.4f} | {counts['0']} | "
            f"{counts['1']} | {counts['2']} |"
        )
    lines.append("")
    lines.extend(_markdown_table("By model family", summary["by_model_family"]))
    lines.extend(_markdown_table("By round", summary["by_round"]))
    lines.extend(_markdown_table("By run", summary["by_run"]))
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _run(args: argparse.Namespace) -> None:
    trace_root = args.trace_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _discover(trace_root)
    discovered_count = len(records)
    if args.balanced_per_model is not None:
        records = _balanced_sample(
            records,
            model_families=args.model_families.split(","),
            per_model=args.balanced_per_model,
            seed=args.sample_seed,
        )
    elif args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"no complete memory pairs found under {trace_root}")

    os.environ["FRONTIERCS_EXPLORE_JUDGE_CONCURRENCY"] = str(args.concurrency)
    config = judge_config()
    if not config["api_key"]:
        raise ValueError("no exploration/OpenAI API key is available")
    metadata = {
        "schema_version": 1,
        "trace_root": str(trace_root),
        "output_dir": str(output_dir),
        "prompt_hash": JUDGE_PROMPT_HASH,
        "judge_model": config["model"],
        "judge_api_base": config["api_base"],
        "occurrences": len(records),
        "discovered_occurrences": discovered_count,
        "unique_pairs": len({record["pair_hash"] for record in records}),
        "concurrency": args.concurrency,
        "sampling": (
            {
                "mode": "balanced_by_model_and_round",
                "model_families": args.model_families.split(","),
                "per_model": args.balanced_per_model,
                "seed": args.sample_seed,
            }
            if args.balanced_per_model is not None
            else {"mode": "first_n" if args.limit is not None else "all"}
        ),
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    (output_dir / "SYSTEM_PROMPT.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")

    checkpoint_path = output_dir / "unique_scores.json"
    checkpoint_payload: dict[str, Any] = {}
    if checkpoint_path.is_file():
        loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if loaded.get("prompt_hash") != JUDGE_PROMPT_HASH:
            raise ValueError("checkpoint prompt hash differs; use a new output directory")
        if loaded.get("judge_model") != config["model"]:
            raise ValueError("checkpoint judge model differs; use a new output directory")
        checkpoint_payload = dict(loaded.get("scores") or {})

    representative: dict[str, dict[str, Any]] = {}
    for record in records:
        representative.setdefault(record["pair_hash"], record)
    pending = [
        record
        for pair_hash, record in representative.items()
        if pair_hash not in checkpoint_payload or checkpoint_payload[pair_hash] is None
    ]
    print(
        f"[exploration-eval] occurrences={len(records)} "
        f"unique={len(representative)} pending={len(pending)} model={config['model']}",
        flush=True,
    )

    async def score(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        result = await judge_memory_delta(
            record["previous_memory"], record["updated_memory"]
        )
        return str(record["pair_hash"]), result

    tasks = [asyncio.create_task(score(record)) for record in pending]
    completed = 0
    for task in asyncio.as_completed(tasks):
        pair_hash, result = await task
        checkpoint_payload[pair_hash] = result
        completed += 1
        if completed % 10 == 0 or completed == len(tasks):
            _atomic_json(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "prompt_hash": JUDGE_PROMPT_HASH,
                    "judge_model": config["model"],
                    "scores": checkpoint_payload,
                },
            )
            print(
                f"[exploration-eval] completed={completed}/{len(tasks)}",
                flush=True,
            )

    _write_outputs(output_dir, records, checkpoint_payload, metadata)
    unavailable = sum(value is None for value in checkpoint_payload.values())
    print(
        f"[exploration-eval] done output={output_dir} unavailable_unique={unavailable}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--balanced-per-model", type=int)
    parser.add_argument(
        "--model-families",
        default="gpt55,qwen35_4b,qwen35_9b,qwen35_35b_a3b,qwen36_27b",
    )
    parser.add_argument("--sample-seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.balanced_per_model is not None and args.balanced_per_model < 1:
        parser.error("--balanced-per-model must be at least 1")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
