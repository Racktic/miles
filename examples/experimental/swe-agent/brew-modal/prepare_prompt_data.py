#!/usr/bin/env python3
"""Normalize SWE-style prompt JSONL metadata for Brew OH-Core Modal rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_NAMES = {
    "r2e-gym": "r2e-gym",
    "swegym": "gym",
    "custom": "custom",
}


def _infer_instance_id(metadata: dict[str, Any], row: dict[str, Any]) -> str:
    for source in (metadata, row):
        instance_id = source.get("instance_id")
        if instance_id:
            return str(instance_id)

    repo = metadata.get("repo_name") or row.get("repo_name") or metadata.get("repo") or row.get("repo")
    commit = (
        metadata.get("commit_hash")
        or row.get("commit_hash")
        or metadata.get("base_commit")
        or row.get("base_commit")
    )
    if repo and commit:
        return f"{repo}__{commit}"

    raise ValueError("row is missing instance_id and repo/commit metadata")


def _infer_prompt(metadata: dict[str, Any], row: dict[str, Any]) -> str:
    for source in (row, metadata):
        for key in ("prompt", "problem_statement", "issue", "description"):
            value = source.get(key)
            if value:
                return str(value)
    return ""


def normalize_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or row)
    metadata["instance_id"] = _infer_instance_id(metadata, row)
    metadata["dataset"] = args.dataset_name or DATASET_NAMES[args.dataset_kind]
    metadata["split"] = metadata.get("split") or args.split
    metadata["runner"] = args.runner
    metadata["runner_entrypoint"] = args.runner_entrypoint
    metadata["env_type"] = args.env_type
    metadata["task_type"] = "swe"
    metadata["max_iterations"] = args.max_iterations
    metadata["task_timeout"] = args.task_timeout
    metadata["step_timeout"] = args.step_timeout
    metadata["eval_timeout"] = args.eval_timeout
    metadata["env_timeout"] = args.env_timeout
    metadata["create_timeout"] = args.create_timeout

    return {
        "prompt": _infer_prompt(metadata, row),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Miles/raw JSONL prompt data.")
    parser.add_argument("--output", required=True, help="Output normalized JSONL prompt data.")
    parser.add_argument(
        "--dataset-kind",
        choices=sorted(DATASET_NAMES),
        default="r2e-gym",
        help="Metadata preset to apply.",
    )
    parser.add_argument("--dataset-name", default="", help="Override metadata.dataset.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--runner", default="oh-core")
    parser.add_argument("--runner-entrypoint", default="run_oh_core")
    parser.add_argument("--env-type", default="modal")
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--step-timeout", type=int, default=180)
    parser.add_argument("--eval-timeout", type=int, default=600)
    parser.add_argument("--env-timeout", type=int, default=180)
    parser.add_argument("--create-timeout", type=int, default=600)
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if args.limit and count >= args.limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            normalized = normalize_row(row, args)
            fout.write(json.dumps(normalized, ensure_ascii=True) + "\n")
            count += 1

    print(f"normalized {count} rows: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
