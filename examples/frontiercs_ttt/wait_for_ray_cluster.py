#!/usr/bin/env python3
"""Wait until a Ray cluster has enough homogeneous GPU nodes for training."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClusterSnapshot:
    alive_nodes: int
    eligible_nodes: int
    total_gpus: float


def cluster_snapshot(
    nodes: list[dict[str, Any]],
    cluster_resources: dict[str, float],
    minimum_gpus_per_node: float,
) -> ClusterSnapshot:
    alive = [node for node in nodes if node.get("Alive")]
    eligible = [
        node
        for node in alive
        if float((node.get("Resources") or {}).get("GPU", 0.0))
        >= minimum_gpus_per_node
    ]
    return ClusterSnapshot(
        alive_nodes=len(alive),
        eligible_nodes=len(eligible),
        total_gpus=float(cluster_resources.get("GPU", 0.0)),
    )


def missing_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if not os.path.exists(path)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--expected-gpus", type=float, required=True)
    parser.add_argument("--minimum-gpus-per-node", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--required-path", action="append", default=[])
    args = parser.parse_args()
    if args.expected_nodes < 1:
        parser.error("--expected-nodes must be at least 1")
    if args.expected_gpus <= 0 or args.minimum_gpus_per_node <= 0:
        parser.error("GPU requirements must be positive")
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        parser.error("timeouts must be positive")
    return args


def main() -> int:
    args = parse_args()
    import ray

    ray.init(address=args.address, ignore_reinit_error=True, logging_level="ERROR")
    deadline = time.monotonic() + args.timeout_seconds
    previous: ClusterSnapshot | None = None
    try:
        while True:
            snapshot = cluster_snapshot(
                ray.nodes(), ray.cluster_resources(), args.minimum_gpus_per_node
            )
            if snapshot != previous:
                print(
                    "[frontiercs-ray] "
                    f"alive_nodes={snapshot.alive_nodes}/{args.expected_nodes} "
                    f"gpu_nodes={snapshot.eligible_nodes}/{args.expected_nodes} "
                    f"gpus={snapshot.total_gpus:g}/{args.expected_gpus:g}",
                    flush=True,
                )
                previous = snapshot
            ready = (
                snapshot.eligible_nodes >= args.expected_nodes
                and snapshot.total_gpus >= args.expected_gpus
            )
            if ready:
                if args.required_path:
                    from ray.util.scheduling_strategies import (
                        NodeAffinitySchedulingStrategy,
                    )

                    check_paths = ray.remote(num_cpus=0)(missing_paths)
                    eligible_nodes = [
                        node
                        for node in ray.nodes()
                        if node.get("Alive")
                        and float((node.get("Resources") or {}).get("GPU", 0.0))
                        >= args.minimum_gpus_per_node
                    ]
                    checks = []
                    for node in eligible_nodes:
                        checks.append(
                            (
                                str(node.get("NodeManagerAddress") or node.get("NodeID")),
                                check_paths.options(
                                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                                        node_id=node["NodeID"], soft=False
                                    )
                                ).remote(args.required_path),
                            )
                        )
                    failed = [
                        (node_name, missing)
                        for (node_name, _), missing in zip(
                            checks, ray.get([ref for _, ref in checks])
                        )
                        if missing
                    ]
                    if failed:
                        for node_name, paths in failed:
                            print(
                                f"[frontiercs-ray] node {node_name} is missing: "
                                + ", ".join(paths),
                                flush=True,
                            )
                        return 1
                print("[frontiercs-ray] cluster and shared paths are ready", flush=True)
                return 0
            if time.monotonic() >= deadline:
                print(
                    "[frontiercs-ray] timed out waiting for the requested "
                    "nodes and GPUs",
                    flush=True,
                )
                return 1
            time.sleep(args.poll_seconds)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
