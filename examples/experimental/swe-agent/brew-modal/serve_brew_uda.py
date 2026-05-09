#!/usr/bin/env python3
"""Expose Brew runners through a Miles UDA-compatible `/run` API.

The script intentionally reuses Brew's RunRequest model and runner registry.
Brew must be importable, for example by adding the Brew repository to PYTHONPATH.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI

from brew.core.models import RunRequest
from brew.core.runners import (
    build_runner_params,
    load_runner_callable,
    resolve_request_runner,
)


LOG_LEVEL = os.environ.get("BREW_UDA_LOG_LEVEL", "INFO").upper()
OUTPUT_ROOT = Path(
    os.environ.get("BREW_UDA_OUTPUT_DIR", "results/brew_uda_server")
).expanduser()
CONCURRENCY = int(os.environ.get("BREW_UDA_CONCURRENCY", "1"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("brew_uda_server")

app = FastAPI(
    title="brew UDA-compatible server",
    description="Compatibility adapter for Miles SWE-agent rollout.",
    version="0.1.0",
)
_semaphore = asyncio.Semaphore(CONCURRENCY)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "brew_uda_server",
        "output_root": str(OUTPUT_ROOT.resolve()),
        "concurrency": CONCURRENCY,
    }


@app.post("/run")
async def run(body: dict[str, Any]) -> dict[str, Any]:
    request = _parse_request(body)
    output_dir = _build_output_dir(request)

    logger.info(
        "[%s] start runner=%s task=%s env=%s output_dir=%s",
        request.instance_id,
        request.runner or request.agent,
        request.task,
        request.env_type,
        output_dir,
    )

    async with _semaphore:
        try:
            result = await asyncio.to_thread(_run_request, request, output_dir)
        except Exception as exc:
            logger.exception("[%s] run failed: %s", request.instance_id, exc)
            response = _build_error_response(request, output_dir, exc)
        else:
            response = _build_success_response(request, output_dir, result)

    _save_response(output_dir, request.instance_id, response)
    logger.info(
        "[%s] done exit_status=%s reward=%s",
        request.instance_id,
        response.get("exit_status"),
        response.get("reward"),
    )
    return response


def _parse_request(body: dict[str, Any]) -> RunRequest:
    normalized = dict(body)
    if "task" not in normalized and "task_type" in normalized:
        normalized["task"] = normalized["task_type"]
    return RunRequest.model_validate(normalized)


def _build_output_dir(request: RunRequest) -> str:
    run_name = (request.run_name or "").strip()
    if not run_name:
        try:
            spec = resolve_request_runner(request)
            runner_path = Path(spec.task) / spec.agent
        except ValueError:
            runner_path = Path(request.task) / (request.runner or request.agent)
        model_name = request.model_name.strip() if request.model_name else "model"
        run_name = str(runner_path / model_name)
    return str(
        Path(OUTPUT_ROOT, run_name, request.instance_id, uuid4().hex[:8])
        .expanduser()
        .resolve()
    )


def _run_request(request: RunRequest, output_dir: str) -> Any:
    spec = resolve_request_runner(request)
    runner = load_runner_callable(spec.module, spec.entrypoint)
    params = build_runner_params(request, output_dir)
    return runner(**params)


def _build_success_response(
    request: RunRequest,
    output_dir: str,
    result: Any,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        result = {}

    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    error = result.get("error") or metadata.get("error")
    exit_status = str(result.get("exit_status") or ("error" if error else "completed"))
    reward = result.get("reward", 0.0)
    messages = result.get("messages") or []
    tools = result.get("tools") or []
    agent_metrics = result.get("agent_metrics") or {}

    response = {
        "instance_id": request.instance_id,
        "exit_status": exit_status,
        "output_dir": output_dir,
        "error": error,
        "reward": reward,
        "messages": messages,
        "tools": tools,
        "metadata": metadata,
        "agent_metrics": agent_metrics,
        "info": {
            "exit_status": exit_status,
            "output_dir": output_dir,
            "error": error,
        },
    }
    return response


def _build_error_response(
    request: RunRequest,
    output_dir: str,
    exc: Exception,
) -> dict[str, Any]:
    error = f"{exc}\n{traceback.format_exc()}"
    return {
        "instance_id": request.instance_id,
        "exit_status": "error",
        "output_dir": output_dir,
        "error": error,
        "reward": 0.0,
        "messages": [],
        "tools": [],
        "metadata": {"error": str(exc)},
        "agent_metrics": {},
        "info": {
            "exit_status": "error",
            "output_dir": output_dir,
            "error": str(exc),
        },
    }


def _save_response(output_dir: str, instance_id: str, response: dict[str, Any]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{instance_id}.uda_response.json", "w") as handle:
        json.dump(response, handle, indent=2)


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Brew UDA-compatible adapter.")
    parser.add_argument("--host", default=os.environ.get("BREW_UDA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("BREW_UDA_PORT", "11000")))
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
