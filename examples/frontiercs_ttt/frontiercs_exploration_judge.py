"""LLM judge for Frontier-CS memory-delta exploration reward.

The semantic input is deliberately limited to the memory before and after one
round. Four dimensions are scored on the same 0/1/2 scale used by the
codebase-adaptation exploration judge, then averaged into ``explore_score`` in
[0, 1]. No problem statement, candidate code, task score, or diagnostics are
shown to this judge.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from typing import Any

import requests


SYSTEM_PROMPT = """You are judging whether one round produced meaningful learning progress in a shared memory for solving related algorithmic optimization problems.

You will be given exactly two inputs: the memory before the round and the memory after the round. Judge only their semantic delta. Retained or rephrased content earns no new credit, and length or writing quality is irrelevant. You do not see the problems, code, feedback, or ground truth, so use only evidence made explicit in the memories.

Static task facts, objectives, constraints, output formats, source code, scores without a lesson, and unsupported strategy suggestions are not discoveries by themselves. Diagnostic printing is not directly rewarded; knowledge obtained from a measurement or a probe that can resolve uncertainty is valuable. Claims need not be ground-truth correct, but top scores require visible evidence or a clear evidence-to-decision link.

Score four dimensions independently with integers 0, 1, or 2. Do not give both new_discoveries and actionable_knowledge a 2 merely because the update contains many concrete bullets: the former measures information gained, while the latter measures the useful policy derived from that information.

1. new_discoveries — What new evidence or information gain appears?
0 = No new observation or information gain; only restatement, static task facts, bookkeeping, implementation details, or an unsupported recommendation.
1 = A concrete local observation, failure signature, measurement, or evidence-motivated hypothesis is added, but it comes from a single outcome, remains tentative, or does not strongly distinguish what is true.
2 = The delta demonstrates meaningful information gain through a before/after intervention, repeated observations establishing a pattern, a newly exposed hidden constraint, or evidence that distinguishes competing hypotheses and materially redirects later search.

2. error_correction — How strongly does the update revise the previous memory itself?
0 = No specific prior belief is revised. This MUST be 0 when the previous memory is empty. Fixing a candidate/program or correcting a claim first introduced in the updated memory does not count.
1 = A belief visible in the previous memory is narrowed, qualified, or partially revised, but the evidence is limited or the earlier conclusion is not decisively replaced.
2 = A consequential prior belief or uncertainty is clearly overturned or resolved by a concrete observation or contradiction, changing what later attempts should believe or do.

3. actionable_knowledge — What new evidence-linked policy can guide a later attempt?
A proposed action need not have been tested, but it should be specific and testable; generic suggestions do not count.
0 = No new action guidance, or only generic advice, static facts, retrospective description, or a recommendation with no concrete target.
1 = A specific next action, diagnostic probe, or implementation change is proposed, but it is speculative, narrowly local, not clearly derived from new evidence, or missing an expected readout or consequence.
2 = The delta converts new evidence into an operational policy: it identifies the condition or scope, the action or probe to take, and the expected observation or consequence that will guide the next decision. A list of implementation details or parameter values alone is insufficient.

4. high_level_abstraction — How broadly does the new knowledge transfer?
0 = It is tied to one exact instance, answer, code listing, isolated parameter, or problem-specific trick, or merely uses broad wording.
1 = It gives a reusable principle for closely related problems or algorithms and identifies the shared scope or mechanism.
2 = It identifies a common mechanism, invariant, trade-off, or evidence-to-decision principle that can guide materially different optimization problems, including when it applies and what it changes.

Calibration examples:
- "Single-vertex moves failed, so larger moves may help" is normally discovery 1 and actionability 1: it is a useful local observation and hypothesis, not a demonstrated policy.
- "Reducing the internal deadline from 1.8s to 0.85s changed the score from 40 to 98.95 and eliminated no-output failures across 30 cases; therefore stop search by 0.85–0.90s and print the best-so-far" can merit discovery 2 and actionability 2 because it contains a measured intervention and an operational rule.

If the memories are semantically unchanged or the updated memory is empty or purely cosmetic, assign 0 on all dimensions. If the update discards substantial useful prior knowledge without explaining why it became invalid, do not award top-level discovery, actionability, or abstraction. Treat 1 as the normal score for useful but limited progress and reserve 2 for the stronger thresholds above.

In brief_reason, identify the strongest actual delta and the evidence supporting the assigned levels.

Return only valid JSON:
{
  "brief_reason": "one short sentence",
  "new_discoveries": int,
  "error_correction": int,
  "actionable_knowledge": int,
  "high_level_abstraction": int
}
"""

JUDGE_VERSION = "v4-specific-testable"
USER_PROMPT_TEMPLATE = (
    "Previous memory M_(k-1):\n{previous_memory}\n\n"
    "Updated memory M_k:\n{updated_memory}\n"
)
JUDGE_PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + "\0" + USER_PROMPT_TEMPLATE).encode("utf-8")
).hexdigest()


EXPLORE_DIMS = (
    "new_discoveries",
    "error_correction",
    "actionable_knowledge",
    "high_level_abstraction",
)

_SEM: asyncio.Semaphore | None = None
_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_MAX = 4096


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def judge_config() -> dict[str, Any]:
    return {
        "api_key": _env("FRONTIERCS_EXPLORE_JUDGE_API_KEY")
        or _env("OPENAI_API_KEY"),
        "api_base": _env(
            "FRONTIERCS_EXPLORE_JUDGE_API_BASE", "https://api.openai.com/v1"
        ),
        "model": _env("FRONTIERCS_EXPLORE_JUDGE_MODEL", "gpt-5-mini"),
        "timeout": float(_env("FRONTIERCS_EXPLORE_JUDGE_TIMEOUT", "60")),
        "concurrency": max(
            1, int(_env("FRONTIERCS_EXPLORE_JUDGE_CONCURRENCY", "64"))
        ),
    }


def _semaphore(concurrency: int) -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(concurrency)
    return _SEM


def _build_user_prompt(previous_memory: str, updated_memory: str) -> str:
    previous = (
        (previous_memory or "").strip() or "(empty memory before the first trial)"
    )
    updated = (updated_memory or "").strip() or "(empty updated memory)"
    return USER_PROMPT_TEMPLATE.format(
        previous_memory=previous,
        updated_memory=updated,
    )


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(
        r"```$", "", re.sub(r"^```(?:json)?", "", (text or "").strip()).strip()
    ).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"could not parse judge JSON: {text[:200]!r}") from None
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("exploration judge response must be a JSON object")
    return parsed


def _judge_result(parsed: dict[str, Any]) -> dict[str, Any]:
    dims = {
        name: max(0, min(2, int(parsed.get(name, 0))))
        for name in EXPLORE_DIMS
    }
    return {
        **dims,
        "explore_score": sum(dims.values()) / (2.0 * len(EXPLORE_DIMS)),
        "brief_reason": str(parsed.get("brief_reason") or "")[:300],
    }


def _enforce_semantic_invariants(
    result: dict[str, Any], previous_memory: str
) -> dict[str, Any]:
    """Apply rubric rules that should not depend on judge compliance."""
    normalized = dict(result)
    if not (previous_memory or "").strip():
        normalized["error_correction"] = 0
    normalized["explore_score"] = sum(
        int(normalized[name]) for name in EXPLORE_DIMS
    ) / (2.0 * len(EXPLORE_DIMS))
    return normalized


def _post_once(
    config: dict[str, Any], previous_memory: str, updated_memory: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config["model"],
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(previous_memory, updated_memory),
            },
        ],
    }
    if str(config["model"]).startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = 2048
        payload["reasoning_effort"] = "minimal"
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = 512
    response = requests.post(
        str(config["api_base"]).rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=float(config["timeout"]),
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return _enforce_semantic_invariants(
        _judge_result(_extract_json(content)), previous_memory
    )


def _failure_text(exc: Exception) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    suffix = f" HTTP={status}" if status is not None else ""
    return f"{type(exc).__name__}: {exc}{suffix}"


async def judge_memory_delta(
    previous_memory: str, updated_memory: str
) -> dict[str, Any] | None:
    """Return the 0/1/2 rubric and normalized score, or ``None`` on failure."""
    config = judge_config()
    if not config["api_key"]:
        print(
            "[frontiercs_exploration_judge] WARNING no "
            "FRONTIERCS_EXPLORE_JUDGE_API_KEY or OPENAI_API_KEY; no score",
            flush=True,
        )
        return None
    digest = hashlib.sha256(
        (
            str(config["model"])
            + f"\0frontiercs-memory-delta-{JUDGE_VERSION}\0"
            + JUDGE_PROMPT_HASH
            + "\0"
            + (previous_memory or "")
            + "\0"
            + (updated_memory or "")
        ).encode("utf-8")
    ).hexdigest()
    if digest in _CACHE:
        return dict(_CACHE[digest])
    async with _semaphore(int(config["concurrency"])):
        for attempt in range(2):
            try:
                result = await asyncio.to_thread(
                    _post_once, config, previous_memory or "", updated_memory or ""
                )
                if len(_CACHE) >= _CACHE_MAX:
                    _CACHE.clear()
                _CACHE[digest] = dict(result)
                return result
            except Exception as exc:
                print(
                    "[frontiercs_exploration_judge] "
                    f"attempt {attempt + 1}/2 failed: {_failure_text(exc)}",
                    flush=True,
                )
                if attempt == 1:
                    return None
                await asyncio.sleep(1.0)
    return None
