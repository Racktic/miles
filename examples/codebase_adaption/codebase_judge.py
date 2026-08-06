"""Async LLM judge for the ACT exploration reward (memory-delta signal).

Scores the semantic delta M_{k-1} -> M_k of the carry-forward memory: did trial k's ACT
work yield knowledge worth keeping (new codebase facts, corrected beliefs, concrete leads
for future trials)? The four dimensions are EQUALLY weighted:
explore_score = sum(4 dims, each 0..2) / 8 in [0, 1].

Ported from examples/alchemy/alchemy_judge.py with three deliberate changes:
  - rubric rewritten for SWE/codebase semantics (alchemy's was potion-specific);
  - EVERY failure path logs a [codebase_judge] line (alchemy degraded silently, which once
    hid a multi-step judge outage — see examples/alchemy/notes/deepseek_outage_20260706.md);
  - the in-process cache is size-capped so week-long runs do not grow it unboundedly.

Self-contained on purpose (no imports from the rest of the example) so ray rollout workers
pull no extra deps. Hard contract: any failure (missing key, timeout, HTTP error, parse
error) returns None so the caller degrades gracefully and training never blocks or crashes.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import requests

SYSTEM_PROMPT = """You are judging whether the agent's latest bug-fixing trial led to meaningful learning progress about the codebase it is working in.

You will be given:
1. The previous memory before one trial.
2. The updated memory after that trial.

The memory is the agent's only carry-over between trials: it later solves more issues in the same and other repositories with nothing but this text as prior knowledge.

Do NOT judge writing style or verbosity.
Do NOT reward a longer memory unless it adds substantive new knowledge.
Do NOT reward restating the previous memory in different words.
Do NOT reward generic software advice such as "read the error message carefully" or "run the tests" unless it is tied to a concrete file, module, command, or failure pattern.
Do NOT require the update to be correct with respect to ground truth; judge only whether the updated memory shows useful learning progress compared with the previous memory.
Do NOT give top scores for facts that only describe the issue just solved (its file, its bug, its fix). Recording the latest issue is the baseline behavior, not an achievement; top scores are reserved for knowledge reusable on future, different issues.
PENALIZE (score 0 on non_redundant_change and new_discoveries) updates that discard most of the previous memory's distinct knowledge and replace it with notes about only the latest issue, unless the discarded content was itself redundant.

Reward updates that:
- add new discoveries about the repository (structure, key modules, APIs, invariants), recurring bug patterns, test/tooling usage, or debugging strategies that worked;
- correct previous wrong, uncertain, or overconfident beliefs about the codebase or workflow;
- create concrete, checkable leads for future trials (specific files, functions, commands, or failure signatures to look at);
- differ from the previous memory in a meaningful, non-redundant way while keeping still-useful prior knowledge.

Score the update on four dimensions, each from 0 to 2:

1. new_discoveries:
0 = no new knowledge about the codebase or workflow
1 = new facts specific to the issue just solved (its file, its bug, its fix) — the default for any completed trial
2 = new knowledge reusable BEYOND the issue just solved: repository structure or conventions, a bug pattern seen across multiple issues, test/tooling infrastructure, or a debugging strategy that generalizes

2. error_correction:
0 = no correction
1 = clarifies or weakly revises a prior belief
2 = clearly corrects a previous mistake or resolves important uncertainty

3. verification_targets:
0 = no new lead for future trials
1 = vague lead, or a lead only restating where the just-solved issue was
2 = concrete lead useful for FUTURE, DIFFERENT issues (e.g. "when symptom X, check file/function Y", a named recurring location, a reusable verification command)

4. non_redundant_change:
0 = mostly redundant or cosmetic, OR wiped previous distinct knowledge to describe only the latest issue
1 = some meaningful change with prior knowledge kept
2 = substantially different in a useful way while retaining still-useful prior knowledge

Return only valid JSON:
{
  "brief_reason": "one short sentence",
  "new_discoveries": int,
  "error_correction": int,
  "verification_targets": int,
  "non_redundant_change": int
}
"""

_DIMS = ("new_discoveries", "error_correction", "verification_targets", "non_redundant_change")

# Lazily-created so the semaphore binds to the rollout's running event loop (not import-time).
_SEM: asyncio.Semaphore | None = None
_CACHE: dict[int, dict] = {}          # hash((prev, cur)) -> judge result dict
_CACHE_MAX = 4096                     # bound memory growth over multi-day runs


def _env(name: str, default: str = "") -> str:
    # Ray's runtime_env forwards unset shell vars as EMPTY strings, so "set but empty"
    # must fall back to the default exactly like "unset".
    value = os.environ.get(name, "").strip()
    return value or default


def judge_cfg() -> dict:
    return {
        "api_key": _env("CODEBASE_JUDGE_API_KEY") or _env("OPENAI_API_KEY") or _env("DEEPSEEK_API_KEY"),
        "api_base": _env("CODEBASE_JUDGE_API_BASE", "https://api.openai.com/v1"),
        "model": _env("CODEBASE_JUDGE_MODEL", "gpt-5-mini"),
        "timeout": float(_env("CODEBASE_JUDGE_TIMEOUT", "60")),
        "concurrency": max(1, int(_env("CODEBASE_JUDGE_CONCURRENCY", "64"))),
    }


def _sem(concurrency: int) -> asyncio.Semaphore:
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(concurrency)
    return _SEM


def _build_user_prompt(prev: str, cur: str) -> str:
    p = (prev or "").strip() or "(empty memory before the first trial)"
    c = (cur or "").strip() or "(empty updated memory)"
    return f"Previous memory M_(k-1):\n{p}\n\nUpdated memory M_k:\n{c}\n"


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", text.strip()).strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"could not parse JSON from response: {text[:200]}")


def _judge_result(parsed: dict) -> dict:
    """Clamp the 4 dims to [0,2]; equal-weighted explore_score in [0,1] plus a short reason."""
    dims = {k: max(0, min(2, int(parsed.get(k, 0)))) for k in _DIMS}
    return {
        **dims,
        "explore_score": sum(dims.values()) / 8.0,
        "brief_reason": str(parsed.get("brief_reason", ""))[:300],
    }


def _post_once(cfg: dict, prev: str, cur: str) -> dict:
    payload = {
        "model": cfg["model"],
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(prev, cur)},
        ],
    }
    if cfg["model"].startswith(("gpt-5", "o1", "o3", "o4")):
        # Reasoning models reject temperature!=1 and max_tokens; the completion cap must also
        # cover reasoning tokens, so keep it well above the ~200-token JSON answer.
        payload["max_completion_tokens"] = 2048
        payload["reasoning_effort"] = "minimal"
    else:
        payload["temperature"] = 0.0
        payload["max_tokens"] = 512
    resp = requests.post(
        cfg["api_base"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        json=payload,
        timeout=cfg["timeout"],
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _judge_result(_extract_json(raw))


def _describe_failure(exc: Exception) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    body = ""
    if status is not None:
        try:
            body = f" body={exc.response.text[:120]!r}"
        except Exception:
            pass
    return f"{type(exc).__name__}: {exc}" + (f" (HTTP {status}{body})" if status is not None else "")


async def judge_explore(prev: str, cur: str) -> dict | None:
    """Judge the memory delta prev->cur. Returns {4 dims, explore_score, brief_reason} or None
    on ANY failure (caller degrades; every failure is logged so outages are greppable).

    Bounded by a process-wide semaphore; identical (prev, cur) pairs are cached; the blocking
    HTTP request runs in a worker thread so the rollout event loop is never blocked. One retry.
    """
    cfg = judge_cfg()
    if not cfg["api_key"]:
        print("[codebase_judge] WARNING no API key (CODEBASE_JUDGE_API_KEY/OPENAI_API_KEY) -> None", flush=True)
        return None
    key = hash((prev or "", cur or ""))
    if key in _CACHE:
        return _CACHE[key]
    async with _sem(cfg["concurrency"]):
        for attempt in range(2):
            try:
                res = await asyncio.to_thread(_post_once, cfg, prev or "", cur or "")
                if len(_CACHE) >= _CACHE_MAX:
                    _CACHE.clear()
                _CACHE[key] = res
                return res
            except Exception as exc:
                print(
                    f"[codebase_judge] attempt {attempt + 1}/2 failed: {_describe_failure(exc)}",
                    flush=True,
                )
                if attempt == 1:
                    return None
                await asyncio.sleep(1.0)
    return None
