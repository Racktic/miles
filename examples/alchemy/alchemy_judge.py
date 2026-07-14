"""Async DeepSeek judge for the ACT exploration reward (memory-delta signal).

Scores the semantic delta M_{k-1} -> M_k (how much trial k explored) with the SAME prompt + parse that
`validate_memory_delta_judge.py` used for the offline validation (report notes/ACT_REWARD_DESIGN_REPORT.md
§14). The four dimensions are EQUALLY weighted: explore_score = sum(4 dims, each 0..2) / 8 in [0,1].

Self-contained on purpose (copied prompt/parse rather than importing the validator) so ray rollout workers
do not pull in argparse/statistics/oracle deps. The ONLY hard requirement: any failure (timeout, HTTP error,
parse error, missing key) returns None so the caller degrades gracefully and training never blocks/crashes.
"""
from __future__ import annotations

import asyncio
import json
import os
import re

import requests


# ---- copied verbatim from validate_memory_delta_judge.py (the validated prompt) -------------------------
SYSTEM_PROMPT = """You are judging whether the agent's latest trial led to meaningful exploration progress.

You will be given:
1. The previous memory before one trial.
2. The updated memory after that trial.

Do NOT judge writing style or verbosity.
Do NOT reward a longer memory unless it adds substantive exploration progress.
Do NOT reward restating the previous memory in different words.
Do NOT reward generic advice such as "try more potions" unless it names a concrete hypothesis or target.
Do NOT require the update to be correct with respect to hidden ground truth; judge only whether the updated memory shows useful exploration progress compared with the previous memory.

Reward updates that:
- add new discoveries about potion effects, stone transformations, reward-relevant patterns, or useful strategies;
- correct previous wrong, uncertain, or overconfident beliefs;
- create concrete hypotheses or verification targets for future trials;
- differ from the previous memory in a meaningful, non-redundant way.

Score the update on four dimensions, each from 0 to 2:

1. new_discoveries:
0 = no new discovery
1 = minor or tentative new discovery
2 = clear useful new discovery

2. error_correction:
0 = no correction
1 = clarifies or weakly revises a prior belief
2 = clearly corrects a previous mistake or resolves important uncertainty

3. verification_targets:
0 = no new exploration target
1 = vague or partial hypothesis to test
2 = concrete useful hypothesis or action target for future exploration

4. non_redundant_change:
0 = mostly redundant or cosmetic
1 = some meaningful change
2 = substantially different in a useful way

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
_CACHE: dict[int, float] = {}            # hash((prev, cur)) -> explore_score (dedup identical sibling memories)


def _cfg() -> dict:
    return {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "api_base": os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        "model": os.environ.get("ALCHEMY_JUDGE_MODEL", "deepseek-chat"),
        "timeout": float(os.environ.get("ALCHEMY_JUDGE_TIMEOUT", "60")),
        "concurrency": max(1, int(os.environ.get("ALCHEMY_JUDGE_CONCURRENCY", "64"))),
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
        return json.loads(cleaned[start:end + 1])
    raise ValueError(f"could not parse JSON from response: {text[:200]}")


def _judge_result(parsed: dict) -> dict:
    """Clamp the 4 dims to [0,2] and build the result: per-dim ints + equal-weighted explore_score + reason."""
    dims = {k: max(0, min(2, int(parsed.get(k, 0)))) for k in _DIMS}
    return {
        **dims,
        "explore_score": sum(dims.values()) / 8.0,     # equal 1:1:1:1 weighting
        "brief_reason": str(parsed.get("brief_reason", ""))[:300],
    }


def _post_once(cfg: dict, prev: str, cur: str) -> float:
    payload = {
        "model": cfg["model"],
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(prev, cur)},
        ],
    }
    resp = requests.post(
        cfg["api_base"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        json=payload,
        timeout=cfg["timeout"],
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _judge_result(_extract_json(raw))


async def judge_explore(prev: str, cur: str) -> dict | None:
    """Judge the memory delta prev->cur. Returns a dict with the 4 dims + explore_score + brief_reason,
    or None on ANY failure (caller degrades).

    Bounded by a process-wide semaphore; identical (prev, cur) pairs are cached. Runs the blocking HTTP
    request in a worker thread so the rollout event loop is never blocked. One retry on transient error.
    """
    cfg = _cfg()
    if not cfg["api_key"]:
        return None
    key = hash((prev or "", cur or ""))
    if key in _CACHE:
        return _CACHE[key]
    async with _sem(cfg["concurrency"]):
        for attempt in range(2):                          # 1 retry on transient failure
            try:
                res = await asyncio.to_thread(_post_once, cfg, prev or "", cur or "")
                _CACHE[key] = res
                return res
            except Exception:
                if attempt == 1:
                    return None                           # degrade: never raise into the rollout
                await asyncio.sleep(1.0)
    return None
