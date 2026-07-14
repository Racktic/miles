#!/usr/bin/env python3
"""Offline validation for ACT exploration reward from memory deltas.

This script judges pairs (M_{k-1}, M_k) from saved Alchemy trajectories with
DeepSeek's OpenAI-compatible chat API. It is intentionally offline-only: the
user chooses the trajectory directories to evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUT = Path(__file__).resolve().parent / "logs" / "act_judge_validation"


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


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def infer_label(traj_dir: Path) -> str:
    if traj_dir.name == "traj":
        return traj_dir.parent.name
    return traj_dir.name


def parse_episode_filter(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def episode_num(path: Path) -> int | None:
    m = re.search(r"ep_?(\d+)\.json$", path.name)
    return int(m.group(1)) if m else None


def iter_pairs(
    traj_dir: Path,
    label: str,
    episode_filter: set[int] | None,
    max_episodes: int | None,
    skip_first_pair: bool,
) -> list[dict[str, Any]]:
    files = sorted(traj_dir.glob("ep*.json"), key=lambda p: (episode_num(p) is None, episode_num(p) or 0, p.name))
    if episode_filter is None and max_episodes is not None:
        files = files[:max_episodes]
    pairs: list[dict[str, Any]] = []
    for path in files:
        ep_num = episode_num(path)
        if episode_filter is not None and ep_num not in episode_filter:
            continue
        try:
            obj = json.loads(path.read_text())
        except Exception as exc:
            print(f"[warn] skip unreadable {path}: {exc}")
            continue
        summaries = obj.get("summaries") or []
        if not isinstance(summaries, list) or not summaries:
            continue
        scores = obj.get("per_trial_scores") or []
        start_k = 1 if skip_first_pair else 0
        for k, cur in enumerate(summaries[start_k:], start=start_k):
            prev = "" if k == 0 else str(summaries[k - 1])
            pairs.append(
                {
                    "run": label,
                    "traj_file": str(path),
                    "episode_file_index": ep_num,
                    "episode_index": obj.get("episode_index"),
                    "episode_id": obj.get("episode_id"),
                    "memory_window_size": obj.get("memory_window_size"),
                    "trial_index": k,
                    "previous_memory": prev,
                    "updated_memory": str(cur),
                    "prev_len": len(prev),
                    "updated_len": len(str(cur)),
                    "len_delta": len(str(cur)) - len(prev),
                    "trial_reward": scores[k] if k < len(scores) else None,
                    "next_trial_reward": scores[k + 1] if k + 1 < len(scores) else None,
                }
            )
    return pairs


def build_user_prompt(pair: dict[str, Any]) -> str:
    prev = pair["previous_memory"].strip() or "(empty memory before the first trial)"
    cur = pair["updated_memory"].strip() or "(empty updated memory)"
    return f"""Previous memory M_(k-1):
{prev}

Updated memory M_k:
{cur}
"""


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"could not parse JSON from response: {text[:300]}")


def call_deepseek(pair: dict[str, Any], model: str, api_base: str, api_key: str, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(pair)},
        ],
    }
    resp = requests.post(
        api_base.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["choices"][0]["message"]["content"]
    parsed = extract_json(raw)
    for key in ["new_discoveries", "error_correction", "verification_targets", "non_redundant_change"]:
        parsed[key] = int(parsed.get(key, 0))
        parsed[key] = max(0, min(2, parsed[key]))
    parsed["explore_score"] = (
        parsed["new_discoveries"]
        + parsed["error_correction"]
        + parsed["verification_targets"]
        + parsed["non_redundant_change"]
    ) / 8.0
    return {"raw_response": raw, "judge": parsed, "usage": data.get("usage", {})}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if "judge" in row:
            by_run.setdefault(row["run"], []).append(row)
    out: dict[str, Any] = {}
    fields = [
        "new_discoveries",
        "error_correction",
        "verification_targets",
        "non_redundant_change",
        "explore_score",
    ]
    for run, items in by_run.items():
        stats: dict[str, Any] = {"n": len(items)}
        for field in fields:
            vals = [float(x["judge"][field]) for x in items]
            stats[field] = {
                "mean": statistics.fmean(vals) if vals else None,
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
        stats["updated_len"] = {
            "mean": statistics.fmean([x["updated_len"] for x in items]) if items else None,
            "std": statistics.stdev([x["updated_len"] for x in items]) if len(items) > 1 else 0.0,
        }
        stats["len_delta"] = {
            "mean": statistics.fmean([x["len_delta"] for x in items]) if items else None,
            "std": statistics.stdev([x["len_delta"] for x in items]) if len(items) > 1 else 0.0,
        }
        out[run] = stats
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", action="append", required=True, help="Trajectory dir containing ep*.json. Repeatable.")
    ap.add_argument("--label", action="append", help="Optional label for each --traj-dir.")
    ap.add_argument("--episodes", default=None, help="Episode file indices, e.g. 0,1,5-8. Applies to all traj dirs.")
    ap.add_argument("--max-episodes-per-run", type=int, default=None,
                    help="Use the first N episode files per run after numeric filename sorting. Ignored when --episodes is set.")
    ap.add_argument("--skip-first-pair", action="store_true",
                    help="Skip empty-memory -> M0 and only judge adjacent non-empty memory pairs M0->M1, ...")
    ap.add_argument("--max-pairs-per-run", type=int, default=20)
    ap.add_argument("--sample", choices=["first", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--api-base", default=os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"))
    ap.add_argument("--env-file", default=str(DEFAULT_ENV))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true", help="Collect and write selected pairs without calling the API.")
    args = ap.parse_args()

    load_env(Path(args.env_file))
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Put it in .env or export it.")

    traj_dirs = [Path(p).expanduser().resolve() for p in args.traj_dir]
    labels = args.label or []
    if labels and len(labels) != len(traj_dirs):
        raise SystemExit("--label count must match --traj-dir count")
    labels = labels or [infer_label(p) for p in traj_dirs]

    episode_filter = parse_episode_filter(args.episodes)
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for traj_dir, label in zip(traj_dirs, labels):
        pairs = iter_pairs(traj_dir, label, episode_filter, args.max_episodes_per_run, args.skip_first_pair)
        if args.sample == "random":
            rng.shuffle(pairs)
        pairs = pairs[: args.max_pairs_per_run]
        print(f"[select] {label}: {len(pairs)} pairs from {traj_dir}")
        selected.extend(pairs)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "pairs.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"

    config_path.write_text(
        json.dumps(
            {
                "traj_dirs": [str(p) for p in traj_dirs],
                "labels": labels,
                "episodes": args.episodes,
                "max_episodes_per_run": args.max_episodes_per_run,
                "skip_first_pair": args.skip_first_pair,
                "max_pairs_per_run": args.max_pairs_per_run,
                "sample": args.sample,
                "seed": args.seed,
                "model": args.model,
                "api_base": args.api_base,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    rows: list[dict[str, Any]] = []
    with rows_path.open("w") as wf:
        for i, pair in enumerate(selected, 1):
            row = dict(pair)
            if args.dry_run:
                row["dry_run_prompt"] = build_user_prompt(pair)
            else:
                try:
                    row.update(call_deepseek(pair, args.model, args.api_base, api_key or "", args.timeout))
                except Exception as exc:
                    row["error"] = repr(exc)
                time.sleep(args.sleep)
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            wf.flush()
            print(f"[{i}/{len(selected)}] {row['run']} ep={row['episode_file_index']} trial={row['trial_index']} "
                  f"score={row.get('judge', {}).get('explore_score')} error={row.get('error')}")
            rows.append(row)

    summary_path.write_text(json.dumps(summarize(rows), indent=2, ensure_ascii=False))
    print(f"[done] rows: {rows_path}")
    print(f"[done] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
