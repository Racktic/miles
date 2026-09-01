from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample

from examples.frontiercs_ttt import frontiercs_episode_rollout as episode
from qwen_eval.frontiercs_ttt.prompts import clean_memory
from examples.frontiercs_ttt import frontiercs_rollout as round_rollout
from qwen_eval.frontiercs_ttt.types import JudgeFeedback


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        return {"input_ids": [1] * max(1, (len(text) + 7) // 8)}


def _input(tmp_path):
    args = SimpleNamespace(
        n_samples_per_prompt=1,
        frontiercs_output_root=str(tmp_path),
        frontiercs_run_id="episode-unit",
        frontiercs_group_size=3,
        frontiercs_candidates_per_problem=1,
        frontiercs_memory_rounds=4,
        frontiercs_act_code_context="none",
        frontiercs_judge_url="http://judge.invalid",
        frontiercs_judge_timeout_seconds=1,
        frontiercs_judge_poll_seconds=0,
        frontiercs_diagnostics_chars_per_candidate=1000,
        frontiercs_act_max_new_tokens=100,
        frontiercs_write_max_new_tokens=50,
        frontiercs_writer_max_prompt_chars=120000,
        frontiercs_enable_thinking=True,
        frontiercs_write_reward_mode="delta",
        seq_length=32768,
        rollout_seed=7,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )
    state = SimpleNamespace(args=args, tokenizer=FakeTokenizer())
    seed = Sample(
        group_index=3,
        index=7,
        metadata={
            "group_id": "color_sat_episode",
            "problem_ids": ["174", "175", "176"],
        },
    )
    return GenerateFnInput(
        state=state,
        sample=seed,
        sampling_params={"temperature": 1.0},
        evaluation=False,
    )


def test_complete_episode_has_clean_delayed_credit_and_exact_replay(
    tmp_path, monkeypatch
):
    asyncio.run(_exercise_complete_episode(tmp_path, monkeypatch))


async def _exercise_complete_episode(tmp_path, monkeypatch):
    infer_calls = []
    judge_calls = []
    scores = [10.0, 20.0, 15.0, 30.0]

    async def fake_infer(url, prompt_ids, params):
        seed = int(params["sampling_seed"])
        local_seed = seed - 7 - 7 * 10_000_000
        round_index = local_seed // 100_000
        is_write = int(params["max_new_tokens"]) == 50
        infer_calls.append(("write" if is_write else "act", round_index, seed))
        if is_write:
            text = (
                f"<think>PRIVATE_WRITE_{round_index}</think>"
                f"### Cross-Problem Knowledge\n- memory-from-round-{round_index}"
            )
        else:
            text = (
                f"<think>PRIVATE_ACT_{round_index}</think>"
                "```cpp\n"
                f"// round={round_index}\n"
                "int main(){return 0;}\n"
                "```"
            )
        token_ids = [10, 11, 12]
        return text, token_ids, [-0.1] * len(token_ids), "stop", {
            "weight_version": "frozen-v0"
        }

    async def fake_evaluate(self, problem_id, code):
        round_index = int(code.split("round=", 1)[1].splitlines()[0])
        judge_calls.append((round_index, problem_id))
        return JudgeFeedback(
            status="done",
            score=scores[round_index],
            diagnostics=f"diagnostic-round-{round_index}-{problem_id}",
        )

    monkeypatch.setattr(round_rollout, "_infer", fake_infer)
    monkeypatch.setattr(episode, "_infer", fake_infer)
    monkeypatch.setattr(
        episode.FrontierAlgorithmJudge, "evaluate", fake_evaluate
    )

    first = await episode.generate_episode(_input(tmp_path))
    assert len(first.samples) == 15
    act_samples = [sample for sample in first.samples if sample.metadata["phase"] == "act"]
    write_samples = [
        sample for sample in first.samples if sample.metadata["phase"] == "write"
    ]
    assert len(act_samples) == 12
    assert len(write_samples) == 3
    assert all(sample.metadata["executed"] for sample in act_samples)
    assert not any(sample.metadata["compile_error"] for sample in act_samples)
    assert not any(sample.metadata["invalid_submission"] for sample in act_samples)
    assert all(sample.metadata["has_diagnostics"] for sample in act_samples)
    assert [sample.reward for sample in write_samples] == pytest.approx(
        [0.10, -0.05, 0.15]
    )
    assert all(sample.metadata["memory_tokens"] > 0 for sample in write_samples)
    assert {sample.metadata["episode_index"] for sample in first.samples} == {7}
    assert {
        tuple(sample.weight_versions or []) for sample in first.samples
    } == {("frozen-v0",)}

    episode_root = (
        tmp_path
        / "episode-unit"
        / "groups"
        / "color_sat_episode.episode-00000007"
    )
    manifest = json.loads((episode_root / "episode_manifest.json").read_text())
    assert manifest["training_unit"] == "complete_group_episode"
    assert manifest["optimizer_updates_inside_episode"] == 0
    committed = json.loads((episode_root / "episode.json").read_text())
    assert committed["act_sample_count"] == 12
    assert committed["write_sample_count"] == 3

    round_one_prompt = (
        episode_root
        / "round_001"
        / "problems"
        / "174"
        / "candidate_00"
        / "act_prompt.txt"
    ).read_text()
    assert "memory-from-round-0" in round_one_prompt
    assert "diagnostic-round-0" not in round_one_prompt
    write_prompt = (episode_root / "round_000" / "write_prompt.txt").read_text()
    assert "diagnostic-round-0-174" in write_prompt
    assert "PRIVATE_ACT_0" not in write_prompt

    calls_before_replay = (len(infer_calls), len(judge_calls))
    replay = await episode.generate_episode(_input(tmp_path))
    assert len(replay.samples) == 15
    assert (len(infer_calls), len(judge_calls)) == calls_before_replay


def test_clean_memory_never_falls_back_to_previous_state():
    assert clean_memory("") == ""
    assert clean_memory("   ") == ""
    assert clean_memory("```text\n\n```") == ""
    assert clean_memory("```bytes\nnew memory\n```") == "new memory"
    assert clean_memory("new memory") == "new memory"
