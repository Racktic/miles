from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample

from examples.frontiercs_ttt import frontiercs_rollout as rollout
from qwen_eval.frontiercs_ttt.types import JudgeFeedback


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        # Keep the fake token count realistic enough for prompt-budget checks.
        return {"input_ids": [1] * max(1, (len(text) + 7) // 8)}


def _input(tmp_path, round_index):
    args = SimpleNamespace(
        n_samples_per_prompt=1,
        frontiercs_output_root=str(tmp_path),
        frontiercs_run_id="unit",
        frontiercs_group_size=2,
        frontiercs_candidates_per_problem=1,
        frontiercs_memory_rounds=2,
        frontiercs_act_code_context="none",
        frontiercs_judge_url="http://judge.invalid",
        frontiercs_judge_timeout_seconds=1,
        frontiercs_judge_poll_seconds=0,
        frontiercs_diagnostics_chars_per_candidate=1000,
        frontiercs_act_max_new_tokens=100,
        frontiercs_write_max_new_tokens=50,
        frontiercs_writer_max_prompt_chars=120000,
        frontiercs_enable_thinking=False,
        frontiercs_write_reward_mode="delta",
        seq_length=32768,
        rollout_seed=7,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        alchemy_current_rollout_id=round_index,
    )
    state = SimpleNamespace(args=args, tokenizer=FakeTokenizer())
    seed = Sample(
        group_index=round_index,
        index=round_index,
        metadata={"group_id": "color_scale", "problem_ids": ["174", "177"]},
    )
    return GenerateFnInput(
        state=state,
        sample=seed,
        sampling_params={"temperature": 1.0},
        evaluation=False,
    )


def test_two_round_delayed_write_and_idempotent_replay(tmp_path, monkeypatch):
    asyncio.run(_exercise_two_round_rollout(tmp_path, monkeypatch))


async def _exercise_two_round_rollout(tmp_path, monkeypatch):
    infer_calls = []
    judge_calls = []
    active_round = {"value": 0}

    async def fake_infer(url, prompt_ids, params):
        infer_calls.append((active_round["value"], params["sampling_seed"]))
        is_write = params["max_new_tokens"] == 50
        if is_write:
            text = (
                "### Cross-Problem Knowledge\n- verify graph invariants\n\n"
                "### Problem-Specific Verified Findings\n- round-zero evidence"
            )
        else:
            text = (
                "PRIVATE_ACT_EXPLANATION\n"
                f"```cpp\n// round={active_round['value']}\nint main(){{return 0;}}\n```"
            )
        ids = [10, 11, 12]
        return text, ids, [-0.1] * len(ids), "stop", {"weight_version": str(active_round["value"])}

    async def fake_evaluate(self, problem_id, code):
        judge_calls.append((active_round["value"], problem_id, code))
        assert f"round={active_round['value']}" in code
        score = 20.0 if active_round["value"] == 0 else 40.0
        return JudgeFeedback(
            status="done",
            score=score,
            diagnostics=f"real-diagnostic-round-{active_round['value']}",
        )

    monkeypatch.setattr(rollout, "_infer", fake_infer)
    monkeypatch.setattr(rollout.FrontierAlgorithmJudge, "evaluate", fake_evaluate)

    first = await rollout.generate(_input(tmp_path, 0))
    assert len(first.samples) == 2
    assert all((sample.metadata or {})["phase"] == "act" for sample in first.samples)

    run_root = tmp_path / "unit"
    write_prompt = (
        run_root / "groups" / "color_scale" / "round_000" / "write_prompt.txt"
    ).read_text()
    assert "real-diagnostic-round-0" in write_prompt
    assert "int main()" in write_prompt
    assert "PRIVATE_ACT_EXPLANATION" not in write_prompt

    active_round["value"] = 1
    second = await rollout.generate(_input(tmp_path, 1))
    assert len(second.samples) == 3
    write_samples = [sample for sample in second.samples if sample.metadata["phase"] == "write"]
    assert len(write_samples) == 1
    assert write_samples[0].reward == pytest.approx(0.2)
    assert write_samples[0].metadata["produced_round"] == 0
    assert write_samples[0].metadata["downstream_round"] == 1

    round_one_prompt = (
        run_root
        / "groups"
        / "color_scale"
        / "round_001"
        / "problems"
        / "174"
        / "candidate_00"
        / "act_prompt.txt"
    ).read_text()
    assert "round-zero evidence" in round_one_prompt
    assert "real-diagnostic-round-0" not in round_one_prompt

    calls_before_retry = (len(infer_calls), len(judge_calls))
    replay = await rollout.generate(_input(tmp_path, 1))
    assert len(replay.samples) == 3
    assert (len(infer_calls), len(judge_calls)) == calls_before_retry
    committed = json.loads(
        (
            run_root / "groups" / "color_scale" / "round_001" / "round.json"
        ).read_text()
    )
    assert committed["state_after"]["next_round"] == 2
