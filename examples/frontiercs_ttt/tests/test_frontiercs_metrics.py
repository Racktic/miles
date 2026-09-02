from types import SimpleNamespace

import pytest

from examples.frontiercs_ttt import frontiercs_metrics
from miles.utils.types import Sample


SCORE_CURVES = (
    (0.0, 20.0, 40.0, 60.0),
    (0.0, 20.0, 40.0, 60.0),
    (0.0, 20.0, 40.0, 60.0),
    (0.0, 20.0, 0.0, 60.0),
    (0.0, 20.0, 0.0, 60.0),
    (0.0, 0.0, 0.0, 0.0),
)


EXPECTED_KEYS = {
    *(f"score/current_mean_r{round_index}" for round_index in range(4)),
    *(f"score/positive_frac_r{round_index}" for round_index in range(4)),
    *(f"score/best_mean_r{round_index}" for round_index in range(4)),
    *(f"act/executed_frac_r{round_index}" for round_index in range(4)),
    "act/executed_frac",
    "act/compile_error_frac",
    "act/invalid_submission_frac",
    "act/length_stop_frac",
    "write/length_stop_frac",
    "sample_length/act_mean",
    "sample_length/write_mean",
    "diagnostics/nonempty_frac",
    "memory/changed_frac",
    "memory/empty_frac",
    *(f"memory_length/after_r{round_index}_mean" for round_index in range(3)),
    "training_signal/write_reward_mean",
    "training_signal/grpo_zero_std_group_frac",
    "training_signal/act_advantage_abs_mean",
    "training_signal/write_advantage_abs_mean",
}


def _args():
    return SimpleNamespace(
        frontiercs_memory_rounds=4,
        frontiercs_candidates_per_problem=1,
        frontiercs_act_advantage_mode="temporal_problem_relative",
        frontiercs_write_advantage_mode="direct",
        frontiercs_write_advantage_scale=1.0,
        frontiercs_act_explore_beta=0.0,
        grpo_std_normalization=True,
        wandb_always_use_train_step=False,
    )


def _episode_samples():
    samples = []
    for membership_index, scores in enumerate(SCORE_CURVES):
        episode_index = membership_index // 3
        problem_index = membership_index % 3
        for round_index, score in enumerate(scores):
            executed = score > 0.0
            compile_error = not executed and membership_index % 2 == 0
            invalid_submission = not executed and not compile_error
            samples.append(
                Sample(
                    response_length=10 + round_index,
                    reward=score / 100.0,
                    status=(
                        Sample.Status.TRUNCATED
                        if round_index == 0
                        else Sample.Status.COMPLETED
                    ),
                    metadata={
                        "phase": "act",
                        "group_id": f"episode-{episode_index}",
                        "problem_id": f"problem-{problem_index}",
                        "memory_round": round_index,
                        "score_0_100": score,
                        "executed": executed,
                        "compile_error": compile_error,
                        "invalid_submission": invalid_submission,
                        "has_diagnostics": membership_index == 0 and executed,
                    },
                )
            )

    for episode_index, memory_lengths in enumerate(((100, 200, 300), (110, 210, 310))):
        for produced_round, (memory_tokens, reward) in enumerate(
            zip(memory_lengths, (0.1, -0.05, 0.2), strict=True)
        ):
            samples.append(
                Sample(
                    response_length=20,
                    reward=reward,
                    status=(
                        Sample.Status.TRUNCATED
                        if produced_round == 2
                        else Sample.Status.COMPLETED
                    ),
                    metadata={
                        "phase": "write",
                        "group_id": f"episode-{episode_index}",
                        "produced_round": produced_round,
                        "memory_tokens": memory_tokens,
                        "memory_changed": (episode_index + produced_round) % 2 == 0,
                        "memory_empty": episode_index == 0 and produced_round == 1,
                    },
                )
            )
    return samples


@pytest.fixture(autouse=True)
def _clear_frontiercs_environment(monkeypatch):
    for name in (
        "FRONTIERCS_MEMORY_ROUNDS",
        "FRONTIERCS_CANDIDATES_PER_PROBLEM",
        "FRONTIERCS_ACT_ADVANTAGE_MODE",
        "FRONTIERCS_WRITE_ADVANTAGE_MODE",
        "FRONTIERCS_WRITE_ADVANTAGE_SCALE",
        "FRONTIERCS_ACT_EXPLORE_BETA",
    ):
        monkeypatch.delenv(name, raising=False)


def test_complete_episode_metrics_have_exact_keys_and_values():
    metrics = frontiercs_metrics.compute_frontiercs_metrics(
        _args(), [_episode_samples()[:15], _episode_samples()[15:]]
    )

    assert set(metrics) == EXPECTED_KEYS
    assert metrics["score/current_mean_r0"] == 0.0
    assert metrics["score/current_mean_r1"] == pytest.approx(100.0 / 6.0)
    assert metrics["score/current_mean_r2"] == pytest.approx(20.0)
    assert metrics["score/current_mean_r3"] == pytest.approx(50.0)
    assert metrics["score/positive_frac_r0"] == 0.0
    assert metrics["score/positive_frac_r1"] == pytest.approx(5.0 / 6.0)
    assert metrics["score/positive_frac_r2"] == pytest.approx(0.5)
    assert metrics["score/positive_frac_r3"] == pytest.approx(5.0 / 6.0)
    assert metrics["score/best_mean_r0"] == 0.0
    assert metrics["score/best_mean_r1"] == pytest.approx(100.0 / 6.0)
    assert metrics["score/best_mean_r2"] == pytest.approx(80.0 / 3.0)
    assert metrics["score/best_mean_r3"] == pytest.approx(50.0)
    assert metrics["act/executed_frac_r0"] == 0.0
    assert metrics["act/executed_frac_r1"] == pytest.approx(5.0 / 6.0)
    assert metrics["act/executed_frac_r2"] == pytest.approx(0.5)
    assert metrics["act/executed_frac_r3"] == pytest.approx(5.0 / 6.0)

    assert metrics["act/executed_frac"] == pytest.approx(13.0 / 24.0)
    assert metrics["act/compile_error_frac"] == pytest.approx(4.0 / 24.0)
    assert metrics["act/invalid_submission_frac"] == pytest.approx(7.0 / 24.0)
    assert metrics["act/length_stop_frac"] == pytest.approx(6.0 / 24.0)
    assert metrics["write/length_stop_frac"] == pytest.approx(1.0 / 3.0)
    assert metrics["sample_length/act_mean"] == pytest.approx(11.5)
    assert metrics["sample_length/write_mean"] == pytest.approx(20.0)
    assert metrics["diagnostics/nonempty_frac"] == pytest.approx(3.0 / 24.0)
    assert metrics["memory/changed_frac"] == pytest.approx(0.5)
    assert metrics["memory/empty_frac"] == pytest.approx(1.0 / 6.0)

    assert metrics["memory_length/after_r0_mean"] == pytest.approx(105.0)
    assert metrics["memory_length/after_r1_mean"] == pytest.approx(205.0)
    assert metrics["memory_length/after_r2_mean"] == pytest.approx(305.0)
    assert metrics["training_signal/write_reward_mean"] == pytest.approx(1.0 / 12.0)
    assert metrics["training_signal/grpo_zero_std_group_frac"] == pytest.approx(1.0 / 6.0)
    assert metrics["training_signal/act_advantage_abs_mean"] == pytest.approx(
        0.6229983, abs=1e-5
    )
    assert metrics["training_signal/write_advantage_abs_mean"] == pytest.approx(
        7.0 / 60.0
    )


def test_logger_emits_one_numeric_row_and_suppresses_default_sample_metrics(monkeypatch):
    logged = []
    monkeypatch.setattr(
        frontiercs_metrics.tracking_utils,
        "log",
        lambda args, metrics, step_key: logged.append((metrics, step_key)),
    )

    handled = frontiercs_metrics.log_rollout_data(
        rollout_id=7,
        args=_args(),
        samples=[_episode_samples()],
        rollout_extra_metrics={"rollout/generation_time": 3.5},
        rollout_time=9.0,
    )

    assert handled is True
    assert len(logged) == 1
    metrics, step_key = logged[0]
    assert step_key == "rollout/step"
    assert metrics["rollout/step"] == 7
    assert metrics["rollout/generation_time"] == 3.5
    assert EXPECTED_KEYS.issubset(metrics)
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    assert not any("response" in key or "memory_text" in key or "code" in key for key in metrics)


def test_act_only_metrics_keep_memory_generation_observable():
    samples = [
        sample
        for sample in _episode_samples()
        if (sample.metadata or {}).get("phase") == "act"
    ]
    for sample in samples:
        metadata = sample.metadata or {}
        round_index = int(metadata["memory_round"])
        metadata.update(
            {
                "memory_generated_after_round": round_index < 3,
                "memory_terminal_after_round": False,
                "memory_tokens_after_round": 100 + round_index,
                "memory_changed_after_round": True,
                "memory_empty_after_round": False,
                "memory_response_tokens_after_round": 20 + round_index,
                "memory_finish_reason_after_round": "stop",
            }
        )
        sample.metadata = metadata

    metrics = frontiercs_metrics.compute_frontiercs_metrics(_args(), samples)

    assert metrics["memory/changed_frac"] == 1.0
    assert metrics["memory/empty_frac"] == 0.0
    assert metrics["memory_length/after_r0_mean"] == 100.0
    assert metrics["memory_length/after_r1_mean"] == 101.0
    assert metrics["memory_length/after_r2_mean"] == 102.0
    assert metrics["sample_length/write_mean"] == 21.0
    assert metrics["training_signal/write_reward_mean"] == 0.0
    assert metrics["training_signal/write_advantage_abs_mean"] == 0.0
