from types import SimpleNamespace

import pytest

from examples.frontiercs_ttt.frontiercs_advantage import reward_post_process


def _sample(reward, **metadata):
    return SimpleNamespace(reward=reward, metadata=metadata)


def test_raw_act_and_direct_delayed_write_do_not_share_a_normalization_group():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="raw",
        frontiercs_write_advantage_mode="direct",
        frontiercs_act_explore_beta=0.0,
        grpo_std_normalization=True,
    )
    samples = [
        _sample(0.25, phase="act", group_id="g", memory_round=1, problem_id="174"),
        _sample(-0.10, phase="write", group_id="g", produced_round=0, downstream_round=1),
    ]
    raw, advantages = reward_post_process(args, samples)
    assert raw == [0.25, -0.10]
    assert advantages == [0.25, -0.10]


def test_group_relative_is_within_problem_and_round():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="group_relative",
        frontiercs_write_advantage_mode="direct",
        frontiercs_act_explore_beta=0.0,
        grpo_std_normalization=True,
    )
    samples = [
        _sample(0.2, phase="act", group_id="g", memory_round=0, problem_id="174"),
        _sample(0.8, phase="act", group_id="g", memory_round=0, problem_id="174"),
    ]
    _, advantages = reward_post_process(args, samples)
    assert advantages[0] == pytest.approx(-(2**-0.5), abs=1e-5)
    assert advantages[1] == pytest.approx(2**-0.5, abs=1e-5)


def test_group_relative_rejects_k1_instead_of_silently_zeroing_act():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="group_relative",
        frontiercs_write_advantage_mode="direct",
        frontiercs_act_explore_beta=0.0,
        grpo_std_normalization=True,
    )
    with pytest.raises(ValueError, match="requires K>=2"):
        reward_post_process(
            args,
            [_sample(0.5, phase="act", group_id="g", memory_round=0, problem_id="174")],
        )


def test_write_advantage_scale_does_not_change_act_advantage():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="raw",
        frontiercs_write_advantage_mode="direct",
        frontiercs_write_advantage_scale=3.0,
        frontiercs_act_explore_beta=0.0,
        grpo_std_normalization=True,
    )
    samples = [
        _sample(0.4, phase="act", group_id="g", memory_round=1, problem_id="174"),
        _sample(-0.1, phase="write", group_id="g", produced_round=0, downstream_round=1),
    ]
    _, advantages = reward_post_process(args, samples)
    assert advantages == pytest.approx([0.4, -0.3])


def test_temporal_problem_relative_groups_four_rounds_and_excludes_write():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="temporal_problem_relative",
        frontiercs_write_advantage_mode="direct",
        frontiercs_memory_rounds=4,
        frontiercs_candidates_per_problem=1,
        frontiercs_act_explore_beta=0.0,
        grpo_std_normalization=True,
    )
    samples = [
        _sample(0.0, phase="act", group_id="g.episode-1", memory_round=0, problem_id="174"),
        _sample(0.0, phase="act", group_id="g.episode-1", memory_round=1, problem_id="174"),
        _sample(0.2, phase="act", group_id="g.episode-1", memory_round=2, problem_id="174"),
        _sample(0.6, phase="act", group_id="g.episode-1", memory_round=3, problem_id="174"),
        _sample(-0.1, phase="write", group_id="g.episode-1", produced_round=0, downstream_round=1),
    ]

    _, advantages = reward_post_process(args, samples)
    assert advantages[:4] == pytest.approx(
        [-2**-0.5, -2**-0.5, 0.0, 2**0.5], abs=1e-5
    )
    assert advantages[4] == pytest.approx(-0.1)
