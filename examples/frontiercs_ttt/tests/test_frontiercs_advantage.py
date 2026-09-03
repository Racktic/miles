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


def test_exploration_is_normalized_within_one_episode_problem_across_rounds():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="raw",
        frontiercs_write_advantage_mode="direct",
        frontiercs_memory_rounds=4,
        frontiercs_candidates_per_problem=1,
        frontiercs_act_explore_beta=2.0,
        grpo_std_normalization=True,
    )
    first_scores = [0.0, 0.25, 0.5, 1.0]
    samples = [
        _sample(
            0.0,
            phase="act",
            group_id="episode-a",
            memory_round=round_index,
            problem_id="174",
            explore_score=score,
        )
        for round_index, score in enumerate(first_scores)
    ] + [
        _sample(
            0.0,
            phase="act",
            group_id="episode-b",
            memory_round=round_index,
            problem_id="174",
            explore_score=0.5,
        )
        for round_index in range(4)
    ]

    _, advantages = reward_post_process(args, samples)
    mean = sum(first_scores) / len(first_scores)
    std = (
        sum((value - mean) ** 2 for value in first_scores)
        / (len(first_scores) - 1)
    ) ** 0.5 + 1e-6
    assert advantages[:4] == pytest.approx(
        [2.0 * (value - mean) / std for value in first_scores]
    )
    assert advantages[4:] == pytest.approx([0.0] * 4)


def test_exploration_never_mixes_different_problems_in_one_episode():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="raw",
        frontiercs_write_advantage_mode="direct",
        frontiercs_memory_rounds=4,
        frontiercs_candidates_per_problem=1,
        frontiercs_act_explore_beta=1.0,
        grpo_std_normalization=True,
    )
    varying = [0.0, 0.25, 0.5, 1.0]
    samples = [
        _sample(
            0.0,
            phase="act",
            group_id="episode-a",
            memory_round=round_index,
            problem_id="174",
            explore_score=score,
        )
        for round_index, score in enumerate(varying)
    ] + [
        _sample(
            0.0,
            phase="act",
            group_id="episode-a",
            memory_round=round_index,
            problem_id="175",
            explore_score=0.5,
        )
        for round_index in range(4)
    ]

    _, advantages = reward_post_process(args, samples)
    mean = sum(varying) / len(varying)
    std = (
        sum((value - mean) ** 2 for value in varying) / (len(varying) - 1)
    ) ** 0.5 + 1e-6
    assert advantages[:4] == pytest.approx(
        [(value - mean) / std for value in varying]
    )
    assert advantages[4:] == pytest.approx([0.0] * 4)


def test_exploration_group_contains_all_round_candidate_members_for_one_problem():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="raw",
        frontiercs_write_advantage_mode="direct",
        frontiercs_memory_rounds=4,
        frontiercs_candidates_per_problem=2,
        frontiercs_act_explore_beta=1.0,
        grpo_std_normalization=True,
    )
    round_scores = [0.0, 0.25, 0.5, 1.0]
    samples = [
        _sample(
            0.0,
            phase="act",
            group_id="episode-a",
            memory_round=round_index,
            candidate_index=candidate_index,
            problem_id="174",
            explore_score=round_scores[round_index],
        )
        for round_index in range(4)
        for candidate_index in range(2)
    ]

    _, advantages = reward_post_process(args, samples)
    repeated_scores = [value for value in round_scores for _ in range(2)]
    mean = sum(repeated_scores) / len(repeated_scores)
    std = (
        sum((value - mean) ** 2 for value in repeated_scores)
        / (len(repeated_scores) - 1)
    ) ** 0.5 + 1e-6
    assert advantages == pytest.approx(
        [(value - mean) / std for value in repeated_scores]
    )


def test_incomplete_exploration_episode_problem_is_not_partially_shaped():
    args = SimpleNamespace(
        frontiercs_act_advantage_mode="raw",
        frontiercs_write_advantage_mode="direct",
        frontiercs_memory_rounds=4,
        frontiercs_candidates_per_problem=1,
        frontiercs_act_explore_beta=0.3,
        grpo_std_normalization=True,
    )
    samples = [
        _sample(
            0.0,
            phase="act",
            group_id="episode-a",
            memory_round=round_index,
            problem_id="174",
            **({"explore_score": float(round_index)} if round_index < 3 else {}),
        )
        for round_index in range(4)
    ]

    _, advantages = reward_post_process(args, samples)
    assert advantages == pytest.approx([0.0] * 4)
