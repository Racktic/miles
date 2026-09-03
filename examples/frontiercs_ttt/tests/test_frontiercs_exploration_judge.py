from examples.frontiercs_ttt.frontiercs_exploration_judge import (
    JUDGE_VERSION,
    SYSTEM_PROMPT,
    _build_user_prompt,
    _enforce_semantic_invariants,
    _extract_json,
    _judge_result,
)


def test_memory_pair_is_the_only_user_prompt_payload():
    prompt = _build_user_prompt("old fact", "new fact")
    assert prompt == (
        "Previous memory M_(k-1):\nold fact\n\n"
        "Updated memory M_k:\nnew fact\n"
    )
    assert "problem" not in prompt.lower()
    assert "candidate" not in prompt.lower()
    assert "diagnostic" not in prompt.lower()


def test_empty_memory_labels_match_codebase_adaptation_protocol():
    assert _build_user_prompt("", "") == (
        "Previous memory M_(k-1):\n(empty memory before the first trial)\n\n"
        "Updated memory M_k:\n(empty updated memory)\n"
    )


def test_v4_prompt_allows_specific_testable_unverified_actions():
    assert JUDGE_VERSION == "v4-specific-testable"
    assert (
        "A proposed action need not have been tested, but it should be specific "
        "and testable; generic suggestions do not count."
    ) in SYSTEM_PROMPT


def test_zero_one_two_dimensions_are_clamped_and_normalized():
    result = _judge_result(
        {
            "brief_reason": "useful update",
            "new_discoveries": 2,
            "error_correction": 1,
            "actionable_knowledge": 7,
            "high_level_abstraction": -3,
        }
    )
    assert result == {
        "brief_reason": "useful update",
        "new_discoveries": 2,
        "error_correction": 1,
        "actionable_knowledge": 2,
        "high_level_abstraction": 0,
        "explore_score": 5 / 8,
    }


def test_json_extraction_accepts_fenced_response():
    parsed = _extract_json(
        '```json\n{"new_discoveries": 2, "error_correction": 1}\n```'
    )
    assert parsed["new_discoveries"] == 2
    assert parsed["error_correction"] == 1


def test_empty_previous_memory_cannot_receive_error_correction_credit():
    result = _enforce_semantic_invariants(
        {
            "new_discoveries": 2,
            "error_correction": 2,
            "actionable_knowledge": 1,
            "high_level_abstraction": 1,
            "explore_score": 0.75,
            "brief_reason": "judge violated the empty-memory rule",
        },
        "",
    )
    assert result["error_correction"] == 0
    assert result["explore_score"] == 0.5
