"""TITO `/completions` capture: append-only assembly + masking + canary.

Drives `TaskState.add_completion_round` with synthetic ChatML token ids (no
model, no browser, no FastAPI) to assert that two DeepResearch rounds assemble
into a single training sequence whose loss_mask / logprobs cover EXACTLY the
generated completion content, and that the append-only canary fires when a
round's prompt does not extend the recorded prefix.

`tito_state.py` only depends on numpy, so we load it standalone by path. The
proxy package's `__init__` pulls in FastAPI + sglang, which are absent outside
a full miles environment; loading by path lets this test run anywhere numpy is
present. Run either with pytest (in the miles env) or directly:
``python3 test_tito_completion_capture.py``.
"""

from __future__ import annotations

import importlib.util
import logging
from contextlib import contextmanager
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "tito_state_under_test", Path(__file__).resolve().parent / "tito_state.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TaskState = _mod.TaskState

# Synthetic ChatML vocabulary (arbitrary distinct ints).
ASSISTANT_PREFIX = [10, 11]   # stands for "<|im_start|>assistant\n"
IM_END = 99
NEWLINE = 98
SYS_USER = [1, 2, 3, 4]       # system + user prompt
C1 = [20, 21, 22]             # round-1 raw generation (a <tool_call>…)
C1_LP = [-0.1, -0.2, -0.3]
TOOL_RESULT = [30, 31, 32, 33]  # injected tool response (user/tool turn)
C2 = [40, 41]                 # round-2 raw generation (final answer)
C2_LP = [-0.4, -0.5]


@contextmanager
def _capture_warnings():
    """Capture WARNING records emitted by tito_state's module logger."""
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _H(level=logging.WARNING)
    logger = _mod._logger
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


def _new_state():
    return TaskState(
        new_line_token_id=NEWLINE,
        assistant_prefix_ids=list(ASSISTANT_PREFIX),
        im_end_token_id=IM_END,
    )


def _round1_prompt() -> list[int]:
    # system+user + assistant generation prefix
    return SYS_USER + ASSISTANT_PREFIX


def _round2_prompt() -> list[int]:
    # round-1 recorded stream (asst_prefix + C1 + im_end + nl) re-rendered
    # identically, then the tool result, then the next generation prefix.
    return SYS_USER + ASSISTANT_PREFIX + C1 + [IM_END, NEWLINE] + TOOL_RESULT + ASSISTANT_PREFIX


def test_two_round_append_only_capture():
    state = _new_state()
    # Round 1: completion stopped at <tool_response> (no trailing im_end).
    state.add_completion_round(_round1_prompt(), list(C1), list(C1_LP))
    # Round 2: final answer; SGLang included the stop im_end → add_response strips it.
    state.add_completion_round(_round2_prompt(), C2 + [IM_END], C2_LP + [0.0])

    result = state.finalize()
    tokens = result["tokens"]
    loss_mask = result["loss_mask"]
    logprobs = result["rollout_log_probs"]
    resp_len = result["response_length"]

    # Full append-only trajectory: prompt + wrapped C1 + tool + wrapped C2.
    expected_tokens = (
        SYS_USER
        + ASSISTANT_PREFIX + C1 + [IM_END, NEWLINE]
        + TOOL_RESULT
        + ASSISTANT_PREFIX + C2 + [IM_END, NEWLINE]
    )
    assert tokens == expected_tokens

    # Lengths line up (response-length masks/logprobs).
    assert len(loss_mask) == len(logprobs) == resp_len
    assert resp_len == len(tokens) - len(SYS_USER)

    # loss_mask == 1 EXACTLY over the two completions' content tokens.
    response_tokens = tokens[len(SYS_USER):]
    masked_tokens = [t for t, m in zip(response_tokens, loss_mask) if m == 1]
    assert masked_tokens == C1 + C2
    assert sum(loss_mask) == len(C1) + len(C2)

    # logprob nonzero iff mask==1, and equals the captured per-token logprob.
    for m, lp in zip(loss_mask, logprobs):
        assert (lp != 0.0) == (m == 1)
    assert [lp for lp, m in zip(logprobs, loss_mask) if m == 1] == C1_LP + C2_LP


def test_canary_fires_when_prompt_diverges():
    state = _new_state()
    state.add_completion_round(_round1_prompt(), list(C1), list(C1_LP))

    # Round-2 prompt where the re-rendered C1 differs (77 instead of 22) — the
    # recorded prefix is NOT a prefix of this prompt. The canary must warn.
    diverged = SYS_USER + ASSISTANT_PREFIX + [20, 21, 77] + [IM_END, NEWLINE] + TOOL_RESULT + ASSISTANT_PREFIX
    with _capture_warnings() as records:
        state.add_completion_round(diverged, C2 + [IM_END], C2_LP + [0.0])
    assert any("[TITO-APPEND]" in rec.getMessage() for rec in records)


def test_canary_silent_when_append_only():
    state = _new_state()
    state.add_completion_round(_round1_prompt(), list(C1), list(C1_LP))
    with _capture_warnings() as records:
        state.add_completion_round(_round2_prompt(), C2 + [IM_END], C2_LP + [0.0])
    assert not any("[TITO-APPEND]" in rec.getMessage() for rec in records)


def _assert_valid_sample(rd: dict):
    """Mirror miles' Sample.validate() contract on a per-round dict."""
    assert rd["response_length"] >= 0
    assert len(rd["tokens"]) >= rd["response_length"]
    assert len(rd["loss_mask"]) == rd["response_length"]
    assert len(rd["rollout_log_probs"]) == rd["response_length"]


def test_per_round_fanout_is_self_consistent():
    """Strategy B: each round is an independent (prompt, completion) sample whose
    logprobs match its OWN prompt — faithful regardless of re-rendering."""
    state = _new_state()
    state.add_completion_round(_round1_prompt(), list(C1), list(C1_LP))
    # Round 2 prompt DIVERGES from round 1's raw tokens (re-rendering) — this is
    # exactly the case that breaks append-only, yet per-round capture is unaffected.
    diverged = SYS_USER + ASSISTANT_PREFIX + [20, 21, 77] + [IM_END, NEWLINE] + TOOL_RESULT + ASSISTANT_PREFIX
    state.add_completion_round(diverged, C2 + [IM_END], C2_LP + [0.0])

    rounds = state.finalize_rounds()
    assert len(rounds) == 2
    for rd in rounds:
        _assert_valid_sample(rd)

    # Round 1: tool-call turn, no im_end → every completion token trained.
    r1 = rounds[0]
    assert r1["tokens"] == _round1_prompt() + C1
    assert r1["response_length"] == len(C1)
    assert r1["loss_mask"] == [1, 1, 1]
    assert r1["rollout_log_probs"] == C1_LP

    # Round 2: final answer ending in im_end → im_end kept in tokens but masked.
    r2 = rounds[1]
    assert r2["tokens"] == diverged + C2 + [IM_END]
    assert r2["response_length"] == len(C2) + 1
    assert r2["loss_mask"] == [1, 1, 0]            # im_end masked
    assert r2["rollout_log_probs"] == C2_LP + [0.0]
    # The faithful prompt for round 2 is its OWN (diverged) prompt, not round 1's.
    assert r2["tokens"][: len(diverged)] == diverged


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
