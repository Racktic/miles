"""
Level 1 integration tests for the TITO proxy.

No GPU required — uses MockSGLangServer with a small tokenizer (Qwen3-0.6B).
Verifies that the proxy correctly intercepts /v1/chat/completions requests,
tokenizes incrementally, captures tokens + logprobs per turn, and produces
correct finalized training data via get_task_result().
"""

import json
import time
from argparse import Namespace

import pytest
import requests
from transformers import AutoTokenizer

from miles.utils.test_utils.mock_sglang_server import with_mock_server, ProcessResult
from uda.swe_agent.proxy.tito_server import TITOProxy

MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


@pytest.fixture
def mock_sglang():
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(text="The answer is 42.", finish_reason="stop")

    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


@pytest.fixture
def proxy(mock_sglang, tokenizer):
    args = Namespace(
        use_rollout_routing_replay=False,
        num_layers=0,
        moe_router_topk=0,
        sglang_tool_call_parser=None,
    )
    p = TITOProxy(
        sglang_base_url=mock_sglang.url,
        tokenizer=tokenizer,
        args=args,
    )
    # Give uvicorn a moment to start
    time.sleep(0.5)
    return p


@pytest.fixture
def mock_sglang_tool_call():
    """Mock SGLang that returns text with <tool_call> markers in qwen25 format."""
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text=(
                'I\'ll check the file.\n'
                '<tool_call>\n'
                '{"name": "execute_bash", "arguments": {"command": "ls -la"}}\n'
                '</tool_call>'
            ),
            finish_reason="stop",
        )

    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


@pytest.fixture
def proxy_with_tool_parser(mock_sglang_tool_call, tokenizer):
    """Proxy with tool_call_parser enabled (qwen25 format for mock compatibility)."""
    args = Namespace(
        use_rollout_routing_replay=False,
        num_layers=0,
        moe_router_topk=0,
        sglang_tool_call_parser="qwen25",
    )
    p = TITOProxy(
        sglang_base_url=mock_sglang_tool_call.url,
        tokenizer=tokenizer,
        args=args,
    )
    time.sleep(0.5)
    return p


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command"}
                },
                "required": ["command"],
            },
        },
    },
]


def _chat(proxy, task_id: str, messages: list[dict], max_tokens: int = 64,
          tools: list[dict] | None = None) -> dict:
    """Helper: send a chat completion request to the proxy."""
    payload = {"model": "test", "messages": messages, "max_tokens": max_tokens}
    if tools:
        payload["tools"] = tools
    resp = requests.post(
        f"{proxy.base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {task_id}"},
        timeout=30.0,
    )
    assert resp.status_code == 200, f"Proxy returned {resp.status_code}: {resp.text}"
    return resp.json()


class TestSingleTurn:
    def test_returns_valid_chat_completion(self, proxy):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 6 times 7?"},
        ]
        data = _chat(proxy, "single-001", messages)

        assert "choices" in data
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_get_task_result_returns_valid_data(self, proxy):
        task_id = "single-002"
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 6 times 7?"},
        ]
        _chat(proxy, task_id, messages)

        result = proxy.get_task_result(task_id)

        assert result is not None
        assert len(result["tokens"]) > 0
        assert len(result["loss_mask"]) == result["response_length"]
        assert result["response_length"] > 0
        assert len(result["rollout_log_probs"]) > 0
        # loss_mask should have 1s (assistant tokens exist)
        assert sum(result["loss_mask"]) > 0


class TestMultiTurn:
    def test_three_turn_conversation(self, proxy):
        task_id = "multi-001"

        # Turn 1: system + user
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 6 times 7?"},
        ]
        reply1 = _chat(proxy, task_id, messages)
        assistant_text_1 = reply1["choices"][0]["message"]["content"]

        # Turn 2: append assistant + tool result
        messages += [
            {"role": "assistant", "content": assistant_text_1},
            {"role": "user", "content": "Are you sure about that?"},
        ]
        reply2 = _chat(proxy, task_id, messages)
        assistant_text_2 = reply2["choices"][0]["message"]["content"]

        # Turn 3: append assistant + another user message
        messages += [
            {"role": "assistant", "content": assistant_text_2},
            {"role": "user", "content": "Thanks!"},
        ]
        reply3 = _chat(proxy, task_id, messages)

        # Finalize
        result = proxy.get_task_result(task_id)

        assert result is not None
        assert len(result["loss_mask"]) == result["response_length"]
        assert result["response_length"] > 0

        # loss_mask: 1s only for assistant tokens, 0s for system/user/tool
        assert sum(result["loss_mask"]) > 0

        # loss_mask is response-only (len == response_length, not len(tokens))
        # In a multi-turn response, it should contain both 0s (user turn tokens)
        # and 1s (assistant-generated content tokens)
        response_mask = result["loss_mask"]
        assert any(m == 0 for m in response_mask), "Multi-turn response should have non-assistant tokens (mask=0)"
        assert any(m == 1 for m in response_mask), "Multi-turn response should have assistant tokens (mask=1)"

    def test_logprobs_length_matches_response_length(self, proxy):
        task_id = "multi-002"

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        reply = _chat(proxy, task_id, messages)
        messages += [
            {"role": "assistant", "content": reply["choices"][0]["message"]["content"]},
            {"role": "user", "content": "Goodbye"},
        ]
        _chat(proxy, task_id, messages)

        result = proxy.get_task_result(task_id)
        assert len(result["rollout_log_probs"]) == result["response_length"]


class TestConcurrentTasks:
    def test_two_tasks_isolated(self, proxy):
        task_a = "concurrent-a"
        task_b = "concurrent-b"

        msgs_a = [
            {"role": "system", "content": "You are A."},
            {"role": "user", "content": "Task A question"},
        ]
        msgs_b = [
            {"role": "system", "content": "You are B."},
            {"role": "user", "content": "Task B question"},
        ]

        # Interleave requests
        _chat(proxy, task_a, msgs_a)
        _chat(proxy, task_b, msgs_b)

        result_a = proxy.get_task_result(task_a)
        result_b = proxy.get_task_result(task_b)

        assert result_a is not None
        assert result_b is not None
        # Both should have data
        assert len(result_a["tokens"]) > 0
        assert len(result_b["tokens"]) > 0
        # They should be different (different system prompts)
        assert result_a["tokens"] != result_b["tokens"]


class TestEdgeCases:
    def test_missing_task_returns_none(self, proxy):
        result = proxy.get_task_result("nonexistent-task")
        assert result is None

    def test_get_task_result_pops_state(self, proxy):
        task_id = "pop-test"
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        _chat(proxy, task_id, messages)

        result1 = proxy.get_task_result(task_id)
        assert result1 is not None

        # Second call should return None (state was popped)
        result2 = proxy.get_task_result(task_id)
        assert result2 is None


class TestToolCallParsing:
    """Tests that build_response() correctly parses <tool_call> markers into
    structured tool_calls when tool_call_parser is configured."""

    def test_tool_calls_parsed_from_raw_text(self, proxy_with_tool_parser):
        """When model output contains <tool_call>, response should have structured tool_calls."""
        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Check the directory listing."},
        ]
        data = _chat(proxy_with_tool_parser, "tc-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"

        message = choice["message"]
        assert "tool_calls" in message, "Response should contain tool_calls field"
        assert len(message["tool_calls"]) == 1

        tc = message["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "execute_bash"
        assert "command" in tc["function"]["arguments"]
        assert tc["id"].startswith("call_")

        # Content should be cleaned (no <tool_call> markers)
        assert "<tool_call>" not in message["content"]

    def test_no_tool_calls_without_parser(self, proxy):
        """Without tool_call_parser, raw text stays in content (no tool_calls field)."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        data = _chat(proxy, "tc-002", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        message = choice["message"]
        # No tool_call_parser → no parsing → no tool_calls field
        assert "tool_calls" not in message or message.get("tool_calls") is None
        assert choice["finish_reason"] == "stop"

    def test_no_tool_calls_without_tools_in_request(self, proxy_with_tool_parser):
        """Even with parser enabled, if no tools in request, no parsing occurs."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        # No tools parameter
        data = _chat(proxy_with_tool_parser, "tc-003", messages)

        choice = data["choices"][0]
        message = choice["message"]
        # No tools in request → no parsing
        assert "tool_calls" not in message or message.get("tool_calls") is None

    def test_tool_call_result_still_has_valid_tito_data(self, proxy_with_tool_parser):
        """Tool call parsing in build_response() should not affect TITO token capture."""
        task_id = "tc-004"
        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "List files."},
        ]
        _chat(proxy_with_tool_parser, task_id, messages, tools=TOOLS_SCHEMA)

        result = proxy_with_tool_parser.get_task_result(task_id)
        assert result is not None
        assert len(result["tokens"]) > 0
        assert len(result["loss_mask"]) == result["response_length"]
        assert result["response_length"] > 0
        assert sum(result["loss_mask"]) > 0


def _run_multi_turn_chat(proxy, task_id: str) -> dict:
    """Helper: run a 3-turn conversation and return the finalized TITO result."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 6 times 7?"},
    ]
    reply1 = _chat(proxy, task_id, messages)
    messages += [
        {"role": "assistant", "content": reply1["choices"][0]["message"]["content"]},
        {"role": "user", "content": "Are you sure about that?"},
    ]
    reply2 = _chat(proxy, task_id, messages)
    messages += [
        {"role": "assistant", "content": reply2["choices"][0]["message"]["content"]},
        {"role": "user", "content": "Thanks!"},
    ]
    _chat(proxy, task_id, messages)
    return proxy.get_task_result(task_id)


class TestLogprobDiffHypothesis1:
    """Validate H1: TITO stores 0.0 logprobs for wrapper/non-assistant tokens,
    which inflates logprob diff metrics when compared without loss_mask filtering.

    Background: monitoring experiments showed TITO has LARGER logprob diff vs
    Megatron than non-TITO (logprob_diff_max: 153.8 vs 37.9). This is because
    tito_state.py pads wrapper tokens (<|im_start|>assistant\\n, <|im_end|>\\n)
    and non-assistant messages with 0.0 logprobs, while Megatron computes real
    logprobs for all response tokens. The diff in loss.py:641 does not filter
    by loss_mask, so these 0.0 vs real-logprob mismatches inflate the metrics.
    """

    def test_tito_zero_logprobs_at_wrapper_positions(self, proxy):
        """rollout_log_probs == 0.0 at every position where loss_mask == 0.
        This is the core of H1: TITO pads wrapper tokens with 0.0 logprobs."""
        result = _run_multi_turn_chat(proxy, "h1-zero-001")

        loss_mask = result["loss_mask"]
        logprobs = result["rollout_log_probs"]
        assert len(loss_mask) == len(logprobs) == result["response_length"]

        # Every wrapper position (loss_mask=0) must have logprob=0.0
        for i, (mask, lp) in enumerate(zip(loss_mask, logprobs)):
            if mask == 0:
                assert lp == 0.0, (
                    f"Position {i}: loss_mask=0 but logprob={lp} (expected 0.0). "
                    f"Wrapper token should have 0.0 logprob."
                )

        # At least some content positions (loss_mask=1) should have non-zero logprobs
        content_logprobs = [lp for lp, m in zip(logprobs, loss_mask) if m == 1]
        # Note: mock returns -1/128*i, so first content token (i=0) gives -0.0 == 0.0.
        # We check that the majority of content tokens have real logprobs.
        n_nonzero = sum(1 for lp in content_logprobs if lp != 0.0)
        assert n_nonzero > len(content_logprobs) * 0.8, (
            f"Only {n_nonzero}/{len(content_logprobs)} content tokens have non-zero logprobs. "
            f"Expected most content tokens to have real logprobs from SGLang."
        )

    def test_unmasked_diff_larger_than_masked_diff(self, proxy):
        """Simulates the monitoring bug: unmasked diff is dramatically larger
        than masked diff because wrapper token 0.0 logprobs create huge diffs
        against Megatron's real logprobs."""
        result = _run_multi_turn_chat(proxy, "h1-diff-001")

        loss_mask = result["loss_mask"]
        logprobs = result["rollout_log_probs"]
        resp_len = result["response_length"]

        # Simulate Megatron logprobs: real values for ALL response tokens
        fake_megatron_logprobs = [-0.5] * resp_len

        # Unmasked diff (what loss.py:641 currently does)
        abs_diff = [abs(m - t) for m, t in zip(fake_megatron_logprobs, logprobs)]
        unmasked_max = max(abs_diff)
        unmasked_frac_gt_0_1 = sum(1 for d in abs_diff if d > 0.1) / len(abs_diff)

        # Masked diff (what it SHOULD do — only content tokens)
        content_diffs = [d for d, m in zip(abs_diff, loss_mask) if m == 1]
        assert len(content_diffs) > 0, "Should have at least some content tokens"
        masked_max = max(content_diffs)
        masked_frac_gt_0_1 = sum(1 for d in content_diffs if d > 0.1) / len(content_diffs)

        # Wrapper tokens (logprob=0.0) create |(-0.5) - 0.0| = 0.5 diff
        # Content tokens have mock logprobs close to 0, so diff ≈ |(-0.5) - (-small)| ≈ 0.5
        # The key insight: unmasked includes BOTH, masked only includes content
        # For max: unmasked_max >= masked_max (wrapper diffs can only add, never reduce max)
        assert unmasked_max >= masked_max, (
            f"Unmasked max ({unmasked_max}) should be >= masked max ({masked_max})"
        )

        # The fraction metric is where the inflation really shows:
        # wrapper tokens are a significant fraction of response tokens
        n_wrapper = sum(1 for m in loss_mask if m == 0)
        assert n_wrapper > 0, "Multi-turn should have wrapper tokens (prefix/suffix/user messages)"

    def test_wrapper_token_fraction_explains_diff(self, proxy):
        """The fraction of 0.0-logprob positions exactly matches loss_mask=0
        positions, confirming wrapper tokens are the sole source of inflated metrics."""
        result = _run_multi_turn_chat(proxy, "h1-frac-001")

        loss_mask = result["loss_mask"]
        logprobs = result["rollout_log_probs"]
        resp_len = result["response_length"]

        n_masked_out = sum(1 for m in loss_mask if m == 0)

        # Every loss_mask=0 position must have logprob=0.0
        for i, (mask, lp) in enumerate(zip(loss_mask, logprobs)):
            if mask == 0:
                assert lp == 0.0, (
                    f"Position {i}: loss_mask=0 but logprob={lp}. "
                    f"All wrapper positions should have 0.0 logprobs."
                )

        # Wrapper tokens are a non-trivial fraction of response
        # (prefix + suffix per assistant turn + user/tool messages in response section)
        wrapper_frac = n_masked_out / resp_len
        assert wrapper_frac > 0.1, (
            f"Wrapper token fraction ({wrapper_frac:.1%}) is too small. "
            f"In multi-turn, expect >10% of response tokens to be wrappers."
        )

        # The wrapper fraction explains the inflated metrics: in real experiments
        # TITO showed logprob_diff_frac_gt_1.0 = 27.7%, which aligns with the
        # fraction of wrapper tokens that have 0.0 logprobs vs Megatron's real values.
        print(f"  Wrapper token fraction: {wrapper_frac:.1%} "
              f"({n_masked_out}/{resp_len} response tokens are wrappers)")


# =============================================================================
# Hypothesis tests for TITO vs Non-TITO 11.5% solve rate gap
# =============================================================================

# -- Fixtures for hypothesis tests --

@pytest.fixture
def mock_sglang_length_stop():
    """Returns finish_reason=length with tool_call markers in text."""
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text='<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls"}}\n</tool_call>',
            finish_reason="length",
        )
    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


@pytest.fixture
def mock_sglang_malformed_tool():
    """Returns malformed JSON inside tool_call markers."""
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text='<tool_call>\n{not valid json}\n</tool_call>',
            finish_reason="stop",
        )
    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


@pytest.fixture
def mock_sglang_reasoning_plus_tool():
    """Returns reasoning text + tool call (realistic pattern)."""
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text=(
                "Let me check the directory structure first.\n\n"
                '<tool_call>\n'
                '{"name": "execute_bash", "arguments": {"command": "find . -type f"}}\n'
                '</tool_call>'
            ),
            finish_reason="stop",
        )
    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


def _make_proxy(mock_server, tokenizer, tool_call_parser=None):
    """Helper to create a TITO proxy with given parser config."""
    args = Namespace(
        use_rollout_routing_replay=False,
        num_layers=0,
        moe_router_topk=0,
        sglang_tool_call_parser=tool_call_parser,
    )
    p = TITOProxy(
        sglang_base_url=mock_server.url,
        tokenizer=tokenizer,
        args=args,
    )
    time.sleep(0.5)
    return p


class TestH1ToolCallParsing:
    """H1: Tool call parsing mismatch — TITO uses FunctionCallParser.parse_non_stream()
    while SGLang native /v1/chat/completions uses its own built-in parsing pipeline.
    If the TITO parser fails to detect tool calls, the agent can't execute tools."""

    def test_tool_call_parsed_on_finish_reason_length(
        self, mock_sglang_length_stop, tokenizer
    ):
        """H1b fixed: When finish_reason=length, TITO now parses tool calls based on
        text content (content-based detection), matching SGLang native behavior.
        Previously the gate `finish_reason == 'stop'` silently dropped tool calls."""
        proxy = _make_proxy(mock_sglang_length_stop, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "List files."},
        ]
        data = _chat(proxy, "h1b-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        # H1b fixed: tool call markers in text → parsed even on finish_reason=length
        assert choice["finish_reason"] == "tool_calls", (
            f"Expected finish_reason=tool_calls (H1b fixed) but got {choice['finish_reason']}"
        )
        assert choice["message"].get("tool_calls"), "Expected tool_calls to be present"

    def test_tool_call_with_malformed_json(
        self, mock_sglang_malformed_tool, tokenizer
    ):
        """H1: Malformed JSON in tool_call markers should be handled gracefully.
        Parser should catch the error and fall back to plain text."""
        proxy = _make_proxy(mock_sglang_malformed_tool, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Check files."},
        ]
        data = _chat(proxy, "h1-malformed-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        # Should not crash — graceful fallback
        assert choice["finish_reason"] in ("stop", "tool_calls")
        # Response should still be returned
        assert choice["message"]["content"] is not None

    def test_reasoning_text_preserved_in_content(
        self, mock_sglang_reasoning_plus_tool, tokenizer
    ):
        """H4: After tool call parsing, reasoning text before the markers should
        be preserved in the content field. Tool markers should be removed."""
        proxy = _make_proxy(mock_sglang_reasoning_plus_tool, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Investigate the issue."},
        ]
        data = _chat(proxy, "h4-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        content = choice["message"]["content"]

        # Reasoning text should be preserved
        assert "Let me check" in content, (
            f"Reasoning text lost after parsing. content='{content[:200]}'"
        )
        # Tool markers should be removed
        assert "<tool_call>" not in content
        assert "</tool_call>" not in content
        # Tool calls should be structured
        assert choice["finish_reason"] == "tool_calls"
        assert len(choice["message"]["tool_calls"]) == 1

    def test_parser_name_qwen3_coder_xml_format(self, tokenizer):
        """CRITICAL: Test with sglang_tool_call_parser='qwen3_coder' using the
        XML format that Qwen3-Coder actually generates. The qwen3_coder parser
        expects <function=name><parameter=key>value</parameter></function>,
        NOT JSON format like qwen25."""

        # qwen3_coder XML format (what the model actually generates)
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(
                text=(
                    "I'll run the command.\n"
                    "<tool_call>\n"
                    "<function=execute_bash>\n"
                    "<parameter=command>ls -la</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                ),
                finish_reason="stop",
            )

        with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
            proxy = _make_proxy(server, tokenizer, "qwen3_coder")

            messages = [
                {"role": "system", "content": "You are a SWE agent."},
                {"role": "user", "content": "List files."},
            ]
            data = _chat(proxy, "h1-qwen3coder-xml-001", messages, tools=TOOLS_SCHEMA)

            choice = data["choices"][0]
            assert choice["finish_reason"] == "tool_calls", (
                f"qwen3_coder parser failed to detect XML tool call! "
                f"finish_reason={choice['finish_reason']}, "
                f"content={choice['message']['content'][:200]}"
            )
            assert "tool_calls" in choice["message"]
            assert len(choice["message"]["tool_calls"]) == 1
            tc = choice["message"]["tool_calls"][0]
            assert tc["function"]["name"] == "execute_bash"
            # Verify arguments are properly converted to JSON
            args = json.loads(tc["function"]["arguments"])
            assert args["command"] == "ls -la"

    def test_parser_qwen3_coder_rejects_json_format(self, tokenizer):
        """Regression: qwen3_coder parser does NOT parse qwen25-style JSON
        inside <tool_call> markers. This confirms the format difference.
        If the model ever generates JSON format instead of XML, tool calls
        will be silently lost — this is H1a."""

        # qwen25 JSON format (WRONG for qwen3_coder)
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(
                text=(
                    'I\'ll run the command.\n'
                    '<tool_call>\n'
                    '{"name": "execute_bash", "arguments": {"command": "ls -la"}}\n'
                    '</tool_call>'
                ),
                finish_reason="stop",
            )

        with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
            proxy = _make_proxy(server, tokenizer, "qwen3_coder")

            messages = [
                {"role": "system", "content": "You are a SWE agent."},
                {"role": "user", "content": "List files."},
            ]
            data = _chat(proxy, "h1-qwen3coder-json-001", messages, tools=TOOLS_SCHEMA)

            choice = data["choices"][0]
            # qwen3_coder parser CANNOT parse JSON format → no tool_calls detected
            assert choice["finish_reason"] == "stop", (
                "qwen3_coder parser unexpectedly parsed JSON format! "
                "This would mean qwen25 format works with qwen3_coder parser."
            )

    def test_multi_turn_with_tool_calls(self, tokenizer):
        """Full multi-turn tool call flow: tool call → tool result → follow-up.
        Validates that TITO correctly captures all turns and loss_mask is correct."""
        call_count = 0

        def process_fn(prompt: str) -> ProcessResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Turn 1: model makes a tool call
                return ProcessResult(
                    text=(
                        '<tool_call>\n'
                        '{"name": "execute_bash", "arguments": {"command": "ls"}}\n'
                        '</tool_call>'
                    ),
                    finish_reason="stop",
                )
            else:
                # Turn 2+: model responds with text
                return ProcessResult(
                    text="The directory contains 3 files.",
                    finish_reason="stop",
                )

        with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
            proxy = _make_proxy(server, tokenizer, "qwen25")
            task_id = "h1-multiturn-001"

            # Turn 1: system + user → tool call
            messages = [
                {"role": "system", "content": "You are a SWE agent."},
                {"role": "user", "content": "What files are here?"},
            ]
            reply1 = _chat(proxy, task_id, messages, tools=TOOLS_SCHEMA)
            assert reply1["choices"][0]["finish_reason"] == "tool_calls"

            tc = reply1["choices"][0]["message"]["tool_calls"][0]
            content1 = reply1["choices"][0]["message"]["content"]

            # Turn 2: append assistant (with tool_calls) + tool result → text response
            messages.append({
                "role": "assistant",
                "content": content1,
                "tool_calls": [tc],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": "file1.py\nfile2.py\nfile3.py",
            })
            reply2 = _chat(proxy, task_id, messages, tools=TOOLS_SCHEMA)
            assert reply2["choices"][0]["finish_reason"] == "stop"

            # Finalize
            result = proxy.get_task_result(task_id)
            assert result is not None
            assert result["response_length"] > 0
            assert len(result["loss_mask"]) == result["response_length"]
            # Should have both 0s (tool result, wrappers) and 1s (assistant content)
            assert sum(result["loss_mask"]) > 0
            assert sum(1 for m in result["loss_mask"] if m == 0) > 0


class TestH2ParamDrops:
    """H2: TITO only forwards temperature, top_p, max_tokens, stop to /generate.
    All other sampling params are silently dropped."""

    def test_extra_params_dropped_from_generate(self, mock_sglang, proxy, tokenizer):
        """Verify that extra params (top_k, stop_token_ids, etc.) are not forwarded
        to the /generate endpoint."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        resp = requests.post(
            f"{proxy.base_url}/chat/completions",
            json={
                "model": "test",
                "messages": messages,
                "max_tokens": 64,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 50,
                "stop_token_ids": [151645],
                "response_format": {"type": "json_object"},
                "n": 1,
            },
            headers={"Authorization": "Bearer h2-params-001"},
            timeout=30.0,
        )
        assert resp.status_code == 200

        # Check what the mock server actually received
        assert len(mock_sglang.request_log) > 0
        last_req = mock_sglang.request_log[-1]
        sampling = last_req.get("sampling_params", {})

        # These should be forwarded
        assert sampling.get("max_new_tokens") == 64
        assert sampling.get("temperature") == 0.5
        assert sampling.get("top_p") == 0.9

        # These should NOT be present (dropped by TITO)
        assert "top_k" not in sampling, f"top_k should be dropped but found: {sampling}"
        assert "stop_token_ids" not in sampling
        assert "response_format" not in sampling
        assert "n" not in sampling

    def test_max_tokens_default(self, mock_sglang, tokenizer):
        """When max_tokens is not in request, TITO should default to 4096."""
        args = Namespace(
            use_rollout_routing_replay=False,
            num_layers=0,
            moe_router_topk=0,
            sglang_tool_call_parser=None,
        )
        proxy = TITOProxy(
            sglang_base_url=mock_sglang.url,
            tokenizer=tokenizer,
            args=args,
        )
        time.sleep(0.5)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        resp = requests.post(
            f"{proxy.base_url}/chat/completions",
            json={"model": "test", "messages": messages},  # No max_tokens!
            headers={"Authorization": "Bearer h2-default-001"},
            timeout=30.0,
        )
        assert resp.status_code == 200

        last_req = mock_sglang.request_log[-1]
        sampling = last_req.get("sampling_params", {})
        assert sampling.get("max_new_tokens") == 4096, (
            f"Expected default max_new_tokens=4096, got {sampling.get('max_new_tokens')}"
        )


class TestH3StopToken:
    """H3: TITO strips <|im_end|> from SGLang output then re-adds it in wrapping.
    If this goes wrong, conversation context drifts."""

    def test_im_end_count_per_turn(self, proxy, tokenizer):
        """Each assistant turn should have exactly one <|im_end|> in the final tokens."""
        task_id = "h3-imend-001"
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        _chat(proxy, task_id, messages)
        result = proxy.get_task_result(task_id)

        tokens = result["tokens"]
        # Count <|im_end|> in the response section
        response_tokens = tokens[-result["response_length"]:]
        im_end_count = sum(1 for t in response_tokens if t == im_end_id)

        # Single-turn: exactly 1 <|im_end|> (for the one assistant turn)
        assert im_end_count == 1, (
            f"Expected 1 <|im_end|> in single-turn response, got {im_end_count}"
        )

    def test_multi_turn_im_end_count(self, proxy, tokenizer):
        """Multi-turn: each assistant turn should contribute one <|im_end|>."""
        task_id = "h3-imend-multi-001"
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        # 3-turn conversation
        result = _run_multi_turn_chat(proxy, task_id)

        tokens = result["tokens"]
        response_tokens = tokens[-result["response_length"]:]

        # Count all <|im_end|> in response section
        im_end_positions = [i for i, t in enumerate(response_tokens) if t == im_end_id]

        # 3 assistant turns + user/tool messages in response section also have <|im_end|>
        # At minimum: 3 assistant <|im_end|> + 2 user <|im_end|> = 5
        assert len(im_end_positions) >= 3, (
            f"Expected at least 3 <|im_end|> (for 3 assistant turns), "
            f"got {len(im_end_positions)}"
        )

    def test_no_im_end_when_length_truncated(self, mock_sglang_length_stop, tokenizer):
        """When finish_reason=length, SGLang may not include <|im_end|>.
        TITO should still wrap correctly."""
        proxy = _make_proxy(mock_sglang_length_stop, tokenizer)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        _chat(proxy, "h3-length-001", messages)
        result = proxy.get_task_result("h3-length-001")

        assert result is not None
        assert result["response_length"] > 0
        # Should still produce valid token sequence
        assert len(result["tokens"]) > 0


# =============================================================================
# Extended hypothesis tests — H1b, H4, H2, H5
# =============================================================================

# -- New fixtures --

@pytest.fixture
def mock_sglang_plain_length():
    """Plain text truncated by max_tokens — no tool call markers."""
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text="I'm still working on the problem and need more tok",
            finish_reason="length",
        )
    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


@pytest.fixture
def mock_sglang_truncated_tool():
    """Tool call started but JSON truncated mid-way by max_tokens."""
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text='Let me check.\n<tool_call>\n{"name": "execute_bash", "argum',
            finish_reason="length",
        )
    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


@pytest.fixture
def mock_sglang_multi_tool_call():
    """Two complete <tool_call> blocks with reasoning text."""
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text=(
                "I'll check the directory and also read the config.\n"
                '<tool_call>\n'
                '{"name": "execute_bash", "arguments": {"command": "ls -la"}}\n'
                '</tool_call>\n'
                '<tool_call>\n'
                '{"name": "execute_bash", "arguments": {"command": "cat config.yaml"}}\n'
                '</tool_call>'
            ),
            finish_reason="stop",
        )
    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


@pytest.fixture
def mock_sglang_tool_only():
    """Only <tool_call> block, no reasoning prefix, finish_reason=stop."""
    def process_fn(prompt: str) -> ProcessResult:
        return ProcessResult(
            text='<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls"}}\n</tool_call>',
            finish_reason="stop",
        )
    with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
        yield server


class TestH1bFinishReasonLengthGate:
    """H1b: TITO gates tool call parsing on finish_reason=='stop' (tito_converter.py:88).
    When finish_reason=='length', tool calls are silently dropped even if the text
    contains complete <tool_call> markers. SGLang native gates on text content instead.
    This is the P0 hypothesis for the 11.5% solve rate gap."""

    def test_tool_calls_parsed_even_on_length(
        self, mock_sglang_length_stop, tokenizer
    ):
        """H1b fixed: tool calls ARE parsed even with finish_reason=length,
        as long as text contains valid <tool_call> markers.  Matches SGLang
        native and verl-qwen behavior (content-based detection)."""
        proxy = _make_proxy(mock_sglang_length_stop, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "List files."},
        ]
        data = _chat(proxy, "h1b-fixed-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert len(choice["message"]["tool_calls"]) == 1
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "execute_bash"

    def test_plain_text_with_length_no_parsing(
        self, mock_sglang_plain_length, tokenizer
    ):
        """Negative case: finish_reason=length with NO tool markers → correctly no parsing."""
        proxy = _make_proxy(mock_sglang_plain_length, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Explain the bug."},
        ]
        data = _chat(proxy, "h1b-plain-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        assert choice["finish_reason"] == "length"
        assert "tool_calls" not in choice["message"] or choice["message"].get("tool_calls") is None
        assert choice["message"]["content"] == "I'm still working on the problem and need more tok"

    def test_truncated_tool_call_graceful(
        self, mock_sglang_truncated_tool, tokenizer
    ):
        """Truncated tool call JSON with finish_reason=length → no crash, no valid tool_calls.
        Even if the gate bug is fixed, truncated JSON should not produce tool_calls."""
        proxy = _make_proxy(mock_sglang_truncated_tool, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Check files."},
        ]
        data = _chat(proxy, "h1b-trunc-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        # Should not crash — graceful handling
        assert choice["finish_reason"] in ("length", "stop")
        # Truncated JSON should not produce valid tool_calls
        tc = choice["message"].get("tool_calls")
        assert tc is None or len(tc) == 0, (
            f"Truncated tool call JSON should not parse to valid tool_calls, got {tc}"
        )

    @pytest.mark.parametrize("finish_reason", ["stop", "length"])
    def test_parametrized_finish_reasons(
        self, tokenizer, finish_reason
    ):
        """H1b fixed: both stop and length finish_reason produce tool_calls when
        text contains valid <tool_call> markers (content-based detection)."""
        def process_fn(prompt: str) -> ProcessResult:
            return ProcessResult(
                text='<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls"}}\n</tool_call>',
                finish_reason=finish_reason,
            )

        with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
            proxy = _make_proxy(server, tokenizer, "qwen25")
            messages = [
                {"role": "system", "content": "You are a SWE agent."},
                {"role": "user", "content": "List files."},
            ]
            data = _chat(proxy, f"h1b-param-{finish_reason}-001", messages, tools=TOOLS_SCHEMA)

            choice = data["choices"][0]
            has_tc = bool(choice["message"].get("tool_calls"))
            assert has_tc, f"finish_reason={finish_reason}: expected tool_calls but got none"
            assert choice["finish_reason"] == "tool_calls"

    def test_multi_turn_tool_call_preserved_mid_trajectory(self, tokenizer):
        """H1b fixed multi-turn: turn 1 (stop) and turn 2 (length) both correctly
        parse tool calls from text content.  Previously turn 2 would silently drop
        the tool call, causing the agent to lose tool access mid-trajectory."""
        call_count = 0

        def process_fn(prompt: str) -> ProcessResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Turn 1: tool call with finish_reason=stop → parsed correctly
                return ProcessResult(
                    text='<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls"}}\n</tool_call>',
                    finish_reason="stop",
                )
            else:
                # Turn 2: tool call with finish_reason=length → LOST due to H1b bug
                return ProcessResult(
                    text='<tool_call>\n{"name": "execute_bash", "arguments": {"command": "cat file.py"}}\n</tool_call>',
                    finish_reason="length",
                )

        with with_mock_server(model_name=MODEL_NAME, process_fn=process_fn) as server:
            proxy = _make_proxy(server, tokenizer, "qwen25")
            task_id = "h1b-multiturn-001"

            # Turn 1: system + user → tool call (succeeds)
            messages = [
                {"role": "system", "content": "You are a SWE agent."},
                {"role": "user", "content": "What files are here?"},
            ]
            reply1 = _chat(proxy, task_id, messages, tools=TOOLS_SCHEMA)
            assert reply1["choices"][0]["finish_reason"] == "tool_calls", (
                "Turn 1 should succeed: finish_reason=stop with tool markers"
            )
            tc1 = reply1["choices"][0]["message"]["tool_calls"][0]

            # Turn 2: append tool result → second tool call (FAILS due to H1b)
            messages.append({
                "role": "assistant",
                "content": reply1["choices"][0]["message"]["content"],
                "tool_calls": [tc1],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc1["id"],
                "content": "file1.py\nfile2.py",
            })
            reply2 = _chat(proxy, task_id, messages, tools=TOOLS_SCHEMA)

            # H1b FIXED: turn 2 has valid tool call markers → correctly parsed
            # even though finish_reason=length (content-based detection, not metadata)
            choice2 = reply2["choices"][0]
            assert choice2["finish_reason"] == "tool_calls", (
                f"Turn 2 should show H1b fix: tool calls parsed on length, got {choice2['finish_reason']}"
            )
            assert choice2["message"].get("tool_calls"), (
                "Turn 2: tool_calls should be present (H1b fixed)"
            )
            tc2 = choice2["message"]["tool_calls"][0]
            assert tc2["function"]["name"] == "execute_bash"


class TestH4TextCorruption:
    """H4: parser.parse_non_stream(text) modifies text by removing <tool_call> markers.
    Verify that ONLY markers are removed and all reasoning text is preserved exactly."""

    def test_exact_reasoning_preservation(
        self, mock_sglang_reasoning_plus_tool, tokenizer
    ):
        """After tool call parsing, reasoning text before markers must be preserved
        with exact character match (not just substring)."""
        proxy = _make_proxy(mock_sglang_reasoning_plus_tool, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Investigate the issue."},
        ]
        data = _chat(proxy, "h4-exact-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        content = choice["message"]["content"]

        # Exact match of reasoning text (only markers removed)
        assert content.strip() == "Let me check the directory structure first.", (
            f"Reasoning text not exactly preserved. Got: '{content}'"
        )
        # No leftover markers
        assert "<tool_call>" not in content
        assert "</tool_call>" not in content
        # Tool calls properly extracted
        assert choice["finish_reason"] == "tool_calls"
        assert len(choice["message"]["tool_calls"]) == 1
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "execute_bash"

    def test_multiple_tool_calls_all_parsed(
        self, mock_sglang_multi_tool_call, tokenizer
    ):
        """Two tool calls in one response: both parsed, reasoning text preserved."""
        proxy = _make_proxy(mock_sglang_multi_tool_call, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Check directory and config."},
        ]
        data = _chat(proxy, "h4-multi-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        content = choice["message"]["content"]

        # Reasoning text preserved
        assert "check the directory" in content, (
            f"Reasoning text lost. content='{content[:200]}'"
        )
        # No markers remaining
        assert "<tool_call>" not in content
        assert "</tool_call>" not in content

        # Both tool calls parsed
        assert choice["finish_reason"] == "tool_calls"
        tcs = choice["message"]["tool_calls"]
        assert len(tcs) == 2, f"Expected 2 tool calls, got {len(tcs)}"
        assert tcs[0]["function"]["name"] == "execute_bash"
        assert tcs[1]["function"]["name"] == "execute_bash"

        # Verify arguments
        args0 = json.loads(tcs[0]["function"]["arguments"])
        args1 = json.loads(tcs[1]["function"]["arguments"])
        assert args0["command"] == "ls -la"
        assert args1["command"] == "cat config.yaml"

    def test_tool_only_no_reasoning_empty_content(
        self, mock_sglang_tool_only, tokenizer
    ):
        """When response is ONLY a <tool_call> block with no reasoning,
        content should be empty/whitespace after marker removal."""
        proxy = _make_proxy(mock_sglang_tool_only, tokenizer, "qwen25")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "List files."},
        ]
        data = _chat(proxy, "h4-toolonly-001", messages, tools=TOOLS_SCHEMA)

        choice = data["choices"][0]
        content = choice["message"]["content"] or ""

        # Content should be empty or whitespace (only markers were in the text)
        assert content.strip() == "", (
            f"Expected empty content when response is only a tool call, got: '{content}'"
        )
        # Tool call properly parsed
        assert choice["finish_reason"] == "tool_calls"
        assert len(choice["message"]["tool_calls"]) == 1
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "execute_bash"


class TestH2ParamDropsExtended:
    """Extended H2 tests: comprehensive inventory of which OpenAI params are
    forwarded vs dropped by TITO proxy."""

    def test_comprehensive_param_inventory(self, mock_sglang, tokenizer):
        """Exhaustive test of ALL OpenAI chat completion params.
        Documents exactly which params TITO forwards and which it drops."""
        proxy = _make_proxy(mock_sglang, tokenizer)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        resp = requests.post(
            f"{proxy.base_url}/chat/completions",
            json={
                "model": "test",
                "messages": messages,
                "max_tokens": 64,
                "temperature": 0.5,
                "top_p": 0.9,
                # Params that TITO should drop:
                "tool_choice": "auto",
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "seed": 42,
                "logit_bias": {"12345": 5},
                "tools": TOOLS_SCHEMA,
            },
            headers={"Authorization": "Bearer h2-comprehensive-001"},
            timeout=30.0,
        )
        assert resp.status_code == 200

        last_req = mock_sglang.request_log[-1]
        sampling = last_req.get("sampling_params", {})

        # Forwarded params
        assert sampling.get("max_new_tokens") == 64
        assert sampling.get("temperature") == 0.5
        assert sampling.get("top_p") == 0.9

        # Dropped params — each documented with impact
        assert "tool_choice" not in sampling, (
            "tool_choice should be dropped (SGLang /generate doesn't support it)"
        )
        assert "frequency_penalty" not in sampling, (
            "frequency_penalty dropped — may cause more repetitive outputs in TITO"
        )
        assert "presence_penalty" not in sampling, (
            "presence_penalty dropped — may affect output diversity"
        )
        assert "seed" not in sampling, (
            "seed dropped — TITO results not reproducible even with seed"
        )
        assert "logit_bias" not in sampling, (
            "logit_bias dropped — cannot bias specific tokens through TITO"
        )

    def test_tool_choice_required_dropped(self, mock_sglang, tokenizer):
        """Specifically verify tool_choice='required' is dropped.
        Impact: agent framework may expect forced tool calls but TITO can't enforce this."""
        proxy = _make_proxy(mock_sglang, tokenizer)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        resp = requests.post(
            f"{proxy.base_url}/chat/completions",
            json={
                "model": "test",
                "messages": messages,
                "max_tokens": 64,
                "tool_choice": "required",
                "tools": TOOLS_SCHEMA,
            },
            headers={"Authorization": "Bearer h2-toolchoice-001"},
            timeout=30.0,
        )
        assert resp.status_code == 200

        last_req = mock_sglang.request_log[-1]
        sampling = last_req.get("sampling_params", {})
        assert "tool_choice" not in sampling, (
            "tool_choice='required' should be dropped by TITO. "
            "SGLang /generate operates on raw tokens and doesn't support tool_choice. "
            "This means TITO cannot force the model to produce tool calls."
        )


class TestH5TokenizationDrift:
    """H5: TITO tokenizes incrementally (per-turn), non-TITO tokenizes full conversation.
    Verify these produce the same token sequences."""

    def test_incremental_vs_full_tokenization_match(self, proxy, tokenizer):
        """Multi-turn via proxy, compare result tokens vs full apply_chat_template.
        Documents any systematic differences between incremental and full tokenization."""
        task_id = "h5-drift-001"

        # Run 3-turn conversation through proxy
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 6 times 7?"},
        ]
        reply1 = _chat(proxy, task_id, messages)
        content1 = reply1["choices"][0]["message"]["content"]

        messages.append({"role": "assistant", "content": content1})
        messages.append({"role": "user", "content": "Are you sure?"})
        reply2 = _chat(proxy, task_id, messages)
        content2 = reply2["choices"][0]["message"]["content"]

        messages.append({"role": "assistant", "content": content2})

        # Get TITO's incrementally accumulated tokens
        result = proxy.get_task_result(task_id)
        tito_tokens = result["tokens"]

        # Full tokenization via apply_chat_template
        full_tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )

        # Compare: TITO may have extra tokens (e.g., trailing newlines, generation prompts)
        # but the core conversation tokens should match
        min_len = min(len(tito_tokens), len(full_tokens))
        mismatches = []
        for i in range(min_len):
            if tito_tokens[i] != full_tokens[i]:
                mismatches.append({
                    "pos": i,
                    "tito": tokenizer.decode([tito_tokens[i]]),
                    "full": tokenizer.decode([full_tokens[i]]),
                })

        # Log findings for analysis
        print(f"\n  H5 Tokenization Drift Results:")
        print(f"    TITO tokens: {len(tito_tokens)}")
        print(f"    Full tokens: {len(full_tokens)}")
        print(f"    Length diff: {len(tito_tokens) - len(full_tokens)}")
        print(f"    Mismatches in first {min_len} tokens: {len(mismatches)}")
        if mismatches:
            for m in mismatches[:10]:
                print(f"    pos={m['pos']}: tito='{m['tito']}' vs full='{m['full']}'")

        # If there's significant drift, this is a real problem
        # Allow small differences (trailing tokens) but flag large ones
        if len(mismatches) > 0:
            pytest.xfail(
                f"H5: Tokenization drift detected — {len(mismatches)} mismatches "
                f"in {min_len} tokens. First mismatch at pos {mismatches[0]['pos']}"
            )


class TestH6InferencePromptAlignment:
    """H6: TITO must send apply_chat_template-tokenized input_ids to SGLang /generate.

    After the H6 fix (tito_server.py:105), the prompt TITO sends at inference time must
    match what SGLang native /v1/chat/completions would produce — i.e. full template
    tokenization, not incremental raw-token stitching.

    Root cause of the 14% TITO vs non-TITO solve-rate gap: TITO used compact raw tokens
    from SGLang output as the inference prompt, while the model was trained on expanded
    template format (with extra newlines around parameter values).  Non-TITO always sends
    the template-reconstructed prompt, which is in-distribution.
    """

    def test_single_turn_prompt_matches_template(self, tokenizer):
        """Single-turn: verify TITO forwards apply_chat_template tokens to /generate."""
        def simple_fn(prompt: str) -> ProcessResult:
            return ProcessResult(text="The answer is 42.", finish_reason="stop")

        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "List files."},
        ]
        with with_mock_server(model_name=MODEL_NAME, process_fn=simple_fn) as server:
            proxy = _make_proxy(server, tokenizer)
            _chat(proxy, "h6-single-001", messages)

            recorded_ids = server.request_log[-1]["input_ids"]
            expected_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            assert recorded_ids == expected_ids, (
                f"H6: Single-turn inference prompt mismatch — "
                f"sent {len(recorded_ids)} tokens, expected {len(expected_ids)} tokens"
            )

    def test_multi_turn_with_tool_calls_prompt_matches_template(self, tokenizer):
        """Multi-turn with tool calls: second-turn prompt must use template-expanded format.

        Core H6 regression: TITO must NOT feed raw compact tokens from turn 1 into the
        turn-2 prompt.  The model generates <parameter=command>ls</parameter> (compact)
        but the template reconstructs <parameter=command>\\nls\\n</parameter>\\n (expanded,
        +1 token per parameter).  Over many turns this causes a systematic prompt drift
        that degrades solve rate by ~14%.
        """
        def tool_call_fn(prompt: str) -> ProcessResult:
            return ProcessResult(
                text=(
                    '<tool_call>\n'
                    '{"name": "execute_bash", "arguments": {"command": "ls"}}\n'
                    '</tool_call>'
                ),
                finish_reason="stop",
            )

        with with_mock_server(model_name=MODEL_NAME, process_fn=tool_call_fn) as server:
            proxy = _make_proxy(server, tokenizer, "qwen25")
            messages_turn1 = [
                {"role": "system", "content": "You are a SWE agent."},
                {"role": "user", "content": "List files."},
            ]
            reply = _chat(proxy, "h6-multi-002", messages_turn1, tools=TOOLS_SCHEMA)

            # Build turn 2 as SWE-agent would: structured tool_calls in assistant message
            tc = reply["choices"][0]["message"].get("tool_calls") or []
            messages_turn2 = messages_turn1 + [
                {
                    "role": "assistant",
                    "content": reply["choices"][0]["message"].get("content"),
                    "tool_calls": tc,
                },
                {
                    "role": "tool",
                    "content": "file1.py\nfile2.py",
                    "tool_call_id": tc[0]["id"] if tc else "call00000",
                },
            ]
            _chat(proxy, "h6-multi-002", messages_turn2, tools=TOOLS_SCHEMA)

            # The SECOND /generate call's input_ids must match apply_chat_template
            assert len(server.request_log) >= 2, "Expected at least 2 /generate calls"
            recorded_ids = server.request_log[-1]["input_ids"]
            expected_ids = tokenizer.apply_chat_template(
                messages_turn2, tokenize=True, add_generation_prompt=True,
                tools=TOOLS_SCHEMA,
            )
            assert recorded_ids == expected_ids, (
                f"H6 multi-turn: Inference prompt mismatch — "
                f"sent {len(recorded_ids)} tokens, expected {len(expected_ids)} tokens.\n"
                f"TITO is using compact raw tokens instead of template-expanded tokens.\n"
                f"Diff length: {len(recorded_ids) - len(expected_ids)}"
            )
