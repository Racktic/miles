"""Compare TITO vs non-TITO tokenization using a real SGLang engine.

Test flow:
  1. Send prompt to SGLang /generate → get generated text + exact token IDs
  2. Build full conversation messages (prompt + assistant response text)
  3. TITO path:     use SGLang's exact output token IDs (from output_token_logprobs)
  4. Non-TITO path: re-tokenize from text via build_tokens_and_mask_from_messages()
  5. Compare token sequences and loss masks

Test categories:
  - Multi-turn user↔assistant conversations
  - Tool-calling: assistant with tool_calls → tool response → assistant
  - Mixed: multi-turn with interleaved tool calls
  - Edge cases: empty content, long responses, special characters

Usage:
    # Connect to an existing SGLang server:
    python tests/proxy/test_tito_vs_non_tito_tokens.py --skip-launch --port 30000

    # Or launch a new server:
    python tests/proxy/test_tito_vs_non_tito_tokens.py \
        --model-path /data/checkpoints/Qwen3-Coder-30B-A3B-Instruct \
        --tp-size 4 --port 30000
"""

import argparse
import json
import multiprocessing
import random
import signal
import sys
import time

import requests
from transformers import AutoTokenizer

sys.path.insert(0, ".")
sys.path.insert(0, "uda/swe_agent")

from uda.swe_agent.generate_with_swe_agent import build_tokens_and_mask_from_messages
from uda.swe_agent.proxy.tito_converter import ChatConverter
from uda.swe_agent.proxy.tito_state import TaskState


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def launch_server(model_path: str, tp_size: int, port: int, tool_call_parser: str = None) -> multiprocessing.Process:
    from sglang.srt.entrypoints.http_server import launch_server as _launch
    from sglang.srt.server_args import ServerArgs

    kwargs = dict(
        model_path=model_path,
        tp_size=tp_size,
        port=port,
        host="127.0.0.1",
        trust_remote_code=True,
    )
    if tool_call_parser:
        kwargs["tool_call_parser"] = tool_call_parser

    server_args = ServerArgs(**kwargs)
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=_launch, args=(server_args,))
    p.start()
    return p


def wait_for_server(base_url: str, timeout: int = 600) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(5)
    raise TimeoutError(f"Server not ready after {timeout}s")


# ---------------------------------------------------------------------------
# TITO state helpers (shared setup for all test functions)
# ---------------------------------------------------------------------------

def make_tito_state(tokenizer):
    """Create a fresh TaskState + ChatConverter + prefix IDs for TITO simulation."""
    new_line_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    gen_prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "."}], tokenize=False, add_generation_prompt=True
    )
    no_gen_prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "."}], tokenize=False, add_generation_prompt=False
    )
    assistant_prefix_ids = tokenizer.encode(
        gen_prompt_text[len(no_gen_prompt_text):], add_special_tokens=False
    )
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    state = TaskState(
        new_line_token_id=new_line_token_id,
        assistant_prefix_ids=assistant_prefix_ids,
        im_end_token_id=im_end_token_id,
    )
    converter = ChatConverter(tokenizer=tokenizer)
    return state, converter, assistant_prefix_ids


def generate_one_turn(base_url, input_ids, assistant_prefix_ids, max_new_tokens=64):
    """Send input_ids + generation prompt to SGLang /generate, return output_ids and text."""
    payload = {
        "input_ids": input_ids + list(assistant_prefix_ids),
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.7,
        },
        "return_logprob": True,
    }
    r = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    r.raise_for_status()
    output = r.json()

    if "text" not in output:
        return None, None, None

    text = output["text"]
    meta_info = output["meta_info"]
    output_token_logprobs = meta_info.get("output_token_logprobs", [])
    output_ids = [item[1] for item in output_token_logprobs]
    output_logprobs = [item[0] for item in output_token_logprobs]
    return text, output_ids, output_logprobs


# ---------------------------------------------------------------------------
# Test: Multi-turn user↔assistant
# ---------------------------------------------------------------------------

def do_multi_turn_generate(base_url, tokenizer, system_msg, user_msg,
                           num_turns=3, max_new_tokens=64):
    """Simulate a multi-turn user↔assistant conversation."""
    state, converter, assistant_prefix_ids = make_tito_state(tokenizer)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    items = converter.tokenize_new_messages(messages, pre_msg_length=0)
    state.add_message_items(items)

    for turn in range(num_turns):
        input_ids = state.get_input_ids()
        text, output_ids, output_logprobs = generate_one_turn(
            base_url, input_ids, assistant_prefix_ids, max_new_tokens
        )
        if text is None:
            print(f"  WARNING: SGLang error on turn {turn}")
            break

        state.add_response(output_ids, output_logprobs)
        messages.append({"role": "assistant", "content": text})

        if turn < num_turns - 1:
            followup = {"role": "user", "content": f"Continue from turn {turn + 1}."}
            messages.append(followup)
            new_items = converter.tokenize_new_messages(messages, pre_msg_length=len(messages) - 1)
            state.add_message_items(new_items)

    tito_result = state.finalize()
    return {
        "messages": messages,
        "tito_tokens": tito_result["tokens"],
        "tito_loss_mask": tito_result["loss_mask"],
        "tito_response_length": tito_result["response_length"],
    }


# ---------------------------------------------------------------------------
# Test: Tool-calling (assistant→tool→assistant)
# ---------------------------------------------------------------------------

def do_tool_call_generate(base_url, tokenizer, system_msg, user_msg,
                          num_tool_rounds=2, max_new_tokens=64):
    """Simulate assistant making tool calls and receiving tool responses.

    Flow per round:
      1. Model generates assistant response (simulated as content)
      2. We inject a tool response
      3. Model generates follow-up assistant response

    Note: In production, the model generates actual tool_call syntax (e.g., <tool_call>...)
    which TITO captures as raw tokens. The SWE-agent framework parses this into structured
    tool_calls. Since we can't simulate real tool_call generation here, we test with plain
    assistant content + tool responses (which exercises the same tokenization paths).
    """
    state, converter, assistant_prefix_ids = make_tito_state(tokenizer)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"}
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "Edit a file by replacing text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "path": {"type": "string"},
                        "old_str": {"type": "string"},
                        "new_str": {"type": "string"},
                    },
                    "required": ["command", "path"],
                },
            },
        },
    ]

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    items = converter.tokenize_new_messages(messages, pre_msg_length=0, tools=tools)
    state.add_message_items(items)

    for round_idx in range(num_tool_rounds):
        # Step 1: Model generates a response (plain content)
        input_ids = state.get_input_ids()
        text, output_ids, output_logprobs = generate_one_turn(
            base_url, input_ids, assistant_prefix_ids, max_new_tokens
        )
        if text is None:
            print(f"  WARNING: SGLang error on round {round_idx} step 1")
            break

        state.add_response(output_ids, output_logprobs)
        messages.append({"role": "assistant", "content": text})

        # Step 2: Add tool response
        tool_response = {
            "role": "tool",
            "content": f"total 4\ndrwxr-xr-x 2 root root 4096 Mar 8 00:00 round_{round_idx}\n-rw-r--r-- 1 root root 42 Mar 8 00:00 output.txt",
        }
        messages.append(tool_response)
        new_items = converter.tokenize_new_messages(messages, pre_msg_length=len(messages) - 1, tools=tools)
        state.add_message_items(new_items)

        # Step 3: Model generates follow-up after tool response
        input_ids = state.get_input_ids()
        text2, output_ids2, output_logprobs2 = generate_one_turn(
            base_url, input_ids, assistant_prefix_ids, max_new_tokens
        )
        if text2 is None:
            print(f"  WARNING: SGLang error on round {round_idx} step 3")
            break

        state.add_response(output_ids2, output_logprobs2)
        messages.append({"role": "assistant", "content": text2})

        # Add user follow-up if not last round
        if round_idx < num_tool_rounds - 1:
            followup = {"role": "user", "content": f"Now check round {round_idx + 1}."}
            messages.append(followup)
            new_items = converter.tokenize_new_messages(messages, pre_msg_length=len(messages) - 1, tools=tools)
            state.add_message_items(new_items)

    tito_result = state.finalize()
    return {
        "messages": messages,
        "tito_tokens": tito_result["tokens"],
        "tito_loss_mask": tito_result["loss_mask"],
        "tito_response_length": tito_result["response_length"],
        "tools": tools,
    }


# ---------------------------------------------------------------------------
# Test: Mixed multi-turn with tool calls and plain user messages
# ---------------------------------------------------------------------------

def do_mixed_conversation(base_url, tokenizer, max_new_tokens=64):
    """Simulate a realistic SWE-agent conversation:
    system → user → assistant → tool → assistant → user → assistant

    Uses plain assistant content (no tool_calls field) since TITO captures raw model
    output, not the reformatted tool_call syntax from apply_chat_template.
    """
    state, converter, assistant_prefix_ids = make_tito_state(tokenizer)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"}
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    system_msg = "You are a software engineer. Use execute_bash to run commands."
    user_msg = "There's a bug in main.py line 42. The function returns None instead of 0. Fix it."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    items = converter.tokenize_new_messages(messages, pre_msg_length=0, tools=tools)
    state.add_message_items(items)

    # Turn 1: assistant generates
    input_ids = state.get_input_ids()
    text1, ids1, lp1 = generate_one_turn(base_url, input_ids, assistant_prefix_ids, max_new_tokens)
    state.add_response(ids1, lp1)
    messages.append({"role": "assistant", "content": text1})

    # Turn 2: tool response
    tool_resp = {
        "role": "tool",
        "content": "def compute(x):\n    if x > 0:\n        return x * 2\n    # BUG: missing return 0",
    }
    messages.append(tool_resp)
    new_items = converter.tokenize_new_messages(messages, pre_msg_length=len(messages) - 1, tools=tools)
    state.add_message_items(new_items)

    # Turn 3: assistant analyzes
    input_ids = state.get_input_ids()
    text2, ids2, lp2 = generate_one_turn(base_url, input_ids, assistant_prefix_ids, max_new_tokens)
    state.add_response(ids2, lp2)
    messages.append({"role": "assistant", "content": text2})

    # Turn 4: user follow-up
    followup = {"role": "user", "content": "Did you fix the return statement?"}
    messages.append(followup)
    new_items = converter.tokenize_new_messages(messages, pre_msg_length=len(messages) - 1, tools=tools)
    state.add_message_items(new_items)

    # Turn 5: assistant final response
    input_ids = state.get_input_ids()
    text3, ids3, lp3 = generate_one_turn(base_url, input_ids, assistant_prefix_ids, max_new_tokens)
    state.add_response(ids3, lp3)
    messages.append({"role": "assistant", "content": text3})

    tito_result = state.finalize()
    return {
        "messages": messages,
        "tito_tokens": tito_result["tokens"],
        "tito_loss_mask": tito_result["loss_mask"],
        "tito_response_length": tito_result["response_length"],
        "tools": tools,
    }


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

def do_single_turn(base_url, tokenizer, system_msg, user_msg, max_new_tokens=64):
    """Single assistant turn — simplest possible case."""
    state, converter, assistant_prefix_ids = make_tito_state(tokenizer)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    items = converter.tokenize_new_messages(messages, pre_msg_length=0)
    state.add_message_items(items)

    input_ids = state.get_input_ids()
    text, output_ids, output_logprobs = generate_one_turn(
        base_url, input_ids, assistant_prefix_ids, max_new_tokens
    )
    state.add_response(output_ids, output_logprobs)
    messages.append({"role": "assistant", "content": text})

    tito_result = state.finalize()
    return {
        "messages": messages,
        "tito_tokens": tito_result["tokens"],
        "tito_loss_mask": tito_result["loss_mask"],
        "tito_response_length": tito_result["response_length"],
    }


def do_multiple_tool_calls_single_turn(base_url, tokenizer, max_new_tokens=64):
    """Assistant generates, then receives multiple tool responses, then generates again."""
    state, converter, assistant_prefix_ids = make_tito_state(tokenizer)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"}
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    messages = [
        {"role": "system", "content": "You are a SWE agent with bash and editor tools."},
        {"role": "user", "content": "Check the git status and also read the config file."},
    ]
    items = converter.tokenize_new_messages(messages, pre_msg_length=0, tools=tools)
    state.add_message_items(items)

    # Assistant generates (plain content)
    input_ids = state.get_input_ids()
    text, ids, lp = generate_one_turn(base_url, input_ids, assistant_prefix_ids, max_new_tokens)
    state.add_response(ids, lp)
    messages.append({"role": "assistant", "content": text})

    # Two tool responses
    for content in [
        "On branch main\nnothing to commit, working tree clean",
        "debug: false\nport: 8080\nlog_level: info",
    ]:
        messages.append({"role": "tool", "content": content})
        new_items = converter.tokenize_new_messages(messages, pre_msg_length=len(messages) - 1, tools=tools)
        state.add_message_items(new_items)

    # Final assistant response
    input_ids = state.get_input_ids()
    text2, ids2, lp2 = generate_one_turn(base_url, input_ids, assistant_prefix_ids, max_new_tokens)
    state.add_response(ids2, lp2)
    messages.append({"role": "assistant", "content": text2})

    tito_result = state.finalize()
    return {
        "messages": messages,
        "tito_tokens": tito_result["tokens"],
        "tito_loss_mask": tito_result["loss_mask"],
        "tito_response_length": tito_result["response_length"],
        "tools": tools,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_tokens(test_name, messages, tito_tokens, tito_mask, tito_resp_len,
                   tokenizer, tools=None):
    """Compare TITO tokens vs non-TITO build_tokens_and_mask_from_messages()."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

    roles = [m.get("role") for m in messages]
    n_assistant = sum(1 for r in roles if r == "assistant")
    n_tool = sum(1 for r in roles if r == "tool")
    n_tool_calls = sum(1 for m in messages if m.get("tool_calls"))
    print(f"  Messages: {len(messages)} (assistant={n_assistant}, tool={n_tool}, tool_calls={n_tool_calls})")

    # Non-TITO path
    non_tito_tokens, non_tito_mask, _, non_tito_resp_len = build_tokens_and_mask_from_messages(
        messages, tokenizer, tools=tools
    )

    print(f"  TITO:     {len(tito_tokens)} tokens, response_len={tito_resp_len}, mask_sum={sum(tito_mask)}")
    print(f"  Non-TITO: {len(non_tito_tokens)} tokens, response_len={non_tito_resp_len}, mask_sum={sum(non_tito_mask)}")

    passed = True

    # Compare full token sequences
    if tito_tokens == non_tito_tokens:
        print(f"  TOKENS: MATCH ({len(tito_tokens)} tokens)")
    else:
        passed = False
        print(f"  TOKENS: MISMATCH!")
        min_len = min(len(tito_tokens), len(non_tito_tokens))
        diffs = sum(1 for i in range(min_len) if tito_tokens[i] != non_tito_tokens[i])
        print(f"    Length: tito={len(tito_tokens)} vs non_tito={len(non_tito_tokens)}")
        print(f"    Diffs in shared range [0:{min_len}]: {diffs}")

        for i in range(min_len):
            if tito_tokens[i] != non_tito_tokens[i]:
                ctx_start = max(0, i - 3)
                ctx_end = min(min_len, i + 4)
                print(f"    First diff at index {i}:")
                print(f"      TITO     [{ctx_start}:{ctx_end}]: {tito_tokens[ctx_start:ctx_end]}")
                print(f"      Non-TITO [{ctx_start}:{ctx_end}]: {non_tito_tokens[ctx_start:ctx_end]}")
                print(f"      TITO     decoded: {tokenizer.decode(tito_tokens[ctx_start:ctx_end])!r}")
                print(f"      Non-TITO decoded: {tokenizer.decode(non_tito_tokens[ctx_start:ctx_end])!r}")
                break

        if len(tito_tokens) != len(non_tito_tokens):
            longer = "tito" if len(tito_tokens) > len(non_tito_tokens) else "non_tito"
            longer_tokens = tito_tokens if len(tito_tokens) > len(non_tito_tokens) else non_tito_tokens
            extra = longer_tokens[min_len:]
            print(f"    Extra tokens in {longer}: {extra[:20]}{'...' if len(extra) > 20 else ''}")
            print(f"    Extra decoded: {tokenizer.decode(extra[:20])!r}")

    # Compare loss masks
    if tito_mask == non_tito_mask:
        print(f"  LOSS_MASK: MATCH (sum={sum(tito_mask)})")
    else:
        passed = False
        print(f"  LOSS_MASK: MISMATCH!")
        print(f"    TITO     mask: len={len(tito_mask)}, sum={sum(tito_mask)}")
        print(f"    Non-TITO mask: len={len(non_tito_mask)}, sum={sum(non_tito_mask)}")
        min_len = min(len(tito_mask), len(non_tito_mask))
        for i in range(min_len):
            if tito_mask[i] != non_tito_mask[i]:
                print(f"    First diff at index {i}: tito={tito_mask[i]} vs non_tito={non_tito_mask[i]}")
                tokens = tito_tokens if len(tito_tokens) > i else non_tito_tokens
                prompt_len = len(tito_tokens) - len(tito_mask)
                abs_idx = prompt_len + i
                if abs_idx < len(tokens):
                    ctx_start = max(0, abs_idx - 2)
                    ctx_end = min(len(tokens), abs_idx + 3)
                    print(f"    Token context: {tokenizer.decode(tokens[ctx_start:ctx_end])!r}")
                break

    # Compare response lengths
    if tito_resp_len != non_tito_resp_len:
        if passed:  # only flag if tokens matched (resp_len diff might be secondary)
            passed = False
        print(f"  RESPONSE_LEN: MISMATCH! tito={tito_resp_len} vs non_tito={non_tito_resp_len}")

    return passed


# ---------------------------------------------------------------------------
# Tool call hypothesis tests (H1: parser format, H2: param drops)
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command in the terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "Edit a file by replacing text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The editor command."},
                    "path": {"type": "string", "description": "File path."},
                    "old_str": {"type": "string", "description": "Text to replace."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                },
                "required": ["command", "path"],
            },
        },
    },
]

# Prompts designed to elicit tool calls from the model
TOOL_CALL_PROMPTS = [
    {
        "name": "simple_ls",
        "system": "You are a SWE agent. Use the execute_bash tool to run commands.",
        "user": "List all Python files in the current directory. Use execute_bash.",
    },
    {
        "name": "find_bug",
        "system": "You are a software engineer debugging a Python project. You have access to execute_bash and str_replace_editor tools.",
        "user": "Check if there is a file called main.py in the current directory using the bash tool.",
    },
    {
        "name": "read_file",
        "system": "You are a SWE agent. Use tools to investigate code.",
        "user": "Run `cat README.md` to read the readme file. Use execute_bash.",
    },
]


def do_tool_call_comparison(base_url, tokenizer, tool_call_parser,
                            prompt_config, max_new_tokens=256):
    """Compare tool call parsing: TITO (build_response) vs SGLang native /v1/chat/completions.

    Returns dict with native response, TITO response, and raw model text.
    """
    messages = [
        {"role": "system", "content": prompt_config["system"]},
        {"role": "user", "content": prompt_config["user"]},
    ]

    # Path A: SGLang native /v1/chat/completions
    native_resp = requests.post(f"{base_url}/v1/chat/completions", json={
        "model": "default",
        "messages": messages,
        "tools": TOOLS_SCHEMA,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
    }, timeout=120).json()

    # Path B: /generate → TITO build_response
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, tools=TOOLS_SCHEMA
    )
    gen_resp = requests.post(f"{base_url}/generate", json={
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        "return_logprob": True,
    }, timeout=120).json()

    raw_text = gen_resp.get("text", "")

    # Build TITO response using ChatConverter
    converter = ChatConverter(tokenizer=tokenizer, tool_call_parser=tool_call_parser)
    tito_resp = converter.build_response(
        request_data={"model": "default", "tools": TOOLS_SCHEMA},
        generate_output=gen_resp,
        text=raw_text,
    )

    return {
        "native": native_resp,
        "tito": tito_resp,
        "raw_text": raw_text,
        "prompt_name": prompt_config["name"],
    }


def test_raw_model_output_format(base_url, tokenizer, max_new_tokens=256):
    """THE DEFINITIVE TEST: What format does the model actually generate?

    Sends prompts with tools to /generate, inspects raw text for:
    - XML format: <function=name><parameter=key>value</parameter></function>
    - JSON format: {"name": ..., "arguments": ...}
    """
    print(f"\n{'='*70}")
    print("TEST: Raw Model Output Format Detection")
    print(f"{'='*70}")

    xml_count = 0
    json_count = 0
    no_tool_count = 0

    for prompt_config in TOOL_CALL_PROMPTS:
        messages = [
            {"role": "system", "content": prompt_config["system"]},
            {"role": "user", "content": prompt_config["user"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, tools=TOOLS_SCHEMA
        )
        gen_resp = requests.post(f"{base_url}/generate", json={
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
            "return_logprob": True,
        }, timeout=120).json()

        raw_text = gen_resp.get("text", "")
        has_tool_call_marker = "<tool_call>" in raw_text
        has_xml_function = "<function=" in raw_text
        has_json_name = '"name"' in raw_text and "<tool_call>" in raw_text

        if has_xml_function:
            fmt = "XML"
            xml_count += 1
        elif has_json_name:
            fmt = "JSON"
            json_count += 1
        elif has_tool_call_marker:
            fmt = "UNKNOWN (has <tool_call> but neither XML nor JSON)"
        else:
            fmt = "NO_TOOL_CALL"
            no_tool_count += 1

        print(f"\n  [{prompt_config['name']}] Format: {fmt}")
        print(f"    has_<tool_call>: {has_tool_call_marker}")
        print(f"    has_<function=:  {has_xml_function}")
        print(f"    has_json_name:   {has_json_name}")
        # Print first 500 chars of raw text for inspection
        preview = raw_text[:500].replace('\n', '\n    ')
        print(f"    Raw text preview:\n    {preview}")

    print(f"\n  SUMMARY: XML={xml_count}, JSON={json_count}, no_tool={no_tool_count}")

    if xml_count > 0 and json_count == 0:
        print("  VERDICT: Model generates XML format → qwen3_coder parser is CORRECT")
        return "xml"
    elif json_count > 0 and xml_count == 0:
        print("  VERDICT: Model generates JSON format → qwen3_coder parser is WRONG, use qwen25!")
        print("  >>> THIS IS THE ROOT CAUSE OF THE 11.5% SOLVE RATE GAP <<<")
        return "json"
    elif xml_count > 0 and json_count > 0:
        print("  VERDICT: Model generates BOTH formats → need hybrid parser (Solution C)")
        return "mixed"
    else:
        print("  VERDICT: Model did not generate any tool calls (try longer max_new_tokens)")
        return "none"


def test_tito_vs_native_tool_call_parsing(base_url, tokenizer, tool_call_parser,
                                           max_new_tokens=256):
    """Compare TITO's tool call parsing against SGLang native /v1/chat/completions.

    For each prompt:
    - Native path: SGLang's built-in tool call parsing
    - TITO path: /generate → ChatConverter.build_response() with FunctionCallParser

    Checks: finish_reason, tool_calls detected, function names, arguments.
    """
    print(f"\n{'='*70}")
    print(f"TEST: TITO vs Native Tool Call Parsing (parser={tool_call_parser})")
    print(f"{'='*70}")

    all_passed = True

    for prompt_config in TOOL_CALL_PROMPTS:
        result = do_tool_call_comparison(
            base_url, tokenizer, tool_call_parser, prompt_config, max_new_tokens
        )
        native = result["native"]
        tito = result["tito"]
        raw_text = result["raw_text"]

        native_choice = native.get("choices", [{}])[0]
        tito_choice = tito.get("choices", [{}])[0]

        native_fr = native_choice.get("finish_reason", "N/A")
        tito_fr = tito_choice.get("finish_reason", "N/A")

        native_tc = native_choice.get("message", {}).get("tool_calls") or []
        tito_tc = tito_choice.get("message", {}).get("tool_calls") or []

        # Extract tool names
        native_names = sorted([tc["function"]["name"] for tc in native_tc]) if native_tc else []
        tito_names = sorted([tc["function"]["name"] for tc in tito_tc]) if tito_tc else []

        passed = True
        issues = []

        # Check finish_reason match
        if native_fr != tito_fr:
            # "tool_calls" vs "stop" is the key mismatch
            if native_fr == "tool_calls" and tito_fr == "stop":
                issues.append(f"TITO missed tool calls! native={native_fr} tito={tito_fr}")
            elif native_fr == "stop" and tito_fr == "tool_calls":
                issues.append(f"TITO found extra tool calls! native={native_fr} tito={tito_fr}")
            else:
                issues.append(f"finish_reason mismatch: native={native_fr} tito={tito_fr}")
            passed = False

        # Check tool call count
        if len(native_tc) != len(tito_tc):
            issues.append(f"tool_call count: native={len(native_tc)} tito={len(tito_tc)}")
            passed = False

        # Check tool names
        if native_names != tito_names:
            issues.append(f"tool names: native={native_names} tito={tito_names}")
            passed = False

        # Check arguments are valid JSON
        for i, tc in enumerate(tito_tc):
            args_str = tc.get("function", {}).get("arguments", "")
            try:
                json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                issues.append(f"tito tool_calls[{i}].arguments is not valid JSON: {args_str[:100]}")
                passed = False

        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {result['prompt_name']}")
        print(f"    Native: finish_reason={native_fr}, tool_calls={len(native_tc)}, names={native_names}")
        print(f"    TITO:   finish_reason={tito_fr}, tool_calls={len(tito_tc)}, names={tito_names}")

        if issues:
            for issue in issues:
                print(f"    ISSUE: {issue}")
            # Print raw text for debugging
            preview = raw_text[:300].replace('\n', '\n    ')
            print(f"    Raw model output:\n    {preview}")

        if not passed:
            all_passed = False

    return all_passed


def test_tool_call_parser_comparison(base_url, tokenizer, max_new_tokens=256):
    """Try parsing the same raw model text with BOTH qwen3_coder and qwen25 parsers.

    If qwen3_coder finds 0 calls but qwen25 finds 1+, the format mismatch is confirmed.
    """
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from pydantic import TypeAdapter
    from sglang.srt.entrypoints.openai.protocol import Tool

    print(f"\n{'='*70}")
    print("TEST: Parser Comparison (qwen3_coder vs qwen25 on same raw text)")
    print(f"{'='*70}")

    parsed_tools = TypeAdapter(list[Tool]).validate_python(TOOLS_SCHEMA)

    for prompt_config in TOOL_CALL_PROMPTS:
        messages = [
            {"role": "system", "content": prompt_config["system"]},
            {"role": "user", "content": prompt_config["user"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, tools=TOOLS_SCHEMA
        )
        gen_resp = requests.post(f"{base_url}/generate", json={
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
            "return_logprob": True,
        }, timeout=120).json()

        raw_text = gen_resp.get("text", "")

        # Parse with qwen3_coder
        try:
            parser_xml = FunctionCallParser(parsed_tools, "qwen3_coder")
            _, calls_xml = parser_xml.parse_non_stream(raw_text)
            n_xml = len(calls_xml) if calls_xml else 0
            xml_names = [c.name for c in calls_xml] if calls_xml else []
        except Exception as e:
            n_xml = -1
            xml_names = [f"ERROR: {e}"]

        # Parse with qwen25
        try:
            parser_json = FunctionCallParser(parsed_tools, "qwen25")
            _, calls_json = parser_json.parse_non_stream(raw_text)
            n_json = len(calls_json) if calls_json else 0
            json_names = [c.name for c in calls_json] if calls_json else []
        except Exception as e:
            n_json = -1
            json_names = [f"ERROR: {e}"]

        mismatch = (n_xml != n_json) or (xml_names != json_names)
        status = "MISMATCH" if mismatch else "MATCH"

        print(f"\n  [{status}] {prompt_config['name']}")
        print(f"    qwen3_coder: {n_xml} calls, names={xml_names}")
        print(f"    qwen25:      {n_json} calls, names={json_names}")

        if mismatch:
            if n_xml == 0 and n_json > 0:
                print(f"    >>> SMOKING GUN: qwen3_coder FAILS but qwen25 SUCCEEDS <<<")
                print(f"    >>> Model generates JSON format, not XML <<<")
            elif n_xml > 0 and n_json == 0:
                print(f"    >>> Model generates XML format (qwen3_coder is correct) <<<")


def test_tito_response_usable_by_agent(base_url, tokenizer, tool_call_parser,
                                        max_new_tokens=256):
    """Validate TITO responses are usable by the SWE-agent.

    Checks that tool_calls have all required fields with correct types.
    """
    print(f"\n{'='*70}")
    print(f"TEST: TITO Response Usable by Agent (parser={tool_call_parser})")
    print(f"{'='*70}")

    all_passed = True

    for prompt_config in TOOL_CALL_PROMPTS:
        result = do_tool_call_comparison(
            base_url, tokenizer, tool_call_parser, prompt_config, max_new_tokens
        )
        tito = result["tito"]
        tito_choice = tito.get("choices", [{}])[0]
        tito_tc = tito_choice.get("message", {}).get("tool_calls") or []

        passed = True
        issues = []

        if not tito_tc:
            if "<tool_call>" in result["raw_text"]:
                issues.append("No tool_calls in TITO response but raw text has <tool_call> markers!")
                passed = False
            else:
                issues.append("No tool_calls (model didn't generate any)")

        for i, tc in enumerate(tito_tc):
            # Check id
            tc_id = tc.get("id", "")
            if not tc_id or not tc_id.startswith("call_"):
                issues.append(f"tool_calls[{i}].id missing or bad format: {tc_id!r}")
                passed = False

            # Check type
            if tc.get("type") != "function":
                issues.append(f"tool_calls[{i}].type != 'function': {tc.get('type')!r}")
                passed = False

            # Check function.name
            func_name = tc.get("function", {}).get("name", "")
            valid_names = {t["function"]["name"] for t in TOOLS_SCHEMA}
            if func_name not in valid_names:
                issues.append(f"tool_calls[{i}].function.name={func_name!r} not in {valid_names}")
                passed = False

            # Check function.arguments is valid JSON
            args_str = tc.get("function", {}).get("arguments", "")
            try:
                args = json.loads(args_str)
                if not isinstance(args, dict):
                    issues.append(f"tool_calls[{i}].arguments is not a dict: {type(args)}")
                    passed = False
            except (json.JSONDecodeError, TypeError) as e:
                issues.append(f"tool_calls[{i}].arguments invalid JSON: {e}")
                passed = False

        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {prompt_config['name']}")
        print(f"    tool_calls count: {len(tito_tc)}")
        for issue in issues:
            print(f"    {issue}")

        if not passed:
            all_passed = False

    return all_passed


def run_tool_call_hypothesis_tests(base_url, tokenizer, tool_call_parser, max_new_tokens):
    """Run all tool call hypothesis tests and return results."""
    print(f"\n{'#'*70}")
    print(f"# TOOL CALL HYPOTHESIS TESTS (parser={tool_call_parser})")
    print(f"{'#'*70}")

    results = []

    # Test 1: What format does the model generate?
    format_result = test_raw_model_output_format(base_url, tokenizer, max_new_tokens)
    results.append(("format-detection", "Raw model output format", format_result != "none"))

    # Test 2: TITO vs native comparison
    passed = test_tito_vs_native_tool_call_parsing(
        base_url, tokenizer, tool_call_parser, max_new_tokens
    )
    results.append(("tito-vs-native", "TITO vs Native tool call parsing", passed))

    # Test 3: Parser comparison (qwen3_coder vs qwen25)
    test_tool_call_parser_comparison(base_url, tokenizer, max_new_tokens)
    results.append(("parser-comparison", "Parser comparison (informational)", True))

    # Test 4: Agent usability
    passed = test_tito_response_usable_by_agent(
        base_url, tokenizer, tool_call_parser, max_new_tokens
    )
    results.append(("agent-usability", "TITO response usable by agent", passed))

    # Summary
    print(f"\n{'='*70}")
    print("TOOL CALL HYPOTHESIS TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Model output format: {format_result.upper()}")
    for cat, name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    if format_result == "json":
        print(f"\n  DIAGNOSIS: Model generates JSON format but parser is '{tool_call_parser}'")
        print(f"  FIX: Change --sglang-tool-call-parser to 'qwen25'")
    elif format_result == "xml":
        print(f"\n  DIAGNOSIS: Parser format is correct. Tool call parsing is NOT the root cause.")
        print(f"  Next: Investigate H2 (param drops) or H3 (stop token)")
    elif format_result == "mixed":
        print(f"\n  DIAGNOSIS: Model generates both formats. Need hybrid parser fallback.")

    return results


# ---------------------------------------------------------------------------
# Expanded prompts for statistical tests
# ---------------------------------------------------------------------------

TOOL_CALL_PROMPTS_EXPANDED = [
    # Simple commands
    {"name": "simple_ls", "system": "You are a SWE agent. Use the execute_bash tool to run commands.",
     "user": "List all Python files in the current directory. Use execute_bash."},
    {"name": "cat_file", "system": "You are a SWE agent. Use tools to investigate code.",
     "user": "Run `cat README.md` to read the readme file. Use execute_bash."},
    {"name": "grep_search", "system": "You are a SWE agent with bash and editor tools.",
     "user": "Search for 'TODO' in all .py files using grep. Use execute_bash."},
    {"name": "git_status", "system": "You are a SWE agent. Use the execute_bash tool.",
     "user": "Check the current git status. Use execute_bash."},
    # Editor operations
    {"name": "edit_file", "system": "You are a SWE agent. Use str_replace_editor to edit files.",
     "user": "Replace 'old_function' with 'new_function' in src/main.py using the editor tool."},
    {"name": "view_file", "system": "You are a SWE agent with bash and editor tools.",
     "user": "View the first 20 lines of src/utils.py using str_replace_editor with command='view'."},
    # Complex
    {"name": "debug_error", "system": "You are a software engineer debugging a Python project. Use tools.",
     "user": "There's a TypeError on line 42 of utils.py. Run `python -c 'import utils'` to reproduce it."},
    {"name": "run_tests", "system": "You are a SWE agent. Use execute_bash to run tests.",
     "user": "Run `pytest tests/ -x` to find the first failing test."},
    {"name": "find_bug", "system": "You are a software engineer debugging a Python project. You have access to execute_bash and str_replace_editor tools.",
     "user": "Check if there is a file called main.py in the current directory using the bash tool."},
    {"name": "check_imports", "system": "You are a SWE agent. Use tools to investigate.",
     "user": "Check all Python imports in src/app.py using `grep import src/app.py`. Use execute_bash."},
]


# ---------------------------------------------------------------------------
# Part 3: H1 Deep Investigation — has_tool_call() gate accuracy
# ---------------------------------------------------------------------------

def print_tool_call_start_token(tool_call_parser="qwen3_coder"):
    """Diagnostic: Print the tool_call_start_token used by the has_tool_call() gate."""
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from pydantic import TypeAdapter
    from sglang.srt.entrypoints.openai.protocol import Tool

    parsed_tools = TypeAdapter(list[Tool]).validate_python(TOOLS_SCHEMA)

    for parser_name in [tool_call_parser, "qwen25"]:
        try:
            parser = FunctionCallParser(parsed_tools, parser_name)
            token = getattr(parser.detector, 'tool_call_start_token', 'UNKNOWN')
            print(f"  Parser '{parser_name}' → detector={type(parser.detector).__name__}"
                  f" → tool_call_start_token={token!r}")
        except Exception as e:
            print(f"  Parser '{parser_name}' → ERROR: {e}")


def _generate_with_tools(base_url, tokenizer, prompt_config, temperature=1.0,
                          max_new_tokens=512):
    """Generate raw text from /generate with tools in the prompt."""
    messages = [
        {"role": "system", "content": prompt_config["system"]},
        {"role": "user", "content": prompt_config["user"]},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, tools=TOOLS_SCHEMA
    )
    gen_resp = requests.post(f"{base_url}/generate", json={
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": temperature},
        "return_logprob": True,
    }, timeout=120).json()
    return gen_resp.get("text", ""), gen_resp


def test_has_tool_call_gate_accuracy(base_url, tokenizer, tool_call_parser,
                                      num_samples=30, temperature=1.0,
                                      max_new_tokens=512):
    """THE KEY TEST: Does has_tool_call() gate silently drop valid tool calls?

    For each sample at the specified temperature:
    1. Generate raw text from /generate
    2. Check: parser.has_tool_call(raw_text) — what the gate returns
    3. Force: parser.parse_non_stream(raw_text) — what would be found
    4. Compare: gate=False but parse found calls? → SMOKING GUN

    If gate_false_parse_found > 0, H1 is CONFIRMED as root cause.
    """
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from pydantic import TypeAdapter
    from sglang.srt.entrypoints.openai.protocol import Tool

    print(f"\n{'='*70}")
    print(f"TEST: has_tool_call() Gate Accuracy ({num_samples} samples, temp={temperature})")
    print(f"{'='*70}")

    parsed_tools = TypeAdapter(list[Tool]).validate_python(TOOLS_SCHEMA)
    parser = FunctionCallParser(parsed_tools, tool_call_parser)
    start_token = getattr(parser.detector, 'tool_call_start_token', 'UNKNOWN')
    print(f"  tool_call_start_token = {start_token!r}")

    stats = {
        "gate_true_parse_found": 0,
        "gate_true_parse_empty": 0,
        "gate_false_parse_found": 0,  # THE SMOKING GUN
        "gate_false_no_calls": 0,
    }
    gate_misses = []  # Store details of false negatives

    for i in range(num_samples):
        prompt_config = random.choice(TOOL_CALL_PROMPTS_EXPANDED)
        raw_text, _ = _generate_with_tools(
            base_url, tokenizer, prompt_config, temperature, max_new_tokens
        )

        # Check the gate
        gate_result = parser.has_tool_call(raw_text)

        # Force-parse regardless of gate
        try:
            _, call_info_list = parser.parse_non_stream(raw_text)
            n_parsed = len(call_info_list) if call_info_list else 0
            parsed_names = [c.name for c in call_info_list] if call_info_list else []
        except Exception:
            n_parsed = 0
            parsed_names = []

        # Also check for raw markers
        has_tool_call_marker = "<tool_call>" in raw_text
        has_function_marker = "<function=" in raw_text

        # Classify
        if gate_result and n_parsed > 0:
            stats["gate_true_parse_found"] += 1
            status = "OK"
        elif gate_result and n_parsed == 0:
            stats["gate_true_parse_empty"] += 1
            status = "OK (gate=True but parse empty)"
        elif not gate_result and n_parsed > 0:
            stats["gate_false_parse_found"] += 1
            status = "**GATE MISS**"
            gate_misses.append({
                "sample": i + 1,
                "prompt": prompt_config["name"],
                "text_preview": raw_text[:300],
                "parsed_names": parsed_names,
                "has_tool_call": has_tool_call_marker,
                "has_function": has_function_marker,
            })
        else:
            stats["gate_false_no_calls"] += 1
            status = "OK (no calls)"

        print(f"  Sample {i+1:2d}/{num_samples}: gate={gate_result!s:5s} "
              f"parsed={n_parsed} names={parsed_names!s:40s} "
              f"<tool_call>={has_tool_call_marker!s:5s} "
              f"<function=={has_function_marker!s:5s} {status}")

    # Print gate miss details
    if gate_misses:
        print(f"\n  {'!'*60}")
        print(f"  GATE MISSES ({len(gate_misses)} found):")
        print(f"  {'!'*60}")
        for miss in gate_misses:
            print(f"\n  Sample {miss['sample']} ({miss['prompt']}):")
            print(f"    Parsed names: {miss['parsed_names']}")
            print(f"    has_<tool_call>: {miss['has_tool_call']}")
            print(f"    has_<function=: {miss['has_function']}")
            preview = miss['text_preview'].replace('\n', '\n    ')
            print(f"    Text preview:\n    {preview}")

    # Summary
    print(f"\n  SUMMARY:")
    total = sum(stats.values())
    for key, val in stats.items():
        pct = val / total * 100 if total > 0 else 0
        marker = " ← ROOT CAUSE?" if key == "gate_false_parse_found" and val > 0 else ""
        print(f"    {key}: {val}/{total} ({pct:.1f}%){marker}")

    if stats["gate_false_parse_found"] > 0:
        print(f"\n  VERDICT: has_tool_call() gate has "
              f"{stats['gate_false_parse_found']/total*100:.1f}% false negative rate!")
        print(f"  The gate silently drops valid tool calls in the verl-qwen TITO path.")
        print(f"  FIX: Remove gate or add fallback in chat_convert.py:394")
    else:
        print(f"\n  VERDICT: has_tool_call() gate is accurate (0 false negatives)")

    return stats


def test_verl_qwen_vs_standalone_proxy_parsing(base_url, tokenizer, tool_call_parser,
                                                 num_samples=20, temperature=1.0,
                                                 max_new_tokens=512):
    """Compare tool call parsing between verl-qwen (has gate) and standalone proxy (no gate).

    Same raw text → parse via both paths:
    - Path 1 (standalone proxy): parse_non_stream() directly (like tito_converter.py)
    - Path 2 (verl-qwen): has_tool_call() gate → parse_non_stream() (like chat_convert.py)

    Flag cases where standalone finds calls but verl-qwen doesn't.
    """
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from pydantic import TypeAdapter
    from sglang.srt.entrypoints.openai.protocol import Tool

    print(f"\n{'='*70}")
    print(f"TEST: Verl-Qwen vs Standalone Proxy Parsing ({num_samples} samples, temp={temperature})")
    print(f"{'='*70}")

    parsed_tools = TypeAdapter(list[Tool]).validate_python(TOOLS_SCHEMA)

    stats = {"both_found": 0, "both_none": 0, "standalone_only": 0, "verl_only": 0}
    divergences = []

    for i in range(num_samples):
        prompt_config = random.choice(TOOL_CALL_PROMPTS_EXPANDED)
        raw_text, gen_resp = _generate_with_tools(
            base_url, tokenizer, prompt_config, temperature, max_new_tokens
        )

        finish_reason = gen_resp.get("meta_info", {}).get("finish_reason", {})
        fr_type = finish_reason.get("type", "unknown") if isinstance(finish_reason, dict) else str(finish_reason)

        # Path 1: Standalone proxy (no gate, but checks finish_reason=="stop")
        standalone_calls = 0
        standalone_names = []
        if fr_type == "stop":
            try:
                p1 = FunctionCallParser(parsed_tools, tool_call_parser)
                _, calls = p1.parse_non_stream(raw_text)
                standalone_calls = len(calls) if calls else 0
                standalone_names = [c.name for c in calls] if calls else []
            except Exception:
                pass

        # Path 2: Verl-qwen (has_tool_call gate, no finish_reason check)
        verl_calls = 0
        verl_names = []
        p2 = FunctionCallParser(parsed_tools, tool_call_parser)
        if p2.has_tool_call(raw_text):
            try:
                _, calls = p2.parse_non_stream(raw_text)
                verl_calls = len(calls) if calls else 0
                verl_names = [c.name for c in calls] if calls else []
            except Exception:
                pass

        if standalone_calls > 0 and verl_calls > 0:
            stats["both_found"] += 1
            status = "MATCH"
        elif standalone_calls == 0 and verl_calls == 0:
            stats["both_none"] += 1
            status = "MATCH (none)"
        elif standalone_calls > 0 and verl_calls == 0:
            stats["standalone_only"] += 1
            status = "**DIVERGENCE: standalone found, verl-qwen missed**"
            divergences.append({
                "sample": i + 1, "prompt": prompt_config["name"],
                "standalone_names": standalone_names,
                "finish_reason": fr_type,
                "has_tool_call_marker": "<tool_call>" in raw_text,
                "has_function_marker": "<function=" in raw_text,
                "text_preview": raw_text[:300],
            })
        else:
            stats["verl_only"] += 1
            status = "DIVERGENCE: verl-qwen found, standalone missed (finish_reason issue)"

        print(f"  Sample {i+1:2d}/{num_samples}: standalone={standalone_calls} "
              f"verl-qwen={verl_calls} fr={fr_type:6s} {status}")

    if divergences:
        print(f"\n  {'!'*60}")
        print(f"  STANDALONE-ONLY DIVERGENCES ({len(divergences)} found):")
        print(f"  {'!'*60}")
        for d in divergences:
            print(f"\n  Sample {d['sample']} ({d['prompt']}):")
            print(f"    Standalone found: {d['standalone_names']}")
            print(f"    finish_reason: {d['finish_reason']}")
            print(f"    has_<tool_call>: {d['has_tool_call_marker']}")
            print(f"    has_<function=: {d['has_function_marker']}")
            preview = d['text_preview'].replace('\n', '\n    ')
            print(f"    Text:\n    {preview}")

    print(f"\n  SUMMARY: {stats}")
    if stats["standalone_only"] > 0:
        print(f"  VERDICT: {stats['standalone_only']} cases where standalone proxy finds tool calls")
        print(f"           but verl-qwen TITO silently drops them due to has_tool_call() gate!")
    else:
        print(f"  VERDICT: Both paths agree on all samples.")

    return stats


def test_tool_call_at_temperature(base_url, tokenizer, tool_call_parser,
                                    num_samples=20, temperature=1.0,
                                    max_new_tokens=512):
    """Statistical validation: format distribution and gate accuracy at production temp."""
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from pydantic import TypeAdapter
    from sglang.srt.entrypoints.openai.protocol import Tool

    print(f"\n{'='*70}")
    print(f"TEST: Tool Call Format at Temperature={temperature} ({num_samples} samples)")
    print(f"{'='*70}")

    parsed_tools = TypeAdapter(list[Tool]).validate_python(TOOLS_SCHEMA)

    format_counts = {"xml": 0, "json": 0, "both": 0, "none": 0}
    native_vs_tito = {"both_found": 0, "native_only": 0, "tito_only": 0, "neither": 0}

    for i in range(num_samples):
        prompt_config = random.choice(TOOL_CALL_PROMPTS_EXPANDED)
        raw_text, gen_resp = _generate_with_tools(
            base_url, tokenizer, prompt_config, temperature, max_new_tokens
        )

        # Detect format
        has_xml = "<function=" in raw_text
        has_json = '"name"' in raw_text and "<tool_call>" in raw_text and "<function=" not in raw_text
        if has_xml and has_json:
            fmt = "both"
        elif has_xml:
            fmt = "xml"
        elif has_json:
            fmt = "json"
        else:
            fmt = "none"
        format_counts[fmt] += 1

        # TITO parse (standalone proxy style — no gate)
        tito_found = 0
        try:
            p = FunctionCallParser(parsed_tools, tool_call_parser)
            _, calls = p.parse_non_stream(raw_text)
            tito_found = len(calls) if calls else 0
        except Exception:
            pass

        # Native /v1/chat/completions
        messages = [
            {"role": "system", "content": prompt_config["system"]},
            {"role": "user", "content": prompt_config["user"]},
        ]
        try:
            native_resp = requests.post(f"{base_url}/v1/chat/completions", json={
                "model": "default", "messages": messages, "tools": TOOLS_SCHEMA,
                "tool_choice": "auto", "temperature": temperature,
                "max_tokens": max_new_tokens,
            }, timeout=120).json()
            native_tc = native_resp.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
            native_found = len(native_tc)
        except Exception:
            native_found = 0

        if tito_found > 0 and native_found > 0:
            native_vs_tito["both_found"] += 1
        elif native_found > 0 and tito_found == 0:
            native_vs_tito["native_only"] += 1
        elif tito_found > 0 and native_found == 0:
            native_vs_tito["tito_only"] += 1
        else:
            native_vs_tito["neither"] += 1

        print(f"  Sample {i+1:2d}/{num_samples}: fmt={fmt:4s} "
              f"tito_parse={tito_found} native={native_found}")

    print(f"\n  Format distribution: {format_counts}")
    print(f"  Native vs TITO: {native_vs_tito}")

    return {"format_counts": format_counts, "native_vs_tito": native_vs_tito}


# ---------------------------------------------------------------------------
# Part 3: H3 Stop Token Tests
# ---------------------------------------------------------------------------

def test_im_end_in_output_ids(base_url, tokenizer, num_samples=20,
                                max_new_tokens=256):
    """Check whether /generate includes <|im_end|> in output_token_logprobs."""
    print(f"\n{'='*70}")
    print(f"TEST: <|im_end|> in /generate output_ids ({num_samples} samples)")
    print(f"{'='*70}")

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"  im_end_token_id = {im_end_id}")

    stats = {"stop_with_imend": 0, "stop_without_imend": 0,
             "length_with_imend": 0, "length_without_imend": 0, "other": 0}

    for i in range(num_samples):
        prompt_config = random.choice(TOOL_CALL_PROMPTS_EXPANDED)
        raw_text, gen_resp = _generate_with_tools(
            base_url, tokenizer, prompt_config, temperature=1.0,
            max_new_tokens=max_new_tokens
        )

        meta = gen_resp.get("meta_info", {})
        finish_reason = meta.get("finish_reason", {})
        fr_type = finish_reason.get("type", "unknown") if isinstance(finish_reason, dict) else str(finish_reason)

        output_logprobs = meta.get("output_token_logprobs", [])
        if output_logprobs:
            last_token_id = output_logprobs[-1][1]  # (logprob, token_id, ...)
            has_imend = last_token_id == im_end_id
        else:
            last_token_id = None
            has_imend = False

        if fr_type == "stop" and has_imend:
            stats["stop_with_imend"] += 1
        elif fr_type == "stop" and not has_imend:
            stats["stop_without_imend"] += 1
        elif fr_type == "length" and has_imend:
            stats["length_with_imend"] += 1
        elif fr_type == "length" and not has_imend:
            stats["length_without_imend"] += 1
        else:
            stats["other"] += 1

        print(f"  Sample {i+1:2d}/{num_samples}: fr={fr_type:6s} "
              f"last_token={last_token_id} has_imend={has_imend}")

    print(f"\n  SUMMARY: {stats}")

    total_stop = stats["stop_with_imend"] + stats["stop_without_imend"]
    if total_stop > 0:
        pct = stats["stop_with_imend"] / total_stop * 100
        print(f"  When finish_reason=stop: im_end present {pct:.0f}% ({stats['stop_with_imend']}/{total_stop})")

    all_consistent = stats["stop_without_imend"] == 0 and stats["length_with_imend"] == 0
    if all_consistent:
        print(f"  VERDICT: Consistent — stop always has im_end, length never has im_end")
    else:
        print(f"  VERDICT: INCONSISTENT — check stop_without_imend={stats['stop_without_imend']}, "
              f"length_with_imend={stats['length_with_imend']}")

    return {"stats": stats, "all_consistent": all_consistent}


def test_finish_reason_comparison(base_url, tokenizer, tool_call_parser,
                                   num_samples=20, max_new_tokens=512):
    """Compare finish_reason between TITO /generate path and native /v1/chat/completions."""
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from pydantic import TypeAdapter
    from sglang.srt.entrypoints.openai.protocol import Tool

    print(f"\n{'='*70}")
    print(f"TEST: Finish Reason Comparison — TITO vs Native ({num_samples} samples)")
    print(f"{'='*70}")

    parsed_tools = TypeAdapter(list[Tool]).validate_python(TOOLS_SCHEMA)
    mismatches = []

    for i in range(num_samples):
        prompt_config = random.choice(TOOL_CALL_PROMPTS_EXPANDED)
        messages = [
            {"role": "system", "content": prompt_config["system"]},
            {"role": "user", "content": prompt_config["user"]},
        ]

        # TITO path: /generate → detect tool calls → finish_reason
        raw_text, gen_resp = _generate_with_tools(
            base_url, tokenizer, prompt_config, temperature=0.0, max_new_tokens=max_new_tokens
        )
        meta = gen_resp.get("meta_info", {})
        fr_raw = meta.get("finish_reason", {})
        fr_raw_type = fr_raw.get("type", "unknown") if isinstance(fr_raw, dict) else str(fr_raw)

        # Check if TITO would detect tool calls (verl-qwen path)
        p = FunctionCallParser(parsed_tools, tool_call_parser)
        tito_fr = fr_raw_type
        if p.has_tool_call(raw_text) and fr_raw_type == "stop":
            tito_fr = "tool_calls"

        # Native path
        try:
            native_resp = requests.post(f"{base_url}/v1/chat/completions", json={
                "model": "default", "messages": messages, "tools": TOOLS_SCHEMA,
                "tool_choice": "auto", "temperature": 0.0, "max_tokens": max_new_tokens,
            }, timeout=120).json()
            native_fr = native_resp.get("choices", [{}])[0].get("finish_reason", "N/A")
        except Exception as e:
            native_fr = f"ERROR: {e}"

        match = tito_fr == native_fr
        if not match:
            mismatches.append({
                "sample": i + 1, "prompt": prompt_config["name"],
                "tito_fr": tito_fr, "native_fr": native_fr,
                "raw_fr": fr_raw_type,
            })

        status = "MATCH" if match else "**MISMATCH**"
        print(f"  Sample {i+1:2d}/{num_samples}: tito_fr={tito_fr:12s} "
              f"native_fr={native_fr:12s} raw_fr={fr_raw_type:6s} {status}")

    print(f"\n  Mismatches: {len(mismatches)}/{num_samples}")
    if mismatches:
        for m in mismatches:
            print(f"    Sample {m['sample']} ({m['prompt']}): "
                  f"tito={m['tito_fr']} native={m['native_fr']} raw={m['raw_fr']}")

    return {"mismatches": mismatches, "all_match": len(mismatches) == 0}


# ---------------------------------------------------------------------------
# Part 3: H5 Tokenization Drift Tests
# ---------------------------------------------------------------------------

def test_tool_response_tokenization(tokenizer):
    """Test tokenization of tool response messages (role='tool') — CPU only.

    The TITO base-history-stripping uses BASE = [sys, user] as prefix.
    But actual messages include role='tool'. This may tokenize differently.
    """
    print(f"\n{'='*70}")
    print(f"TEST: Tool Response Tokenization Drift (CPU only)")
    print(f"{'='*70}")

    tools = TOOLS_SCHEMA
    converter = ChatConverter(tokenizer=tokenizer)

    test_cases = [
        {"role": "tool", "content": "file1.py\nfile2.py\nREADME.md"},
        {"role": "tool", "content": ""},
        {"role": "tool", "content": '{"error": "FileNotFoundError", "traceback": "..."}'},
        {"role": "tool", "content": "x" * 2000},  # long content
        {"role": "tool", "content": "def foo():\n    return 42\n\nclass Bar:\n    pass"},
    ]

    all_match = True
    for idx, tool_msg in enumerate(test_cases):
        # Build a conversation with this tool message
        messages = [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Run a command."},
            {"role": "assistant", "content": "I'll run ls.", "tool_calls": [{
                "id": "call_test123", "type": "function",
                "function": {"name": "execute_bash", "arguments": {"command": "ls"}}
            }]},
            {**tool_msg, "tool_call_id": "call_test123"},
        ]

        # Full tokenization
        full_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, tools=tools
        )

        # TITO incremental: tokenize first 2 messages, skip assistant, then tool msg
        items = converter.tokenize_new_messages(messages[:2], pre_msg_length=0, tools=tools)
        tito_ids = []
        for item in items:
            tito_ids.extend(item.token_ids)

        # The assistant message tokens come from model output (simulated)
        asst_items = converter.tokenize_new_messages(messages[:3], pre_msg_length=2, tools=tools)
        # Note: ChatConverter skips assistant messages (role == "assistant" → continue)
        # So asst_items should be empty

        # Tool message
        tool_items = converter.tokenize_new_messages(messages[:4], pre_msg_length=3, tools=tools)
        for item in tool_items:
            tito_ids.extend(item.token_ids)

        # Compare lengths (exact comparison requires full TITO state simulation)
        content_preview = tool_msg["content"][:50]
        print(f"\n  Case {idx+1}: role=tool, content={content_preview!r}...")
        print(f"    Full template: {len(full_ids)} tokens")
        print(f"    TITO incremental (partial): {len(tito_ids)} tokens (sys+user+tool only)")

        # Check if tool message tokens appear in the full template
        if tool_items:
            tool_token_ids = tool_items[0].token_ids
            # Find these tokens in full_ids
            tool_str = tokenizer.decode(tool_token_ids)
            full_str = tokenizer.decode(full_ids)
            if tool_str in full_str:
                print(f"    Tool message tokens found in full template: MATCH")
            else:
                print(f"    Tool message tokens NOT found in full template: DRIFT!")
                print(f"    Tool decoded: {tool_str[:100]!r}")
                all_match = False
        else:
            print(f"    No tool items returned (skipped by converter)")

    print(f"\n  VERDICT: {'All match' if all_match else 'DRIFT DETECTED'}")
    return {"all_match": all_match}


# ---------------------------------------------------------------------------
# Part 3: Runner for all hypothesis tests
# ---------------------------------------------------------------------------

def run_h1_deep_tests(base_url, tokenizer, tool_call_parser, num_samples, temperature):
    """Run all H1 deep investigation tests."""
    print(f"\n{'#'*70}")
    print(f"# H1 DEEP INVESTIGATION: has_tool_call() Gate Accuracy")
    print(f"# parser={tool_call_parser}, samples={num_samples}, temp={temperature}")
    print(f"{'#'*70}")

    # A1: Print token
    print(f"\n  --- A1: tool_call_start_token diagnostic ---")
    print_tool_call_start_token(tool_call_parser)

    # A2: Gate accuracy (THE key test)
    gate_stats = test_has_tool_call_gate_accuracy(
        base_url, tokenizer, tool_call_parser,
        num_samples=num_samples, temperature=temperature,
    )

    # A5: Verl-qwen vs standalone comparison
    path_stats = test_verl_qwen_vs_standalone_proxy_parsing(
        base_url, tokenizer, tool_call_parser,
        num_samples=num_samples, temperature=temperature,
    )

    # A3: Temperature statistics
    temp_stats = test_tool_call_at_temperature(
        base_url, tokenizer, tool_call_parser,
        num_samples=num_samples, temperature=temperature,
    )

    # Final summary
    print(f"\n{'='*70}")
    print(f"H1 DEEP INVESTIGATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Gate false negatives: {gate_stats['gate_false_parse_found']}")
    print(f"  Standalone-only divergences: {path_stats['standalone_only']}")
    print(f"  Format distribution: {temp_stats['format_counts']}")
    print(f"  Native vs TITO: {temp_stats['native_vs_tito']}")

    h1_confirmed = (gate_stats["gate_false_parse_found"] > 0 or
                     path_stats["standalone_only"] > 0)
    if h1_confirmed:
        print(f"\n  *** H1 IS CONFIRMED: Tool calls are being silently dropped! ***")
    else:
        print(f"\n  H1 not confirmed with {num_samples} samples at temp={temperature}")

    return h1_confirmed


def run_h3_tests(base_url, tokenizer, tool_call_parser, num_samples):
    """Run all H3 stop token tests."""
    print(f"\n{'#'*70}")
    print(f"# H3: Stop Token / EOS Handling Tests")
    print(f"{'#'*70}")

    imend_result = test_im_end_in_output_ids(base_url, tokenizer, num_samples=num_samples)
    fr_result = test_finish_reason_comparison(
        base_url, tokenizer, tool_call_parser, num_samples=num_samples
    )

    print(f"\n{'='*70}")
    print(f"H3 SUMMARY")
    print(f"{'='*70}")
    print(f"  im_end consistent: {imend_result['all_consistent']}")
    print(f"  finish_reason all match: {fr_result['all_match']}")

    return imend_result["all_consistent"] and fr_result["all_match"]


def run_h5_tests(tokenizer):
    """Run H5 tokenization drift tests (CPU only)."""
    print(f"\n{'#'*70}")
    print(f"# H5: Tokenization Drift Tests (CPU only)")
    print(f"{'#'*70}")

    result = test_tool_response_tokenization(tokenizer)
    return result["all_match"]


# ---------------------------------------------------------------------------
# H1b: finish_reason=length gate test (real SGLang)
# ---------------------------------------------------------------------------

def test_h1b_low_max_tokens(base_url, tokenizer, tool_call_parser, num_samples=10):
    """H1b: Force finish_reason=length with very low max_tokens, check if tool
    call markers are in the text but TITO's build_response() drops them.

    This is the real SGLang version of the P0 hypothesis test.
    """
    from sglang.srt.openai_api.protocol import Tool
    from pydantic import TypeAdapter
    from sglang.srt.function_call_utils import FunctionCallParser

    print(f"\n{'#'*70}")
    print(f"# H1b: finish_reason=length Gate Test (max_tokens=50)")
    print(f"# Testing if TITO drops valid tool calls when finish_reason=length")
    print(f"{'#'*70}")

    tools_schema = [
        {"type": "function", "function": {
            "name": "execute_bash",
            "description": "Execute a bash command",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "The bash command"}
            }, "required": ["command"]},
        }},
    ]

    _, converter_with_parser, assistant_prefix_ids = make_tito_state(tokenizer)
    converter_with_parser = ChatConverter(
        tokenizer=tokenizer, tool_call_parser=tool_call_parser
    )

    h1b_confirmed_count = 0
    total_length_with_markers = 0
    total_length_finish = 0
    total_samples = 0

    for prompt_config in TOOL_CALL_PROMPTS_EXPANDED[:num_samples]:
        state, converter, prefix_ids = make_tito_state(tokenizer)
        messages = [
            {"role": "system", "content": prompt_config["system"]},
            {"role": "user", "content": prompt_config["user"]},
        ]
        items = converter.tokenize_new_messages(messages, pre_msg_length=0)
        state.add_message_items(items)
        input_ids = state.get_input_ids()

        # Use low max_tokens to force finish_reason=length
        payload = {
            "input_ids": input_ids + list(assistant_prefix_ids),
            "sampling_params": {
                "max_new_tokens": 50,  # Very low — likely truncates mid-tool-call
                "temperature": 0.0,
            },
            "return_logprob": True,
        }
        r = requests.post(f"{base_url}/generate", json=payload, timeout=120)
        r.raise_for_status()
        output = r.json()

        if "text" not in output:
            continue

        text = output["text"]
        meta_info = output.get("meta_info", {})
        fr = meta_info.get("finish_reason", {})
        fr_type = fr.get("type", "unknown") if isinstance(fr, dict) else str(fr)
        total_samples += 1

        has_tool_markers = "<tool_call>" in text or "<function=" in text

        if fr_type == "length":
            total_length_finish += 1
            if has_tool_markers:
                total_length_with_markers += 1

                # Now test: does TITO's build_response parse the tool call?
                tito_resp = converter_with_parser.build_response(
                    request_data={"model": "test", "tools": tools_schema},
                    generate_output=output,
                    text=text,
                )
                tito_tc = tito_resp["choices"][0]["message"].get("tool_calls")
                tito_fr = tito_resp["choices"][0]["finish_reason"]

                if not tito_tc:
                    h1b_confirmed_count += 1
                    print(f"  H1b CONFIRMED [{prompt_config['name']}]: "
                          f"finish_reason=length, markers present, "
                          f"tito_tool_calls=None, tito_finish_reason={tito_fr}")
                    print(f"    text[:100]={text[:100]!r}")
                else:
                    print(f"  H1b NOT triggered [{prompt_config['name']}]: "
                          f"finish_reason=length but TITO parsed tool calls anyway")

        elif fr_type == "stop" and has_tool_markers:
            print(f"  [{prompt_config['name']}]: finish_reason=stop with markers (normal case)")

    print(f"\n  Summary:")
    print(f"    Total samples: {total_samples}")
    print(f"    finish_reason=length: {total_length_finish}")
    print(f"    length + tool markers: {total_length_with_markers}")
    print(f"    H1b confirmed (tool calls dropped): {h1b_confirmed_count}")

    return {
        "total_samples": total_samples,
        "length_finish": total_length_finish,
        "length_with_markers": total_length_with_markers,
        "h1b_confirmed": h1b_confirmed_count,
    }


# ---------------------------------------------------------------------------
# H4: Text content corruption test (real SGLang)
# ---------------------------------------------------------------------------

def test_h4_parsing_fidelity(base_url, tokenizer, tool_call_parser, num_samples=10):
    """H4: Generate tool call responses through real SGLang, verify that
    parser.parse_non_stream() only removes <tool_call> markers and preserves
    all reasoning text exactly."""
    from sglang.srt.openai_api.protocol import Tool
    from pydantic import TypeAdapter
    from sglang.srt.function_call_utils import FunctionCallParser

    print(f"\n{'#'*70}")
    print(f"# H4: Text Content Corruption / Parsing Fidelity Test")
    print(f"{'#'*70}")

    tools_schema = [
        {"type": "function", "function": {
            "name": "execute_bash",
            "description": "Execute a bash command",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "The bash command"}
            }, "required": ["command"]},
        }},
        {"type": "function", "function": {
            "name": "str_replace_editor",
            "description": "Edit a file",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"},
                "path": {"type": "string"},
            }, "required": ["command", "path"]},
        }},
    ]

    parsed_tools = TypeAdapter(list[Tool]).validate_python(tools_schema)
    parser = FunctionCallParser(parsed_tools, tool_call_parser)

    corruption_count = 0
    total_with_tools = 0

    for prompt_config in TOOL_CALL_PROMPTS_EXPANDED[:num_samples]:
        state, converter, assistant_prefix_ids = make_tito_state(tokenizer)
        messages = [
            {"role": "system", "content": prompt_config["system"]},
            {"role": "user", "content": prompt_config["user"]},
        ]
        items = converter.tokenize_new_messages(messages, pre_msg_length=0)
        state.add_message_items(items)
        input_ids = state.get_input_ids()

        # Use enough tokens for a full tool call
        payload = {
            "input_ids": input_ids + list(assistant_prefix_ids),
            "sampling_params": {
                "max_new_tokens": 256,
                "temperature": 0.0,
            },
            "return_logprob": True,
        }
        r = requests.post(f"{base_url}/generate", json=payload, timeout=120)
        r.raise_for_status()
        output = r.json()

        if "text" not in output:
            continue

        text = output["text"]
        if not parser.has_tool_call(text):
            continue

        total_with_tools += 1
        original_text = text

        # Parse tool calls (this modifies text)
        parsed_text, tool_calls = parser.parse_non_stream(text)

        # Find the text before the first tool call marker
        marker_pos = original_text.find("<tool_call>")
        if marker_pos == -1:
            marker_pos = original_text.find("<function=")

        if marker_pos > 0:
            reasoning_before = original_text[:marker_pos]
            # Check if reasoning text is preserved in parsed output
            if not parsed_text.startswith(reasoning_before.rstrip()):
                corruption_count += 1
                print(f"  H4 CORRUPTION [{prompt_config['name']}]:")
                print(f"    Original reasoning: {reasoning_before[:100]!r}")
                print(f"    Parsed text start:  {parsed_text[:100]!r}")
            else:
                print(f"  [{prompt_config['name']}]: Reasoning preserved OK "
                      f"(original={len(original_text)}, parsed={len(parsed_text)}, "
                      f"diff={len(original_text)-len(parsed_text)})")

        # Verify no stray markers remain
        if "<tool_call>" in parsed_text or "</tool_call>" in parsed_text:
            corruption_count += 1
            print(f"  H4 STRAY MARKERS [{prompt_config['name']}]: "
                  f"markers remain in parsed text")

    print(f"\n  Summary:")
    print(f"    Samples with tool calls: {total_with_tools}")
    print(f"    Text corruption cases: {corruption_count}")

    return {
        "total_with_tools": total_with_tools,
        "corruption_count": corruption_count,
        "all_clean": corruption_count == 0,
    }


# ---------------------------------------------------------------------------
# H2: Parameter forwarding effect test (real SGLang)
# ---------------------------------------------------------------------------

def test_h2_param_forwarding_effect(base_url, tokenizer, num_samples=5):
    """H2: Compare output diversity when frequency_penalty is sent through TITO
    (where it's dropped) vs directly to SGLang /generate (where it takes effect).
    This is informational — documents the impact of param dropping."""
    print(f"\n{'#'*70}")
    print(f"# H2: Parameter Forwarding Effect (Informational)")
    print(f"{'#'*70}")

    state, converter, assistant_prefix_ids = make_tito_state(tokenizer)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "List the first 10 prime numbers, one per line."},
    ]
    items = converter.tokenize_new_messages(messages, pre_msg_length=0)
    state.add_message_items(items)
    input_ids = state.get_input_ids()

    results_no_penalty = []
    results_with_penalty = []

    for i in range(num_samples):
        # Without frequency_penalty (TITO default behavior)
        payload_base = {
            "input_ids": input_ids + list(assistant_prefix_ids),
            "sampling_params": {
                "max_new_tokens": 128,
                "temperature": 1.0,
            },
            "return_logprob": True,
        }
        r = requests.post(f"{base_url}/generate", json=payload_base, timeout=120)
        r.raise_for_status()
        text_no_penalty = r.json().get("text", "")
        results_no_penalty.append(text_no_penalty)

        # With frequency_penalty=2.0 (dropped by TITO, but SGLang /generate supports it)
        payload_penalty = {
            "input_ids": input_ids + list(assistant_prefix_ids),
            "sampling_params": {
                "max_new_tokens": 128,
                "temperature": 1.0,
                "frequency_penalty": 2.0,
            },
            "return_logprob": True,
        }
        r = requests.post(f"{base_url}/generate", json=payload_penalty, timeout=120)
        r.raise_for_status()
        text_with_penalty = r.json().get("text", "")
        results_with_penalty.append(text_with_penalty)

    # Compare: average unique token ratio (higher = more diverse)
    def unique_token_ratio(texts):
        ratios = []
        for t in texts:
            tokens = tokenizer.encode(t, add_special_tokens=False)
            if len(tokens) > 0:
                ratios.append(len(set(tokens)) / len(tokens))
        return sum(ratios) / len(ratios) if ratios else 0

    ratio_no_penalty = unique_token_ratio(results_no_penalty)
    ratio_with_penalty = unique_token_ratio(results_with_penalty)

    print(f"\n  Unique token ratio (higher = more diverse):")
    print(f"    Without frequency_penalty: {ratio_no_penalty:.3f}")
    print(f"    With frequency_penalty=2.0: {ratio_with_penalty:.3f}")
    print(f"    Difference: {ratio_with_penalty - ratio_no_penalty:+.3f}")
    print(f"\n  Note: TITO drops frequency_penalty, so TITO users always get")
    print(f"  the 'without penalty' behavior regardless of what the agent sends.")

    return {
        "ratio_no_penalty": ratio_no_penalty,
        "ratio_with_penalty": ratio_with_penalty,
    }


def run_h1b_tests(base_url, tokenizer, tool_call_parser, num_samples):
    """Run H1b finish_reason=length gate tests."""
    result = test_h1b_low_max_tokens(base_url, tokenizer, tool_call_parser, num_samples)
    h1b_is_bug = result["h1b_confirmed"] > 0
    if h1b_is_bug:
        print(f"\n  *** H1b CONFIRMED: {result['h1b_confirmed']} tool calls dropped "
              f"due to finish_reason=length gate ***")
    else:
        if result["length_with_markers"] == 0:
            print(f"\n  H1b inconclusive: no samples had finish_reason=length with tool markers")
            print(f"  (Try increasing --num-samples or raising temperature)")
        else:
            print(f"\n  H1b NOT confirmed: tool calls were parsed despite finish_reason=length")
    return h1b_is_bug


def run_h4_tests(base_url, tokenizer, tool_call_parser, num_samples):
    """Run H4 text corruption tests."""
    result = test_h4_parsing_fidelity(base_url, tokenizer, tool_call_parser, num_samples)
    return result["all_clean"]


def run_h2_tests(base_url, tokenizer, num_samples):
    """Run H2 parameter forwarding effect tests."""
    test_h2_param_forwarding_effect(base_url, tokenizer, num_samples)
    return True  # Informational only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare TITO vs non-TITO tokenization via real SGLang")
    parser.add_argument("--model-path", default="/data/checkpoints/Qwen3-Coder-30B-A3B-Instruct",
                        help="Path to model weights")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        help="HF model name for tokenizer")
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--skip-launch", action="store_true", help="Connect to existing server")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens per turn")
    parser.add_argument("--num-turns", type=int, default=3, help="Number of assistant turns per multi-turn test")
    parser.add_argument("--tool-call-parser", default="qwen3_coder",
                        help="Parser name for TITO tool call detection (qwen3_coder, qwen25)")
    parser.add_argument("--run-tool-call-tests", action="store_true",
                        help="Run tool call hypothesis tests (H1 root cause verification)")
    parser.add_argument("--tool-call-only", action="store_true",
                        help="Run ONLY tool call hypothesis tests (skip tokenization tests)")
    # Part 3: Deep hypothesis tests
    parser.add_argument("--run-h1-deep", action="store_true",
                        help="Run H1 deep investigation (gate accuracy, temp=1.0)")
    parser.add_argument("--run-h3", action="store_true",
                        help="Run H3 stop token tests (im_end, finish_reason)")
    parser.add_argument("--run-h5", action="store_true",
                        help="Run H5 tokenization drift tests (CPU only)")
    parser.add_argument("--run-h1b", action="store_true",
                        help="Run H1b: finish_reason=length gate test (low max_tokens)")
    parser.add_argument("--run-h4", action="store_true",
                        help="Run H4: text content corruption / parsing fidelity test")
    parser.add_argument("--run-h2", action="store_true",
                        help="Run H2: parameter forwarding effect test (informational)")
    parser.add_argument("--run-all-hypotheses", action="store_true",
                        help="Run ALL hypothesis tests (H1 deep + H1b + H2 + H3 + H4 + H5)")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of samples for statistical tests")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for H1 tests (default=1.0 matches production)")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    server_proc = None

    def cleanup(signum=None, frame=None):
        if server_proc and server_proc.is_alive():
            print("\nCleaning up: terminating server...")
            server_proc.terminate()
            server_proc.join(timeout=10)
            if server_proc.is_alive():
                server_proc.kill()
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        print(f"[1/5] Loading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        print(f"       Vocab size: {tokenizer.vocab_size}")

        if not args.skip_launch:
            print(f"[2/5] Launching SGLang server on port {args.port} with TP={args.tp_size}...")
            server_proc = launch_server(args.model_path, args.tp_size, args.port, args.tool_call_parser)
        else:
            print(f"[2/5] Connecting to existing server at {base_url}...")

        print(f"       Waiting for server...")
        wait_for_server(base_url)
        print(f"       Server is healthy!")

        results = []

        # ---- Tool call hypothesis tests (optional, run with --run-tool-call-tests or --tool-call-only) ----
        if args.run_tool_call_tests or args.tool_call_only:
            tc_max_tokens = max(args.max_new_tokens, 256)  # need enough tokens for tool calls
            tc_results = run_tool_call_hypothesis_tests(
                base_url, tokenizer, args.tool_call_parser, tc_max_tokens
            )
            for cat, name, p in tc_results:
                results.append((f"hypothesis-{cat}", name, p))

        # ---- Part 3: Deep hypothesis tests ----
        run_h1 = args.run_h1_deep or args.run_all_hypotheses
        run_h1b = args.run_h1b or args.run_all_hypotheses
        run_h2 = args.run_h2 or args.run_all_hypotheses
        run_h3 = args.run_h3 or args.run_all_hypotheses
        run_h4 = args.run_h4 or args.run_all_hypotheses
        run_h5 = args.run_h5 or args.run_all_hypotheses

        if run_h1:
            h1_confirmed = run_h1_deep_tests(
                base_url, tokenizer, args.tool_call_parser,
                args.num_samples, args.temperature
            )
            results.append(("h1-deep", "H1: has_tool_call gate accuracy", not h1_confirmed))

        if run_h1b:
            h1b_confirmed = run_h1b_tests(
                base_url, tokenizer, args.tool_call_parser, args.num_samples
            )
            results.append(("h1b", "H1b: finish_reason=length gate drops tool calls", not h1b_confirmed))

        if run_h2:
            h2_ok = run_h2_tests(base_url, tokenizer, args.num_samples)
            results.append(("h2", "H2: Parameter forwarding effect (informational)", h2_ok))

        if run_h3:
            h3_ok = run_h3_tests(
                base_url, tokenizer, args.tool_call_parser, args.num_samples
            )
            results.append(("h3", "H3: Stop token handling", h3_ok))

        if run_h4:
            h4_ok = run_h4_tests(
                base_url, tokenizer, args.tool_call_parser, args.num_samples
            )
            results.append(("h4", "H4: Text content corruption", h4_ok))

        if run_h5:
            h5_ok = run_h5_tests(tokenizer)
            results.append(("h5", "H5: Tokenization drift", h5_ok))

        # If any hypothesis-only mode, exit after hypothesis tests
        if args.tool_call_only or run_h1 or run_h1b or run_h2 or run_h3 or run_h4 or run_h5:
            if not (args.run_tool_call_tests and not args.tool_call_only):
                # Print summary and exit
                print(f"\n{'='*70}")
                n_passed = sum(1 for _, _, p in results if p)
                print(f"HYPOTHESIS TEST SUMMARY: {n_passed}/{len(results)} tests passed")
                print(f"{'='*70}")
                for cat, name, p in results:
                    status = "PASS" if p else "FAIL"
                    print(f"  [{status}] {name}")
                sys.exit(0 if n_passed == len(results) else 1)

        # ---- Category 1: Multi-turn user↔assistant ----
        print(f"\n[3/5] Category 1: Multi-turn user↔assistant conversations")
        multi_turn_cases = [
            ("Multi-turn: coding question", "You are a helpful coding assistant.",
             "Write a Python function to compute fibonacci numbers."),
            ("Multi-turn: code with classes", "You are a coding assistant.",
             "Write a Python class with __init__, __repr__, and __eq__ methods."),
            ("Multi-turn: special chars", "You are a helpful assistant.",
             "Explain: for all x ∈ ℝ, f(x) → y. Include code with →, λ, and ∀."),
        ]

        for test_name, system_msg, user_msg in multi_turn_cases:
            result = do_multi_turn_generate(
                base_url, tokenizer, system_msg, user_msg,
                num_turns=args.num_turns, max_new_tokens=args.max_new_tokens,
            )
            passed = compare_tokens(
                test_name, result["messages"],
                result["tito_tokens"], result["tito_loss_mask"],
                result["tito_response_length"], tokenizer,
            )
            results.append(("multi-turn", test_name, passed))

        # ---- Category 2: Single turn (simplest case) ----
        print(f"\n[3/5] Category 2: Single-turn baseline")
        result = do_single_turn(
            base_url, tokenizer,
            "You are a helpful assistant.", "Say hello.",
            max_new_tokens=args.max_new_tokens,
        )
        passed = compare_tokens(
            "Single turn: hello", result["messages"],
            result["tito_tokens"], result["tito_loss_mask"],
            result["tito_response_length"], tokenizer,
        )
        results.append(("single-turn", "Single turn: hello", passed))

        # ---- Category 3: Tool calling ----
        print(f"\n[4/5] Category 3: Tool-calling conversations")
        result = do_tool_call_generate(
            base_url, tokenizer,
            "You are a SWE agent. Use execute_bash to run commands.",
            "Check if there are any Python syntax errors in src/.",
            num_tool_rounds=2, max_new_tokens=args.max_new_tokens,
        )
        passed = compare_tokens(
            "Tool calling: 2 rounds bash", result["messages"],
            result["tito_tokens"], result["tito_loss_mask"],
            result["tito_response_length"], tokenizer,
            tools=result.get("tools"),
        )
        results.append(("tool-call", "Tool calling: 2 rounds bash", passed))

        result = do_multiple_tool_calls_single_turn(
            base_url, tokenizer, max_new_tokens=args.max_new_tokens,
        )
        passed = compare_tokens(
            "Tool calling: multiple tool responses", result["messages"],
            result["tito_tokens"], result["tito_loss_mask"],
            result["tito_response_length"], tokenizer,
            tools=result.get("tools"),
        )
        results.append(("tool-call", "Tool calling: multiple tool responses", passed))

        # ---- Category 4: Mixed conversation ----
        print(f"\n[4/5] Category 4: Mixed conversation (tool calls + user follow-ups)")
        result = do_mixed_conversation(
            base_url, tokenizer, max_new_tokens=args.max_new_tokens,
        )
        passed = compare_tokens(
            "Mixed: assistant → tool → assistant → user → assistant",
            result["messages"],
            result["tito_tokens"], result["tito_loss_mask"],
            result["tito_response_length"], tokenizer,
            tools=result.get("tools"),
        )
        results.append(("mixed", "Mixed conversation", passed))

        # ---- Summary ----
        print(f"\n{'='*70}")
        n_passed = sum(1 for _, _, p in results if p)
        print(f"[5/5] SUMMARY: {n_passed}/{len(results)} tests passed")
        print(f"{'='*70}")

        by_category = {}
        for cat, name, p in results:
            by_category.setdefault(cat, []).append((name, p))

        for cat, items in by_category.items():
            cat_passed = sum(1 for _, p in items if p)
            status = "PASS" if cat_passed == len(items) else "FAIL"
            print(f"  [{status}] {cat}: {cat_passed}/{len(items)}")
            for name, p in items:
                print(f"         {'✓' if p else '✗'} {name}")

        if n_passed == len(results):
            print("\nAll tokenization paths produce identical results!")
        else:
            print("\nWARNING: Token differences found between TITO and non-TITO paths!")
        print(f"{'='*70}")

        sys.exit(0 if n_passed == len(results) else 1)

    finally:
        cleanup()


if __name__ == "__main__":
    main()
