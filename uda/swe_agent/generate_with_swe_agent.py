import logging
import json
import asyncio
import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.sglang_rollout import GenerateState, eval_rollout
from miles.utils.async_utils import run
from miles.utils.http_utils import post
from miles.utils.types import Sample
from uda.swe_agent.proxy import TITOProxy

logger = logging.getLogger(__name__)

_tito_proxy = None

def _get_tito_proxy(args, tokenizer):
    global _tito_proxy
    if _tito_proxy is None:
        _tito_proxy = TITOProxy(
            sglang_base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tokenizer = tokenizer,
            args = args,
        )
    return _tito_proxy


async def _get_sglang_logprobs(args, tokens: list[int], response_length: int) -> list[float]:
    """Post-hoc SGLang logprob capture for non-TITO monitoring.

    Sends the finalized token sequence to SGLang /generate with max_new_tokens=0
    and return_logprob=True. Returns per-token logprobs for response tokens only.
    Pattern from: examples/on_policy_distillation/on_policy_distillation.py
    """
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "input_ids": tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    output = await post(url, payload)

    # input_token_logprobs: list of (logprob, token_id) tuples, one per input token
    # First token has no logprob (no prior context), skip it → [1:]
    input_logprobs = output["meta_info"].get("input_token_logprobs", [])
    all_logprobs = [item[0] for item in input_logprobs[1:]]

    # Slice to response-only (last response_length tokens)
    response_logprobs = all_logprobs[-response_length:] if response_length > 0 else []

    logger.info(
        f"[MONITOR] SGLang logprob capture: total_tokens={len(tokens)} "
        f"input_logprobs_len={len(input_logprobs)} "
        f"response_logprobs_len={len(response_logprobs)}"
    )
    return response_logprobs
    
def _fix_tools_to_match_sglang(tools: list[dict]) -> list[dict]:
    """Transform tools the same way sglang does.

    sglang's serving_chat.py calls ``Tool.model_dump()`` on each tool,
    which preserves the OpenAI wrapper ``{"type": "function", "function": {...}}``
    and adds ``strict=False`` (Pydantic default) while stripping unknown fields
    like ``cache_control``.  We replicate this so training-side tokenization
    matches rollout exactly.
    """
    out = []
    for tool in tools:
        fn = tool.get("function", tool)
        sglang_fn = {"name": fn["name"]}
        if fn.get("description") is not None:
            sglang_fn["description"] = fn["description"]
        if fn.get("parameters") is not None:
            sglang_fn["parameters"] = fn["parameters"]
        sglang_fn["strict"] = fn.get("strict", False)
        out.append({"type": "function", "function": sglang_fn})
    return out


def _normalize_tool_calls(messages: list[dict]) -> list[dict]:
    """Normalize tool_calls so arguments is always a dict."""
    out = []
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            out.append(msg)
            continue

        normalized = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tc = {**tc, "function": {**fn, "arguments": args}}
            normalized.append(tc)

        out.append({**msg, "tool_calls": normalized})
    return out


def build_tokens_and_mask_from_messages(
    args: Namespace,
    messages: list[dict],
    tokenizer,
    tools: list[dict] | None = None,
    max_user_turn_tokens: int = 20000,
) -> tuple[list[int], list[int], str, int]:
    """Build token sequence and loss mask from chat messages in O(n) time.

    Uses a constant base conversation offset to extract each message's formatted
    text independently, avoiding O(n^2) repeated full-conversation tokenization.

    For ChatML-style templates (e.g., Qwen), each message is independently formatted
    as ``<|im_start|>role\\ncontent<|im_end|>\\n``, so per-message extraction and
    tokenization produces the same result as full-conversation tokenization.

    Handles assistant messages with tool_calls and tool response messages.
    """
    if not messages or len(messages) <= 2:
        return [], [], "", 0

    # Strip trailing user/tool messages so the last message is assistant
    while messages and messages[-1].get("role") in ("user", "tool"):
        messages = messages[:-1]

    if not messages:
        return [], [], "", 0

    if len(messages) <= 2:
        raise ValueError(f"Messages must be at least 3 messages: {messages}")

    messages = _normalize_tool_calls(messages)

    # Ensure content is never None (Jinja2 template uses str concatenation)
    for msg in messages:
        if msg.get("content") is None:
            msg["content"] = ""

    # Truncate user/tool message content to max_user_turn_tokens
    for msg in messages:
        if msg.get("role") in ("user", "tool"):
            content = msg.get("content", "") or ""
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            if len(content_tokens) > max_user_turn_tokens:
                logger.warning(f"Truncating user/tool message content to {max_user_turn_tokens} tokens")
                logger.warning(f"Content: {content}")
                msg["content"] = tokenizer.decode(
                    content_tokens[:max_user_turn_tokens], skip_special_tokens=False
                )

    # Tokenize prompt (first 2 messages: system + user)
    prompt_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=False, tools=tools
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Pre-compute base conversation string offset for O(1) per-message extraction.
    # By formatting [BASE + single_msg] then slicing off the constant BASE prefix,
    # we get just the new message's formatted text to tokenize.
    _BASE = [{"role": "system", "content": "."}, {"role": "user", "content": "."}]
    base_offset = len(
        tokenizer.apply_chat_template(_BASE, tools=tools, add_generation_prompt=False, tokenize=False)
    )

    # Compute assistant prefix/suffix lengths for content-only loss masking.
    # prefix = <|im_start|>assistant\n, suffix = <|im_end|>\n
    _gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": "."}], tokenize=False, add_generation_prompt=True
    )
    _no_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": "."}], tokenize=False, add_generation_prompt=False
    )
    n_assistant_prefix = len(tokenizer.encode(_gen[len(_no_gen):], add_special_tokens=False))
    _empty_asst = tokenizer.apply_chat_template(
        [*_BASE, {"role": "assistant", "content": ""}],
        tools=tools, add_generation_prompt=False, tokenize=False
    )
    _empty_tokens = tokenizer.encode(_empty_asst[base_offset:], add_special_tokens=False)
    n_assistant_suffix = len(_empty_tokens) - n_assistant_prefix

    # Build response tokens and mask in O(n)
    response_tokens: list[int] = []
    loss_mask: list[int] = []
    use_fn_calling = getattr(args, "agent_use_fn_calling", False)

    for msg in messages[2:]:
        text = tokenizer.apply_chat_template(
            [*_BASE, msg], tools=tools, add_generation_prompt=False, tokenize=False
        )
        msg_tokens = tokenizer.encode(text[base_offset:], add_special_tokens=False)

        if msg.get("role") == "assistant":
            # When function calling is enabled, mask out malformed assistant
            # turns (no tool_calls) so GRPO does not reinforce bad format.
            if use_fn_calling and not msg.get("tool_calls"):
                loss_mask.extend([0] * len(msg_tokens))
            else:
                # mask=0 for ChatML prefix/suffix, mask=1 only for content tokens
                n_content = len(msg_tokens) - n_assistant_prefix - n_assistant_suffix
                loss_mask.extend(
                    [0] * n_assistant_prefix + [1] * n_content + [0] * n_assistant_suffix
                )
        else:
            loss_mask.extend([0] * len(msg_tokens))
        response_tokens.extend(msg_tokens)

    all_tokens = list(prompt_tokens) + response_tokens
    response_length = len(response_tokens)
    response_text = "".join([m.get("content", "") or "" for m in messages[2:]])

    if len(all_tokens) >= args.sglang_context_length:
        logger.warning(f"Sample is too long: {len(all_tokens)} tokens")
        logger.warning(f"messages: {messages}")

    return all_tokens, loss_mask, response_text, response_length


def _resolve(metadata: dict, args: Namespace, key: str, arg_name: str, default):
    """Resolve a parameter with priority: metadata > args > default."""
    if key in metadata:
        return metadata[key]
    val = getattr(args, arg_name, None)
    if val is not None:
        return val
    return default


def build_request(args: Namespace, sample: Sample, sampling_params: dict[str, Any], rollout_id: int = 0) -> dict:

    DEFAULT_STEP_TIMEOUT = 600
    DEFAULT_EVAL_TIMEOUT = 600
    DEFAULT_ENV_TIMEOUT = 120
    DEFAULT_CREATE_TIMEOUT = 600
    DEFAULT_MAX_ITERATIONS = 100
    DEFAULT_TASK_TIMEOUT = 1800

    runtime_env_env_vars = {}
    enroot_cache_path = os.environ.get("ENROOT_CACHE_PATH")
    if enroot_cache_path:
        runtime_env_env_vars["ENROOT_CACHE_PATH"] = enroot_cache_path

    runtime_env = {"env_vars": runtime_env_env_vars}
    metadata = sample.metadata

    if getattr(args, "tito", False):
        proxy = _get_tito_proxy(args, GenerateState(args).tokenizer)
        base_url = proxy.base_url
        api_key = f"tito-{metadata['instance_id']}-{sample.index}"
    else:
        base_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"
        api_key = metadata.get("api_key", "abc-123")


    payload = {
        "instance_id": metadata["instance_id"],
        "task_timeout_s": _resolve(metadata, args, "task_timeout", "agent_task_timeout", DEFAULT_TASK_TIMEOUT),
        "model_name": getattr(args, "model_name", None) or os.path.basename(args.hf_checkpoint.rstrip("/")),
        "run_name": f"{args.wandb_experiment_name}-step_{rollout_id}",
        "base_url": base_url,
        "api_key": api_key,
        "env_type": metadata.get("env_type", "enroot"),
        "sampling_params": sampling_params,
        "runtime_env": runtime_env,
        "runner": _resolve(metadata, args, "runner", "agent_runner", "oh-core"),
        "runner_entrypoint": _resolve(metadata, args, "runner_entrypoint", "agent_runner_entrypoint", "run_oh_core"),
        "task_type": metadata.get("task_type", "swe"),
        "extra_args": {
            "instance_id": metadata["instance_id"],
            "dataset": metadata["dataset"],
            "split": metadata["split"],
            "step_timeout": _resolve(metadata, args, "step_timeout", "agent_step_timeout", DEFAULT_STEP_TIMEOUT),
            "eval_timeout": _resolve(metadata, args, "eval_timeout", "agent_eval_timeout", DEFAULT_EVAL_TIMEOUT),
            "env_timeout": _resolve(metadata, args, "env_timeout", "agent_env_timeout", DEFAULT_ENV_TIMEOUT),
            "create_timeout": _resolve(metadata, args, "create_timeout", "agent_create_timeout", DEFAULT_CREATE_TIMEOUT),
            "max_iterations": _resolve(metadata, args, "max_iterations", "agent_max_iterations", DEFAULT_MAX_ITERATIONS),
            # r2e-gym special
            "use_fn_calling": _resolve(metadata, args, "use_fn_calling", "agent_use_fn_calling", True),
        },
    }
    return payload


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """
    Custom generation function for SWE-Agent integration.

    Orchestrates the interaction with the external Gym environment:
    1. Sends prompt/metadata to Gym.
    2. Receives execution trace (messages) and rewards.
    3. Formats data for Miles training format.

    Note: Performs in-place modification of `sample` for memory efficiency.
    """

    # mocked messages for failed response
    mocked_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    def build_failed_response() -> dict:
        return {
            "messages": mocked_messages,
            "tools": [],
            "reward": 0.0,
            "exit_status": "error",
            "agent_metrics": {},
        }

    # Prepare request for Gym /run endpoint
    state = GenerateState(args)
    rollout_id = getattr(state, "rollout_id", 0)
    payload = build_request(args, sample, sampling_params, rollout_id)

    gym_url = os.getenv("SWE_AGENT_GYM_URL")
    assert gym_url, "SWE_AGENT_GYM_URL is not set"
    try:
        response = await asyncio.wait_for(post(f"{gym_url}/run", payload), timeout=None)
    except Exception as e:
        logger.warning(f"SWE-Agent /run failed: {e}")
        response = build_failed_response()

    exit_status = response.get("exit_status", "unknown")
    logger.debug(f"exit_status: {exit_status}, reward: {response.get('reward', 0.0)}")

    messages = response.get("messages", [])
    if not messages or len(messages) < 3:
        messages = mocked_messages
        exit_status = "error"

    tools = response.get("tools")
    if tools:
        tools = _fix_tools_to_match_sglang(tools)

    # Diagnostic: count tool calls in messages
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]
    asst_with_tc = sum(1 for m in asst_msgs if m.get("tool_calls"))
    asst_with_marker = sum(1 for m in asst_msgs if "<tool_call>" in (m.get("content") or ""))
    mode = "TITO" if getattr(args, "tito", False) else "non-TITO"

    instance_id = sample.metadata.get("instance_id", "unknown")
    task_id = f"{instance_id}-{sample.index}"

    if len(messages) >= 2:
        sample.prompt = messages[:2]

    tito_data = None
    if getattr(args, "tito", False):
        task_id = f"tito-{instance_id}-{sample.index}"
        proxy = _get_tito_proxy(args, state.tokenizer)
        tito_data = proxy.get_task_result(task_id)

        if tito_data is None:
            # Agent failed before making any LLM calls — build fallback data
            logger.warning(f"[TITO] No state for task={task_id}, building fallback from messages")
            tokens, loss_mask, response_text, response_length = build_tokens_and_mask_from_messages(
                args=args, messages=messages, tokenizer=state.tokenizer, tools=tools,
            )
            tito_data = {
                "tokens": tokens,
                "loss_mask": loss_mask,
                "rollout_log_probs": [0.0] * response_length,
                "rollout_routed_experts": None,
                "response": response_text,
                "response_length": response_length,
            }

    if tito_data is not None:
        logger.info(
            f"[TITO] task={task_id} "
            f"tokens={len(tito_data['tokens'])} "
            f"logprobs={len(tito_data['rollout_log_probs'])} "
            f"loss_mask_sum={sum(tito_data['loss_mask'])} "
            f"response_len={tito_data['response_length']} "
            f"has_experts={tito_data['rollout_routed_experts'] is not None}"
        )
        sample.tokens = tito_data["tokens"]
        sample.loss_mask = tito_data["loss_mask"]
        sample.rollout_log_probs = tito_data["rollout_log_probs"]
        sample.rollout_routed_experts = tito_data["rollout_routed_experts"]
        sample.response = tito_data["response"]
        sample.response_length = tito_data["response_length"]

        # H5: cross-compare TITO tokens with re-tokenized tokens
        if messages and len(messages) > 2:
            try:
                retok_tokens, retok_mask, _, retok_resp_len = build_tokens_and_mask_from_messages(
                    args=args, messages=messages, tokenizer=state.tokenizer, tools=tools,
                )
                logger.info(
                    f"[H5-COMPARE] task={task_id} "
                    f"tito_tokens={len(tito_data['tokens'])} retok_tokens={len(retok_tokens)} "
                    f"diff={len(tito_data['tokens']) - len(retok_tokens)} "
                    f"tito_mask_sum={sum(tito_data['loss_mask'])} retok_mask_sum={sum(retok_mask)} "
                    f"tito_resp_len={tito_data['response_length']} retok_resp_len={retok_resp_len}"
                )
                # Dump both token sequences for mismatch analysis (per-sample file to avoid write conflicts)
                save_root = getattr(args, "save", None)
                if save_root:
                    h5_compare_dir = os.path.join(save_root, "h5_compare")
                    os.makedirs(h5_compare_dir, exist_ok=True)
                    safe_task_id = str(task_id).replace("/", "_")
                    h5_path = os.path.join(h5_compare_dir, f"{safe_task_id}_r{rollout_id}.json")
                    record = {
                        "task_id": task_id,
                        "rollout_id": rollout_id,
                        "tito_tokens": tito_data["tokens"],
                        "retok_tokens": retok_tokens,
                        "tito_mask": tito_data["loss_mask"],
                        "retok_mask": retok_mask,
                        "tito_resp_len": tito_data["response_length"],
                        "retok_resp_len": retok_resp_len,
                    }
                    with open(h5_path, "w") as f:
                        json.dump(record, f)
                    logger.info(f"[H5-COMPARE] saved to {h5_path}")
            except Exception as e:
                logger.warning(f"[H5-COMPARE] failed: {e}")
    else:
        # Non-TITO path only
        tokens, loss_mask, response_text, response_length = build_tokens_and_mask_from_messages(
            args=args,
            messages=messages,
            tokenizer=state.tokenizer,
            tools=tools,
        )
        sample.rollout_log_probs = None
        sample.tokens = tokens
        sample.loss_mask = loss_mask
        sample.response = response_text
        sample.response_length = response_length

    # Non-TITO logprob monitoring: capture SGLang logprobs post-hoc
    if getattr(args, "monitor_logprob_diff", False) and not getattr(args, "tito", False):
        if sample.response_length > 0:
            try:
                sglang_logprobs = await _get_sglang_logprobs(args, sample.tokens, sample.response_length)
                sample.rollout_log_probs = sglang_logprobs
            except Exception as e:
                logger.warning(f"[MONITOR] SGLang logprob capture failed: {e}")

    filter_overlong = getattr(args, "agent_filter_overlong", True)
    truncation_penalty = getattr(args, "agent_truncation_penalty", 0.0)
    if exit_status not in ("finished"):
        if not filter_overlong and exit_status in ("max_length", "max_iterations"):
            # keep truncated samples for training
            sample.remove_sample = False
        elif truncation_penalty and exit_status in ("max_length", "max_iterations"):
            # keep truncated samples for training with penalty reward
            sample.remove_sample = False
        else:
            # mask other failed samples (e.g. error)
            sample.remove_sample = True

    agent_metrics = response.get("agent_metrics", {})

    if exit_status not in ("submitted", ""):
        last_msg = messages[-1] if messages else {}
        last_content_preview = str(last_msg.get("content") or "")[:300]
        logger.warning(
            f"[TASK-OUTCOME] mode={mode} task={task_id} "
            f"exit_status={exit_status} "
            f"turns={agent_metrics.get('turns', 0)} "
            f"tool_calls_made={agent_metrics.get('tool_calls', 0)} "
            f"total_msgs={len(messages)} "
            f"last_role={last_msg.get('role', 'none')} "
            f"last_content_preview={repr(last_content_preview)}"
        )
        save_dir = os.environ.get("SAVE_FAILED_TRAJ_DIR")
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            traj_path = os.path.join(save_dir, f"{task_id}.json")
            try:
                with open(traj_path, "w") as f:
                    json.dump({
                        "task_id": task_id,
                        "exit_status": exit_status,
                        "mode": mode,
                        "agent_metrics": agent_metrics,
                        "messages": messages,
                    }, f, indent=2)
                logger.info(f"[TRAJ-SAVED] {traj_path}")
            except Exception as e:
                logger.warning(f"[TRAJ-SAVE-FAILED] {e}")

    sample.metadata["reward"] = response.get("reward", 0.0)
    sample.metadata["eval_report"] = response.get("metadata", {})
    sample.metadata["messages"] = messages
    sample.metadata["exit_status"] = exit_status

    sample.metadata["agent_metrics"] = agent_metrics
    sample.metadata["exit_status"] = exit_status

    # Save diagnostic metadata for per-task comparison
    sample.metadata["diag"] = {
        "assistant_msgs": len(asst_msgs),
        "tool_call_msgs": asst_with_tc,
        "tc_marker_in_content": asst_with_marker,
        "mode": mode,
    }

    logger.info(
        f"[DIAG-TOOLCALLS] mode={mode} task={task_id} "
        f"assistant_msgs={len(asst_msgs)} with_tool_calls={asst_with_tc} "
        f"with_tc_marker_in_content={asst_with_marker}"
    )

    sample.status = Sample.Status.COMPLETED

    solved = (response.get("reward", 0.0) or 0.0) > 0
    logger.info(
        f"[EVAL-SAMPLE] mode={mode} task={task_id} "
        f"{'SOLVED' if solved else 'FAILED'} "
        f"reward={sample.metadata.get('reward', 0.0)} exit_status={exit_status} "
        f"response_len={sample.response_length} "
        f"tokens_len={len(sample.tokens) if sample.tokens else 0} "
        f"loss_mask_sum={sum(sample.loss_mask) if sample.loss_mask else 0} "
        f"agent_turns={agent_metrics.get('turns', 'N/A')} "
        f"agent_tool_calls={agent_metrics.get('tool_calls', 'N/A')}"
    )

    return sample


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward function - already computed in generate()"""
    reward = sample.metadata.get("reward", 0.0)
    exit_status = sample.metadata.get("exit_status", "")
    truncation_penalty = getattr(args, "agent_truncation_penalty", 0.0)
    if truncation_penalty and exit_status in ("max_length", "max_iterations"):
        reward += truncation_penalty
    return reward


def dynamic_filter(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Filter out groups with any aborted samples from training"""
    # has_aborted = any(sample.status == Sample.Status.ABORTED for sample in samples)
    # if has_aborted:
    #     return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging"""
    metrics = {}

    all_metrics = []
    success_metrics = []  # reward == 1
    failure_metrics = []  # reward == 0 (or not 1)
    exit_statuses = []
    for sample in samples:
        if hasattr(sample, "metadata") and sample.metadata:
            agent_metrics = sample.metadata.get("agent_metrics", {})
            if agent_metrics:
                all_metrics.append(agent_metrics)
                reward = sample.metadata.get("reward", 0.0)
                if reward == 1.0:
                    success_metrics.append(agent_metrics)
                else:
                    failure_metrics.append(agent_metrics)

            exit_status = sample.metadata.get("exit_status")
            if exit_status:
                exit_statuses.append(exit_status)

    error_count = sum(1 for status in exit_statuses if str(status).lower() == "error")
    max_length_count = sum(1 for status in exit_statuses if str(status).lower() == "max_length")
    max_iterations_count = sum(1 for status in exit_statuses if str(status).lower() == "max_iterations")
    finished_count = sum(1 for status in exit_statuses if str(status).lower() == "finished")
    exit_status_count = len(exit_statuses)

    metrics["agent/exit_status_count"] = exit_status_count
    metrics["agent/error_count"] = error_count
    metrics["agent/max_length_count"] = max_length_count
    metrics["agent/max_iterations_count"] = max_iterations_count
    metrics["agent/finished_count"] = finished_count

    if not all_metrics:
        return metrics

    # Count metrics - mean and sum
    for key in ["turns", "tool_calls"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    # Iteration stats split by reward outcome
    for key in ["turns", "tool_calls"]:
        success_values = [m.get(key, 0) for m in success_metrics]
        if success_values:
            metrics[f"agent/{key}_mean_reward1"] = sum(success_values) / len(success_values)
        failure_values = [m.get(key, 0) for m in failure_metrics]
        if failure_values:
            metrics[f"agent/{key}_mean_reward0"] = sum(failure_values) / len(failure_values)

    # Time sum metrics - mean across rollouts
    for key in ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    # Time avg metrics - mean of means
    for key in ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    # Ratio metrics (all based on total_time which includes eval)
    for key in ["model_time_ratio", "env_time_ratio", "eval_time_ratio"]:
        values = [m.get(key, 0) for m in all_metrics]
        if values:
            metrics[f"agent/{key}"] = sum(values) / len(values)

    # Total time stats
    values = [m.get("total_time", 0) for m in all_metrics]
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    return metrics


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """
    Custom rollout function that wraps sglang_rollout.generate_rollout_async
    and adds agent metrics aggregation.
    """
    from miles.rollout.sglang_rollout import generate_rollout_async as base_generate_rollout_async

    state = GenerateState(args)
    state.rollout_id = rollout_id

    rollout_output, aborted_samples = await base_generate_rollout_async(args, rollout_id, data_source)

    all_samples = []
    for group in rollout_output.samples:
        if isinstance(group[0], list):
            for sample_list in group:
                all_samples.extend(sample_list)
        else:
            all_samples.extend(group)

    agent_metrics = aggregate_agent_metrics(all_samples)

    metrics = rollout_output.metrics or {}
    metrics.update(agent_metrics)

    logger.info(f"Aggregated agent metrics for rollout {rollout_id}: {agent_metrics}")

    return RolloutFnTrainOutput(samples=rollout_output.samples, metrics=metrics), aborted_samples


def generate_rollout(
    args: Namespace, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    output, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return output


def generate_abortable_samples(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    evaluation: bool = False,
) -> tuple[Any, list[list[Sample]]]:
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source))
