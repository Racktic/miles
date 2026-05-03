"""Compare logprobs between sglang (inference) and megatron (training).

Verifies training-inference consistency by computing logprobs on the same
trajectory tokens with both megatron and sglang, then comparing the results.

Usage:
    # Megatron vs sglang (requires torchrun + miles training config)
    # Pass the SAME config you use for training, plus extra comparison args.
    torchrun --nproc_per_node=4 tools/compare_logprobs.py \
        --config /path/to/training_config.yaml \
        --trajectory-dir /path/to/results/exp-step_0 \
        --sglang-url http://localhost:30000 \
        --compare-num-samples 5 \
        --compare-max-response-tokens 2048

    # Megatron only (compute and save megatron logprobs, no sglang needed)
    torchrun --nproc_per_node=4 tools/compare_logprobs.py \
        --config /path/to/training_config.yaml \
        --trajectory-dir /path/to/results/exp-step_0 \
        --compare-mode megatron-only \
        --compare-num-samples 5

    # Compare with saved sglang logprobs file
    torchrun --nproc_per_node=4 tools/compare_logprobs.py \
        --config /path/to/training_config.yaml \
        --trajectory-dir /path/to/results/exp-step_0 \
        --sglang-logprobs-file /path/to/sglang_logprobs.pt
"""

import json
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Trajectory loading & tokenization
#    (reuses logic from miles/uda/swe_agent/generate_with_swe_agent.py)
# ---------------------------------------------------------------------------

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


def build_tokens_and_mask(
    messages: list[dict],
    tokenizer,
    tools: list[dict] | None = None,
    max_user_turn_tokens: int = 20000,
    context_length: int = 65536,
) -> tuple[list[int], list[int], int]:
    """Tokenize messages and build loss mask.

    Returns (tokens, loss_mask, response_length).
    Adapted from generate_with_swe_agent.py:build_tokens_and_mask_from_messages.
    """
    if not messages:
        return [], [], 0

    while messages and messages[-1].get("role") in ("user", "tool"):
        messages = messages[:-1]
    if not messages or len(messages) <= 2:
        return [], [], 0

    messages = _normalize_tool_calls(messages)
    for msg in messages:
        if msg.get("content") is None:
            msg["content"] = ""

    for msg in messages:
        if msg.get("role") in ("user", "tool"):
            content = msg.get("content", "") or ""
            content_tokens = tokenizer.encode(content, add_special_tokens=False)
            if len(content_tokens) > max_user_turn_tokens:
                msg["content"] = tokenizer.decode(
                    content_tokens[:max_user_turn_tokens], skip_special_tokens=False
                )

    prompt_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=False, tools=tools
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    _BASE = [{"role": "system", "content": "."}, {"role": "user", "content": "."}]
    base_offset = len(
        tokenizer.apply_chat_template(_BASE, tools=tools, add_generation_prompt=False, tokenize=False)
    )

    response_tokens: list[int] = []
    loss_mask: list[int] = []

    for msg in messages[2:]:
        text = tokenizer.apply_chat_template(
            [*_BASE, msg], tools=tools, add_generation_prompt=False, tokenize=False
        )
        msg_tokens = tokenizer.encode(text[base_offset:], add_special_tokens=False)
        mask_val = 1 if msg.get("role") == "assistant" else 0
        loss_mask.extend([mask_val] * len(msg_tokens))
        response_tokens.extend(msg_tokens)

    all_tokens = list(prompt_tokens) + response_tokens
    response_length = len(response_tokens)

    if len(all_tokens) >= context_length:
        logger.warning(f"Sample too long: {len(all_tokens)} tokens >= {context_length}")

    return all_tokens, loss_mask, response_length


def load_trajectories(trajectory_dir: str, num_samples: int = -1) -> list[dict]:
    """Load trajectory.json files from the step directory."""
    traj_dir = Path(trajectory_dir)
    trajectories = []

    for traj_file in sorted(traj_dir.rglob("trajectory.json")):
        with open(traj_file) as f:
            data = json.load(f)
        if not data.get("messages") or len(data["messages"]) <= 2:
            continue
        rollout_hash = traj_file.parent.name
        data["_path"] = str(traj_file)
        data["_rollout_hash"] = rollout_hash
        trajectories.append(data)
        if num_samples > 0 and len(trajectories) >= num_samples:
            break

    logger.info(f"Loaded {len(trajectories)} trajectories from {trajectory_dir}")
    return trajectories


def tokenize_trajectories(
    trajectories: list[dict],
    tokenizer,
    context_length: int = 65536,
    max_response_tokens: int = -1,
) -> list[dict]:
    """Tokenize all trajectories."""
    results = []
    for traj in trajectories:
        messages = traj["messages"]
        tools = traj.get("tools")

        tokens, loss_mask, response_length = build_tokens_and_mask(
            messages, tokenizer, tools=tools, context_length=context_length,
        )
        if not tokens or response_length == 0:
            logger.warning(f"Skipping empty trajectory: {traj.get('_path', 'unknown')}")
            continue

        total_length = len(tokens)

        if max_response_tokens > 0 and response_length > max_response_tokens:
            prompt_length = total_length - response_length
            keep = prompt_length + max_response_tokens
            tokens = tokens[:keep]
            loss_mask = loss_mask[:max_response_tokens]
            response_length = max_response_tokens
            total_length = len(tokens)

        instance_id = traj.get("instance_id", "unknown")
        rollout_hash = traj.get("_rollout_hash", "")
        uid = f"{instance_id}/{rollout_hash}" if rollout_hash else instance_id

        results.append({
            "tokens": tokens,
            "loss_mask": loss_mask,
            "response_length": response_length,
            "total_length": total_length,
            "instance_id": instance_id,
            "uid": uid,
            "_path": traj.get("_path", ""),
        })

    logger.info(f"Tokenized {len(results)} trajectories (skipped {len(trajectories) - len(results)})")
    return results


# ---------------------------------------------------------------------------
# 2. Megatron logprob computation (using miles infrastructure)
# ---------------------------------------------------------------------------

def compute_megatron_logprobs(
    args: Namespace,
    model,
    parallel_state,
    tokenized_samples: list[dict],
) -> dict[str, torch.Tensor]:
    """Compute logprobs using megatron's forward_only pipeline.

    This replicates the exact code path used during miles RL training:
    actor.compute_log_prob() -> forward_only() -> get_log_probs_and_entropy()
    """
    from miles.backends.megatron_utils.model import forward_only
    from miles.backends.training_utils.data import DataIterator
    from miles.backends.training_utils.loss import get_log_probs_and_entropy

    results = {}

    for i, sample in enumerate(tokenized_samples):
        uid = sample["uid"]
        logger.info(
            f"[{i+1}/{len(tokenized_samples)}] Computing megatron logprobs for {uid} "
            f"(total={sample['total_length']}, resp={sample['response_length']})"
        )

        # Build rollout_data dict matching what get_rollout_data() produces
        tokens_tensor = torch.tensor(sample["tokens"], dtype=torch.long, device=torch.cuda.current_device())
        loss_mask_tensor = torch.tensor(sample["loss_mask"], dtype=torch.int, device=torch.cuda.current_device())

        rollout_data = {
            "tokens": [tokens_tensor],
            "loss_masks": [loss_mask_tensor],
            "total_lengths": [sample["total_length"]],
            "response_lengths": [sample["response_length"]],
        }

        if args.qkv_format == "bshd":
            pad_size = parallel_state.tp_size * args.data_pad_size_multiplier
            max_seq_len = (sample["total_length"] + pad_size - 1) // pad_size * pad_size
            rollout_data["max_seq_lens"] = [max_seq_len]

        # Create DataIterator (micro_batch_size=1 since we process one sample at a time)
        data_iterator = [DataIterator(rollout_data, micro_batch_size=1)]
        num_microbatches = [1]

        # Run megatron forward_only — same call path as actor.compute_log_prob()
        output = forward_only(
            get_log_probs_and_entropy,
            args,
            model,
            data_iterator,
            num_microbatches,
            parallel_state,
            store_prefix="",
        )

        if "log_probs" in output and output["log_probs"]:
            log_probs = output["log_probs"][0]  # first (only) sample
            results[uid] = log_probs.cpu().float()
            logger.info(f"  megatron logprobs: mean={log_probs.mean():.4f}, shape={log_probs.shape}")
        else:
            # Not on last PP stage — logprobs only available there
            logger.info(f"  (not on last PP stage, no logprobs returned)")

    return results


# ---------------------------------------------------------------------------
# 3. sglang logprob computation (via running server)
# ---------------------------------------------------------------------------

def compute_sglang_logprobs_via_server(
    tokens: list[int],
    response_length: int,
    sglang_url: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute logprobs by sending the full token sequence to a sglang server.

    Uses sglang's /generate endpoint with input_ids and logprob_start_len
    to get logprobs for the response tokens without generating new tokens.
    """
    import requests

    prompt_length = len(tokens) - response_length

    payload = {
        "input_ids": tokens,
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": temperature,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_length - 1,
    }

    url = sglang_url.rstrip("/")
    if "/generate" not in url and "/v1" not in url:
        url = f"{url}/generate"

    logger.info(f"Requesting sglang logprobs from {url} (prompt_len={prompt_length}, total_len={len(tokens)})")
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    result = resp.json()

    meta_info = result.get("meta_info", {})

    input_token_logprobs = meta_info.get("input_token_logprobs", [])
    if input_token_logprobs:
        logprobs = []
        for item in input_token_logprobs:
            if isinstance(item, (list, tuple)):
                logprobs.append(item[0])
            else:
                logprobs.append(item)
        logprobs = logprobs[:response_length]
        if len(logprobs) < response_length:
            logger.warning(
                f"sglang returned {len(logprobs)} input logprobs, expected {response_length}. "
                "Padding with NaN."
            )
            logprobs.extend([float("nan")] * (response_length - len(logprobs)))
        return torch.tensor(logprobs, dtype=torch.float32)

    output_token_logprobs = meta_info.get("output_token_logprobs", [])
    if output_token_logprobs:
        logprobs = [item[0] if isinstance(item, (list, tuple)) else item for item in output_token_logprobs]
        return torch.tensor(logprobs[:response_length], dtype=torch.float32)

    logger.error(f"sglang returned no logprobs. meta_info keys: {list(meta_info.keys())}")
    return torch.full((response_length,), float("nan"))


def load_sglang_logprobs_from_file(filepath: str) -> dict[str, torch.Tensor]:
    """Load saved sglang logprobs from a .pt file.

    Expected formats:
      1) miles debug rollout: torch.save({"samples": [sample.to_dict(), ...]})
      2) simple dict: torch.save({"uid_1": [logprob_values], ...})
    """
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    result = {}

    if "samples" in data:
        for sample_dict in data["samples"]:
            log_probs = sample_dict.get("rollout_log_probs")
            if log_probs is None:
                continue
            instance_id = sample_dict.get("metadata", {}).get("instance_id", "unknown")
            key = f"{instance_id}_{sample_dict.get('index', 0)}"
            result[key] = torch.tensor(log_probs, dtype=torch.float32)
    elif isinstance(data, dict):
        for key, val in data.items():
            if key in ("rollout_id",):
                continue
            if isinstance(val, (list, torch.Tensor)):
                result[key] = torch.tensor(val, dtype=torch.float32) if not isinstance(val, torch.Tensor) else val.float()

    logger.info(f"Loaded sglang logprobs for {len(result)} samples from {filepath}")
    return result


# ---------------------------------------------------------------------------
# 4. Comparison & reporting
# ---------------------------------------------------------------------------

def compare_logprobs(
    megatron_logprobs: torch.Tensor,
    sglang_logprobs: torch.Tensor,
    loss_mask: list[int] | None = None,
    label: str = "",
) -> dict:
    """Compare two sets of logprobs and compute statistics."""
    assert megatron_logprobs.shape == sglang_logprobs.shape, (
        f"Shape mismatch: {megatron_logprobs.shape} vs {sglang_logprobs.shape}"
    )

    if loss_mask is not None:
        mask = torch.tensor(loss_mask, dtype=torch.bool)
        if mask.shape[0] != megatron_logprobs.shape[0]:
            mask = mask[: megatron_logprobs.shape[0]]
        mg_masked = megatron_logprobs[mask]
        sg_masked = sglang_logprobs[mask]
    else:
        mg_masked = megatron_logprobs
        sg_masked = sglang_logprobs

    valid = ~(torch.isnan(mg_masked) | torch.isnan(sg_masked))
    mg_valid = mg_masked[valid]
    sg_valid = sg_masked[valid]

    if len(mg_valid) == 0:
        return {"label": label, "error": "no valid tokens to compare"}

    diff = (mg_valid - sg_valid).abs()

    stats = {
        "label": label,
        "num_tokens": int(len(mg_valid)),
        "num_masked_tokens": int(mask.sum()) if loss_mask is not None else int(len(megatron_logprobs)),
        "abs_diff_mean": float(diff.mean()),
        "abs_diff_max": float(diff.max()),
        "abs_diff_median": float(diff.median()),
        "abs_diff_p99": float(diff.quantile(0.99)),
        "abs_diff_std": float(diff.std()),
        "megatron_mean": float(mg_valid.mean()),
        "sglang_mean": float(sg_valid.mean()),
        "correlation": float(torch.corrcoef(torch.stack([mg_valid, sg_valid]))[0, 1]),
        "rel_diff_mean": float((diff / (mg_valid.abs() + 1e-8)).mean()),
    }

    worst_indices = diff.topk(min(5, len(diff))).indices.tolist()
    stats["worst_positions"] = [
        {
            "pos": int(idx),
            "megatron": float(mg_valid[idx]),
            "sglang": float(sg_valid[idx]),
            "diff": float(diff[idx]),
        }
        for idx in worst_indices
    ]

    return stats


def print_comparison_report(all_stats: list[dict]):
    """Print a summary comparison report."""
    print("\n" + "=" * 80)
    print("LOGPROB COMPARISON REPORT: Megatron vs sglang")
    print("=" * 80)

    for stats in all_stats:
        if "error" in stats:
            print(f"\n[{stats['label']}] ERROR: {stats['error']}")
            continue

        print(f"\n--- {stats['label']} ---")
        print(f"  Tokens compared (loss_mask=1): {stats['num_tokens']}")
        print(f"  Abs diff  mean={stats['abs_diff_mean']:.6f}  max={stats['abs_diff_max']:.6f}  "
              f"median={stats['abs_diff_median']:.6f}  p99={stats['abs_diff_p99']:.6f}")
        print(f"  Rel diff  mean={stats['rel_diff_mean']:.6f}")
        print(f"  Correlation: {stats['correlation']:.6f}")
        print(f"  Mean logprob  megatron={stats['megatron_mean']:.4f}  sglang={stats['sglang_mean']:.4f}")

        if stats.get("worst_positions"):
            print("  Top-5 worst mismatches:")
            for w in stats["worst_positions"]:
                print(f"    pos={w['pos']:>6d}  megatron={w['megatron']:.6f}  "
                      f"sglang={w['sglang']:.6f}  diff={w['diff']:.6f}")

    valid_stats = [s for s in all_stats if "error" not in s]
    if valid_stats:
        agg_mean = np.mean([s["abs_diff_mean"] for s in valid_stats])
        agg_max = np.max([s["abs_diff_max"] for s in valid_stats])
        agg_corr = np.mean([s["correlation"] for s in valid_stats])
        total_tokens = sum(s["num_tokens"] for s in valid_stats)

        print(f"\n{'=' * 80}")
        print(f"AGGREGATE ({len(valid_stats)} samples, {total_tokens} total tokens)")
        print(f"  Mean abs diff: {agg_mean:.6f}")
        print(f"  Max  abs diff: {agg_max:.6f}")
        print(f"  Mean correlation: {agg_corr:.6f}")

        if agg_mean < 1e-4:
            print("  VERDICT: MATCH (mean diff < 1e-4)")
        elif agg_mean < 1e-2:
            print("  VERDICT: CLOSE (mean diff < 1e-2, likely numerical precision)")
        else:
            print("  VERDICT: MISMATCH (mean diff >= 1e-2, investigate!)")
        print("=" * 80)


# ---------------------------------------------------------------------------
# 5. Args & main
# ---------------------------------------------------------------------------

def add_compare_args(parser):
    """Add comparison-specific args to miles' argument parser."""
    group = parser.add_argument_group("Logprob comparison")
    group.add_argument("--trajectory-dir", required=True,
                       help="Path to step directory with trajectory.json files")
    group.add_argument("--sglang-url", default=None,
                       help="sglang server URL (e.g., http://localhost:30000)")
    group.add_argument("--sglang-logprobs-file", default=None,
                       help="Path to saved sglang logprobs .pt file")
    group.add_argument("--compare-mode", default="megatron-vs-sglang",
                       choices=["megatron-vs-sglang", "megatron-only"],
                       help="Comparison mode")
    group.add_argument("--compare-num-samples", type=int, default=5,
                       help="Number of trajectory samples to compare (-1 for all)")
    group.add_argument("--compare-max-response-tokens", type=int, default=-1,
                       help="Truncate response to N tokens for faster testing")
    group.add_argument("--compare-output", default=None,
                       help="Save comparison results to JSON file")
    return parser


def main():
    # Use miles' argument parser so we get all megatron args from the training config.
    # Launch with: torchrun --nproc_per_node=<TP*PP*CP> tools/compare_logprobs.py --config ...
    from miles.utils.arguments import parse_args
    args = parse_args(add_custom_arguments=add_compare_args)

    import torch.distributed as dist

    # torchrun already initializes torch.distributed; fix up rank/world_size in args
    if dist.is_initialized():
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(args.local_rank)

    # Initialize megatron model parallelism (TP/PP/CP/EP groups)
    from miles.backends.megatron_utils.initialize import init as megatron_init
    megatron_init(args)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    # Load and tokenize trajectories (all ranks do this for data availability)
    trajectories = load_trajectories(args.trajectory_dir, num_samples=args.compare_num_samples)
    if not trajectories:
        logger.error("No trajectories found!")
        return

    context_length = getattr(args, "sglang_context_length", None) or getattr(args, "seq_length", 65536)
    tokenized = tokenize_trajectories(
        trajectories, tokenizer,
        context_length=context_length,
        max_response_tokens=args.compare_max_response_tokens,
    )
    if not tokenized:
        logger.error("No valid tokenized trajectories!")
        return

    # Load megatron model from checkpoint
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer
    from miles.backends.megatron_utils.parallel import create_megatron_parallel_state

    model, _, _, _ = initialize_model_and_optimizer(args, role="actor")
    parallel_state = create_megatron_parallel_state(model=model)

    # Compute megatron logprobs
    megatron_results = compute_megatron_logprobs(args, model, parallel_state, tokenized)

    # Only the last PP stage has logprobs; only rank 0 of that stage does comparison
    if not parallel_state.is_pp_last_stage or parallel_state.dp_rank != 0:
        logger.info("Not on last PP stage / not dp_rank 0, skipping comparison")
        return

    # Load or compute sglang logprobs
    saved_sglang_logprobs = None
    if args.sglang_logprobs_file:
        saved_sglang_logprobs = load_sglang_logprobs_from_file(args.sglang_logprobs_file)

    sglang_results = {}
    if args.compare_mode == "megatron-vs-sglang":
        if saved_sglang_logprobs:
            sglang_results = saved_sglang_logprobs
        elif args.sglang_url:
            for i, sample in enumerate(tokenized):
                uid = sample["uid"]
                logger.info(f"[{i+1}/{len(tokenized)}] Computing sglang logprobs for {uid}")
                lp = compute_sglang_logprobs_via_server(
                    sample["tokens"],
                    sample["response_length"],
                    args.sglang_url,
                    temperature=getattr(args, "rollout_temperature", 1.0),
                )
                sglang_results[uid] = lp
                logger.info(f"  sglang logprobs: mean={lp.mean():.4f}, shape={lp.shape}")
        else:
            logger.error("Need --sglang-url or --sglang-logprobs-file for comparison")
            return

    # Compare
    all_stats = []
    if args.compare_mode == "megatron-vs-sglang":
        for sample in tokenized:
            uid = sample["uid"]
            mg_lp = megatron_results.get(uid)
            sg_lp = sglang_results.get(uid)
            if mg_lp is None or sg_lp is None:
                logger.warning(f"Missing logprobs for {uid}")
                continue
            min_len = min(len(mg_lp), len(sg_lp))
            stats = compare_logprobs(
                mg_lp[:min_len], sg_lp[:min_len],
                loss_mask=sample["loss_mask"][:min_len],
                label=uid,
            )
            all_stats.append(stats)
        print_comparison_report(all_stats)

    elif args.compare_mode == "megatron-only":
        logger.info("megatron-only mode: saving computed logprobs")
        save_data = {}
        for sample in tokenized:
            uid = sample["uid"]
            if uid in megatron_results:
                save_data[uid] = {
                    "logprobs": megatron_results[uid].tolist(),
                    "tokens": sample["tokens"],
                    "loss_mask": sample["loss_mask"],
                    "response_length": sample["response_length"],
                }
        out_path = args.compare_output or "megatron_logprobs.pt"
        torch.save(save_data, out_path)
        logger.info(f"Saved megatron logprobs to {out_path}")

    # Save detailed results
    if args.compare_output and all_stats:
        with open(args.compare_output, "w") as f:
            json.dump(all_stats, f, indent=2)
        logger.info(f"Saved comparison results to {args.compare_output}")


if __name__ == "__main__":
    main()
