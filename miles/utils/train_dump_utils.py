import logging
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import all_gather_with_cp
from miles.backends.training_utils.parallel import ParallelState

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
            ),
            path,
        )


def save_logprob_comparison_data(
    args,
    *,
    rollout_id: int,
    rollout_data,
    parallel_state: ParallelState,
    tokenizer=None,
) -> None:
    path_template = getattr(args, "save_logprob_comparison_data", None)
    if path_template is None:
        return

    if not parallel_state.is_pp_last_stage or parallel_state.tp_rank != 0:
        return

    if "log_probs" not in rollout_data or "rollout_log_probs" not in rollout_data:
        return

    rank = dist.get_rank()
    path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
    logger.info(f"Save logprob comparison data to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    num_local_samples = len(rollout_data["tokens"])
    max_seq_lens = rollout_data.get("max_seq_lens")
    sample_indices = rollout_data.get("sample_indices")
    rewards = rollout_data.get("rewards")
    raw_rewards = rollout_data.get("raw_reward")
    if raw_rewards is not None and len(raw_rewards) != num_local_samples:
        raw_rewards = None

    with torch.no_grad():
        for i, (tokens, response_length, loss_mask, rollout_log_probs, megatron_log_probs) in enumerate(
            zip(
                rollout_data["tokens"],
                rollout_data["response_lengths"],
                rollout_data["loss_masks"],
                rollout_data["rollout_log_probs"],
                rollout_data["log_probs"],
                strict=False,
            )
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            gathered_rollout_log_probs = all_gather_with_cp(
                rollout_log_probs,
                rollout_data["total_lengths"][i],
                response_length,
                parallel_state,
                args.qkv_format,
                max_seq_len,
            ).detach().cpu()
            gathered_megatron_log_probs = all_gather_with_cp(
                megatron_log_probs,
                rollout_data["total_lengths"][i],
                response_length,
                parallel_state,
                args.qkv_format,
                max_seq_len,
            ).detach().cpu()

            tokens_list = tokens.detach().cpu().tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
            response_token_ids = tokens_list[-response_length:]
            loss_mask_tensor = loss_mask.detach().cpu()

            aligned_length = min(
                response_length,
                len(response_token_ids),
                loss_mask_tensor.numel(),
                gathered_rollout_log_probs.numel(),
                gathered_megatron_log_probs.numel(),
            )
            if aligned_length != response_length:
                logger.warning(
                    "Truncating logprob comparison sample %s in rollout %s to %s tokens "
                    "(response_length=%s, response_tokens=%s, loss_mask=%s, rollout_log_probs=%s, megatron_log_probs=%s)",
                    i,
                    rollout_id,
                    aligned_length,
                    response_length,
                    len(response_token_ids),
                    loss_mask_tensor.numel(),
                    gathered_rollout_log_probs.numel(),
                    gathered_megatron_log_probs.numel(),
                )

            response_token_ids = response_token_ids[:aligned_length]
            loss_mask_tensor = loss_mask_tensor[:aligned_length].to(dtype=torch.int)
            gathered_rollout_log_probs = gathered_rollout_log_probs[:aligned_length].float()
            gathered_megatron_log_probs = gathered_megatron_log_probs[:aligned_length].float()
            abs_diff = (gathered_megatron_log_probs - gathered_rollout_log_probs).abs()
            content_mask = loss_mask_tensor.bool()

            response_token_texts = None
            response_text = None
            if tokenizer is not None:
                response_token_texts = [
                    tokenizer.decode([token_id], skip_special_tokens=False) for token_id in response_token_ids
                ]
                response_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)

            samples.append(
                {
                    "sample_index": sample_indices[i] if sample_indices is not None else None,
                    "reward": rewards[i] if rewards is not None else None,
                    "raw_reward": raw_rewards[i] if raw_rewards is not None else None,
                    "total_length": len(tokens_list),
                    "response_length": aligned_length,
                    "tokens": tokens_list,
                    "response_token_ids": response_token_ids,
                    "response_token_texts": response_token_texts,
                    "response_text": response_text,
                    "loss_mask": loss_mask_tensor.tolist(),
                    "rollout_log_probs": gathered_rollout_log_probs.tolist(),
                    "megatron_log_probs": gathered_megatron_log_probs.tolist(),
                    "abs_diff": abs_diff.tolist(),
                    "content_abs_diff_mean": float(abs_diff[content_mask].mean().item()) if content_mask.any() else 0.0,
                    "content_abs_diff_max": float(abs_diff[content_mask].max().item()) if content_mask.any() else 0.0,
                }
            )

    if parallel_state.cp_rank == 0:
        torch.save(
            {
                "rollout_id": rollout_id,
                "rank": rank,
                "dp_rank": parallel_state.dp_rank,
                "cp_rank": parallel_state.cp_rank,
                "tp_rank": parallel_state.tp_rank,
                "samples": samples,
            },
            path,
        )
