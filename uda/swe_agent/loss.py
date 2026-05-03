import torch

from argparse import Namespace

from miles.backends.training_utils.cp_utils import get_logits_and_tokens_offset_with_cp


def get_sum_of_sample_mean_dr_grpo(
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
    parallel_state=None,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
):
    # Dr.GRPO: divide by max_response_length instead of effective token count
    max_response_length = args.sglang_context_length
    cp_size = parallel_state.cp_size if parallel_state is not None else 1

    if cp_size == 1:

        def sum_of_sample_mean_dr_grpo(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum() / max_response_length
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

    else:
        cp_chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, parallel_state, qkv_format, max_seq_len
            )
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_masks.append(torch.cat([loss_mask_0, loss_mask_1], dim=0))
            cp_chunk_lengths.append(chunked_loss_masks[i].size(0))

        def sum_of_sample_mean_dr_grpo(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / max_response_length
                    for x_i, chunked_loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=False
                    )
                ]
            )

    return sum_of_sample_mean_dr_grpo
