from dataclasses import dataclass, field
import logging
import numpy as np

_logger = logging.getLogger(__name__)

@dataclass
class MessageItem:
    index: int                              # position in the messages list
    role: str                               # "system" / "user" / "assistant" / "tool"
    token_ids: list[int]                    # tokenized content for this message
    logprobs: list[float] | None = None     # per-token logprobs (assistant only)
    routed_experts: np.ndarray | None = None  # [num_tokens, num_layers, topk] (assistant only)


class TaskState:
    def __init__(self, new_line_token_id: int,
                 assistant_prefix_ids: list[int] | None = None,
                 im_end_token_id: int | None = None):
        self.messages: list[MessageItem] = []
        self.new_line_token_id = new_line_token_id
        self.assistant_prefix_ids = assistant_prefix_ids or []
        self.im_end_token_id = im_end_token_id

    def add_message_items(self, msg_items: list[MessageItem]):
        self.messages.extend(msg_items)

    def get_num_recorded_messages(self) -> int:
        return len(self.messages)

    def add_response(self, token_ids: list[int], logprobs: list[float], routed_experts: np.ndarray | None = None):
        # Wrap with ChatML: <|im_start|>assistant\n + content + <|im_end|>\n
        # SGLang may include the stop token (<|im_end|>) in output_token_logprobs — strip it
        # before wrapping so we don't get a double <|im_end|>.
        if self.im_end_token_id is not None and token_ids and token_ids[-1] == self.im_end_token_id:
            token_ids = token_ids[:-1]
            logprobs = logprobs[:-1]
            if routed_experts is not None:
                routed_experts = routed_experts[:-1]
            _logger.debug(f"[TITO-STOP] Stripped im_end token. remaining_len={len(token_ids)}")
        elif self.im_end_token_id is not None and token_ids:
            _logger.warning(
                f"[TITO-STOP-MISS] im_end NOT found at end. "
                f"last_token={token_ids[-1]} expected={self.im_end_token_id} "
                f"token_ids_len={len(token_ids)}"
            )

        suffix_ids = []
        if self.im_end_token_id is not None:
            suffix_ids.append(self.im_end_token_id)
        suffix_ids.append(self.new_line_token_id)

        wrapped_ids = list(self.assistant_prefix_ids) + token_ids + suffix_ids
        n_prefix = len(self.assistant_prefix_ids)
        n_suffix = len(suffix_ids)
        wrapped_logprobs = [0.0] * n_prefix + logprobs + [0.0] * n_suffix

        # Pad routed_experts with zeros for added prefix/suffix tokens
        if routed_experts is not None:
            pad_shape = list(routed_experts.shape)
            pad_shape[0] = n_prefix
            prefix_pad = np.zeros(pad_shape, dtype=routed_experts.dtype)
            pad_shape[0] = n_suffix
            suffix_pad = np.zeros(pad_shape, dtype=routed_experts.dtype)
            routed_experts = np.concatenate([prefix_pad, routed_experts, suffix_pad], axis=0)

        item = MessageItem(
            index=len(self.messages),
            role="assistant",
            token_ids=wrapped_ids,
            logprobs=wrapped_logprobs,
            routed_experts=routed_experts,
        )
        self.messages.append(item)

    def get_input_ids(self) -> list[int]:
        input_ids = []
        for item in self.messages:
            input_ids.extend(item.token_ids)
        return input_ids
    
    def get_routed_experts_length(self) -> int:
        length = sum(len(item.token_ids) for item in self.messages
            if item.routed_experts is not None)
        _logger.debug(f"[TITO-STATE] get_routed_experts_length={length} total_messages={len(self.messages)}")
        return length

    def finalize(self) -> dict:
        # Find the first assistant message
        first_assistant_idx = None
        for i, item in enumerate(self.messages):
            if item.role == "assistant":
                first_assistant_idx = i
                break

        if first_assistant_idx is None:
            all_ids = []
            for item in self.messages:
                all_ids.extend(item.token_ids)
            
            return {
                "tokens": all_ids,
                "loss_mask": [],
                "rollout_log_probs": [],
                "rollout_routed_experts": None,
                "response": "",
                "response_length": 0,
            }
        
        # Prompt = everything before first assistant
        prompt_ids = []
        for item in self.messages[:first_assistant_idx]:
            prompt_ids.extend(item.token_ids)
        
        # Response = everything after first assistant
        response_ids = []
        loss_mask = []
        logprobs = []
        routed_experts_list = []

        for item in self.messages[first_assistant_idx:]:
            response_ids.extend(item.token_ids)

            if item.role == "assistant" and item.logprobs is not None:
                # mask=0 for prefix (<|im_start|>assistant\n) and suffix (<|im_end|>\n)
                # mask=1 only for model-generated content tokens
                n_pre = len(self.assistant_prefix_ids)
                n_suf = 2 if self.im_end_token_id is not None else 1
                n_content = len(item.token_ids) - n_pre - n_suf
                loss_mask.extend([0] * n_pre + [1] * n_content + [0] * n_suf)
                logprobs.extend(item.logprobs)
                if item.routed_experts is not None:
                    routed_experts_list.append(item.routed_experts)
            else:
                loss_mask.extend([0] * len(item.token_ids))
                logprobs.extend([0.0] * len(item.token_ids))

        all_tokens = prompt_ids + response_ids

        if routed_experts_list:
            all_routed_experts = np.concatenate(routed_experts_list, axis=0)
        else:
            all_routed_experts = None

        _logger.debug(
            f"[TITO-STATE] finalize: total_messages={len(self.messages)} "
            f"first_assistant_idx={first_assistant_idx} "
            f"routed_experts_list_len={len(routed_experts_list)} "
            f"all_routed_experts_shape={all_routed_experts.shape if all_routed_experts is not None else None} "
            f"tokens_len={len(all_tokens)} response_len={len(response_ids)} "
            f"loss_mask_sum={sum(loss_mask)}"
        )

        return {
            "tokens": all_tokens,
            "loss_mask": loss_mask,
            "rollout_log_probs": logprobs,
            "rollout_routed_experts": all_routed_experts,
            "response": "",  # will be overwritten or unused
            "response_length": len(response_ids),
        }