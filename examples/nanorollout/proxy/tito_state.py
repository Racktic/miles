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
        # Per-round records for the /completions path (Strategy B). Each entry is
        # one self-consistent (prompt, completion) generation; see
        # add_completion_round / finalize_rounds.
        self.completion_rounds: list[dict] = []

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

    def add_completion_round(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
        logprobs: list[float],
        routed_experts: np.ndarray | None = None,
    ):
        """Record one `/completions` round (DeepResearch text-completion path).

        Unlike the chat path, the agent sends the FULL re-rendered prompt each
        round (it applies the chat template itself) and we capture the raw
        generated completion. Under an append-only regime (context-archiving
        disabled) ``prompt_ids`` extends the previously-recorded token prefix:
        the genuinely-new prefix tokens (tool result(s) injected since the last
        round) are recorded as a masked non-assistant message, and the
        completion is recorded via :meth:`add_response` — identical wrapping and
        masking to the chat path, so :meth:`finalize` works unchanged.

        Canary: the recorded prefix must be a prefix of ``prompt_ids``. If it is
        not, the agent re-rendered earlier turns differently from their raw
        generation (e.g. stripped ``<think>``/restructured ``<tool_call>``) or
        context-archiving fired — append-only assembly is then unfaithful and a
        per-round (Strategy B) capture is required. We warn loudly rather than
        silently produce a corrupt training sequence.
        """
        # PRIMARY (Strategy B): record this round as a self-consistent
        # (prompt, completion) pair. The model saw exactly `prompt_ids` (the FULL
        # re-rendered context for this round) and generated `completion_ids` with
        # `logprobs` — so a per-round training sample has the right context for
        # its logprobs regardless of how earlier turns are re-rendered. This is
        # faithful even when the single-sequence append-only assembly is not.
        self.completion_rounds.append({
            "prompt_ids": list(prompt_ids),
            "completion_ids": list(completion_ids),
            "logprobs": list(logprobs),
            "routed_experts": routed_experts,
        })

        # DIAGNOSTIC: also build the append-only single sequence + run the canary.
        # finalize() consumes this; finalize_rounds() uses the records above. The
        # canary measures how often single-sequence assembly would be unfaithful
        # (it warns but does not block — per-round capture is unaffected).
        recorded = self.get_input_ids()
        already = len(recorded)
        if prompt_ids[:already] != recorded:
            diverge = next(
                (i for i in range(min(already, len(prompt_ids)))
                 if prompt_ids[i] != recorded[i]),
                min(already, len(prompt_ids)),
            )
            _logger.warning(
                "[TITO-APPEND] prompt prefix diverged from recorded at token %d "
                "(recorded_len=%d prompt_len=%d). Append-only assembly is "
                "UNFAITHFUL for this round — context-archiving fired or assistant "
                "re-rendering differs from raw generation; Strategy-B per-round "
                "capture is needed before training on this trajectory.",
                diverge, already, len(prompt_ids),
            )

        # New non-assistant prefix = tokens added since the last round, EXCLUDING
        # the trailing assistant generation prefix (`add_response` re-adds it).
        n_pre = len(self.assistant_prefix_ids)
        prefix_end = len(prompt_ids) - n_pre
        new_prefix = prompt_ids[already:prefix_end] if prefix_end > already else []
        if new_prefix:
            self.messages.append(
                MessageItem(
                    index=len(self.messages),
                    role="user",
                    token_ids=list(new_prefix),
                )
            )
        self.add_response(completion_ids, logprobs, routed_experts)

    def finalize_rounds(self) -> list[dict]:
        """Per-round training samples for the /completions path (Strategy B).

        Returns one dict per `/completions` round, each a self-consistent
        training example: ``tokens = prompt_ids + completion_ids`` with a
        response-length ``loss_mask`` (all 1s — every generated token is content
        the policy produced) and the captured ``rollout_log_probs``. Because each
        round carries its OWN full prompt, the logprobs match the training-time
        context exactly — no cross-round re-render mismatch (unlike the single
        append-only sequence from :meth:`finalize`). Same dict shape as
        :meth:`finalize` so each fanned-out sample is consumed identically.

        Routing-replay (`rollout_routed_experts`) is not supported per-round yet:
        Sample.validate() expects one entry per (token-1) over prompt+completion,
        but we only capture experts for the completion. We drop it (None) and warn
        if present; Qwen3.5-4B is dense so this does not arise in practice.
        """
        rounds: list[dict] = []
        for rec in self.completion_rounds:
            completion_ids = list(rec["completion_ids"])
            n = len(completion_ids)
            loss_mask = [1] * n
            # Match the chat/append-only convention: keep the generated end-of-turn
            # marker in `tokens` but don't put loss on it (it's the ChatML wrapper,
            # not content). SGLang sometimes includes <|im_end|> as the last token.
            if self.im_end_token_id is not None and n > 0 and completion_ids[-1] == self.im_end_token_id:
                loss_mask[-1] = 0
            if rec.get("routed_experts") is not None:
                _logger.warning(
                    "[TITO-ROUNDS] dropping routed_experts in per-round capture "
                    "(routing-replay unsupported in Strategy-B mode)."
                )
            rounds.append({
                "tokens": list(rec["prompt_ids"]) + completion_ids,
                "loss_mask": loss_mask,
                "rollout_log_probs": list(rec["logprobs"]),
                "rollout_routed_experts": None,
                "response": "",
                "response_length": n,
            })
        return rounds

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