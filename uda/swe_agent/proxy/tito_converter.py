import json
import logging
import uuid

from pydantic import TypeAdapter
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from .tito_state import MessageItem

logger = logging.getLogger(__name__)


class ChatConverter:
    def __init__(self, tokenizer, tool_call_parser: str | None = None):
        self.tokenizer = tokenizer
        self.tool_call_parser = tool_call_parser
        self.base_prefix_len: int | None = None

    def tokenize_new_messages(self, messages: list[dict], pre_msg_length: int,
                              tools: list[dict] | None = None) -> list[MessageItem]:

        if self.base_prefix_len is None:
            _BASE = [{"role": "system", "content": "."}, {"role": "user", "content": "."}]
            base_ids = self.tokenizer.apply_chat_template(
                _BASE, tokenize=True, add_generation_prompt=False, tools=tools
                )
            self.base_prefix_len = len(base_ids)

        _BASE = [{"role": "system", "content": "."}, {"role": "user", "content": "."}]

        items = []
        new_messages = messages[pre_msg_length:]

        for i, msg in enumerate(new_messages):
            global_idx = pre_msg_length + i
            role = msg.get("role", "user")

            if role == "assistant":
                continue

            if pre_msg_length == 0 and global_idx < 2:
                if global_idx == 0:
                    full_ids = self.tokenizer.apply_chat_template(
                        messages[:2], tokenize=True, add_generation_prompt=False, tools=tools
                    )
                    sys_ids = self.tokenizer.apply_chat_template(
                        [messages[0]], tokenize=True, add_generation_prompt=False, tools=tools
                    )
                    user_ids = full_ids[len(sys_ids):]

                    items.append(MessageItem(
                        index=0, role=messages[0]["role"], token_ids=list(sys_ids)
                    ))
                    items.append(MessageItem(
                        index=1, role=messages[1]["role"], token_ids=list(user_ids)
                    ))
                    continue
                elif global_idx == 1:
                    continue
            else:
                msg_copy = {**msg}
                if msg_copy.get("content") is None:
                    msg_copy["content"] = ""

                full_ids = self.tokenizer.apply_chat_template(
                    [*_BASE, msg_copy], tokenize=True, add_generation_prompt=False, tools=tools
                )
                token_ids = full_ids[self.base_prefix_len:]

                items.append(MessageItem(
                    index=global_idx, role=role, token_ids=list(token_ids)
                ))
        return items

    def build_response(self, request_data: dict, generate_output: dict, text: str) -> dict:
        meta_info = generate_output.get("meta_info", {})
        prompt_tokens = meta_info.get("prompt_tokens", 0)
        completion_tokens = meta_info.get("completion_tokens", 0)

        finish_reason = "stop"
        if meta_info.get("finish_reason", {}).get("type") == "length":
            finish_reason = "length"

        # H1b fix: Parse tool calls based on text content (not finish_reason),
        # matching SGLang native and verl-qwen behavior.  Tool calls that were
        # fully generated before a length cutoff should still be parsed.
        tool_calls_list = None
        tools = request_data.get("tools")
        if tools and self.tool_call_parser:
            try:
                parsed_tools = TypeAdapter(list[Tool]).validate_python(tools)
                parser = FunctionCallParser(parsed_tools, self.tool_call_parser)
                original_text = text
                text, call_info_list = parser.parse_non_stream(text)

                text_changed = text != original_text
                logger.info(
                    f"[TITO-TOOLCALL] parser={self.tool_call_parser} "
                    f"found_calls={len(call_info_list) if call_info_list else 0} "
                    f"text_modified={text_changed} "
                    f"original_len={len(original_text)} parsed_len={len(text)} "
                    f"finish_reason_in={finish_reason} "
                    f"finish_reason_out={'tool_calls' if call_info_list else finish_reason}"
                )
                if call_info_list:
                    finish_reason = "tool_calls"
                    tool_calls_list = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": ci.name,
                                "arguments": ci.parameters
                                if isinstance(ci.parameters, str)
                                else json.dumps(ci.parameters)
                                if ci.parameters
                                else "{}",
                            },
                        }
                        for ci in call_info_list
                    ]
                    for ci in call_info_list:
                        logger.info(
                            f"[TITO-TOOLCALL-DETAIL] name={ci.name} "
                            f"args_type={type(ci.parameters).__name__} "
                            f"args_preview={str(ci.parameters)[:200]}"
                        )
                elif "<tool_call>" in original_text:
                    logger.warning(
                        f"[TITO-TOOLCALL-MISS] <tool_call> marker found but parser returned empty! "
                        f"text_preview={original_text[:500]}"
                    )
            except Exception as e:
                logger.warning(f"[TITO] Tool call parsing error: {e}")

        # Match verl-qwen / SGLang native: content=None (not "") when empty after
        # tool-call extraction — this is what the OpenAI spec requires.
        content = text.strip() if text.strip() else None
        message = {"role": "assistant", "content": content}
        if tool_calls_list:
            message["tool_calls"] = tool_calls_list

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": request_data.get("model", "tito-proxy"),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
