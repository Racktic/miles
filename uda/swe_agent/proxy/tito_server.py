import json
import socket
import threading
import logging
import asyncio

import httpx
import numpy as np
import pybase64
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .tito_state import TaskState
from .tito_converter import ChatConverter

logger = logging.getLogger(__name__)


def _normalize_tool_calls_for_template(messages: list[dict]) -> list[dict]:
    """Qwen3 template expects tool_call arguments as dicts, not JSON strings.
    The H1b fix stores arguments as JSON strings (OpenAI spec).  Before passing
    messages to apply_chat_template we convert them back to dicts so the Jinja2
    template's .items() filter doesn't crash on turn 2+.
    """
    result = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            normalized_calls = []
            for tc in msg["tool_calls"]:
                f = tc.get("function", {})
                args = f.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                normalized_calls.append({**tc, "function": {**f, "arguments": args}})
            msg = {**msg, "tool_calls": normalized_calls}
        result.append(msg)
    return result


class TITOProxy:
    def __init__(self, sglang_base_url: str, tokenizer, args):
        self.sglang_base_url = sglang_base_url  
        self.tokenizer = tokenizer
        self.args = args
        self.tasks: dict[str, TaskState] = {}
        self.converters: dict[str, ChatConverter] = {}
        
        # Compute new_line_token_id once
        self.new_line_token_id = tokenizer.encode("\n")[-1]

        # Compute assistant prefix: <|im_start|>assistant\n
        gen_prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "."}], tokenize=False, add_generation_prompt=True
        )
        no_gen_prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "."}], tokenize=False, add_generation_prompt=False
        )
        self.assistant_prefix_ids = tokenizer.encode(
            gen_prompt_text[len(no_gen_prompt_text):], add_special_tokens=False
        )
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.tool_call_parser = getattr(args, "sglang_tool_call_parser", None)
        self.host_ip = getattr(args, "sglang_router_ip", "0.0.0.0")

        self.port = self._find_free_port()
        self._start_server()

        logger.info(
            f"[TITO-INIT] sglang_base_url={sglang_base_url} "
            f"use_rollout_routing_replay={getattr(args, 'use_rollout_routing_replay', 'MISSING')} "
            f"num_layers={getattr(args, 'num_layers', 'MISSING')} "
            f"moe_router_topk={getattr(args, 'moe_router_topk', 'MISSING')}"
        )

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _start_server(self):
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self.handle_chat_completion(request)
        
        thread = threading.Thread(
            target = uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": self.port, "log_level": "warning"},
            daemon=True,
        )
        thread.start()
        logger.info(f"TITOProxy server started on port {self.port}")
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host_ip}:{self.port}/v1"
    
    async def handle_chat_completion(self, request: Request) -> JSONResponse:
        request_data = await request.json()
        messages = request_data.get("messages", [])
        tools = request_data.get("tools")

        auth = request.headers.get("Authorization", "")
        api_key = auth.replace("Bearer ", "")
        task_id = api_key

        if task_id not in self.tasks:
            self.tasks[task_id] = TaskState(
                new_line_token_id=self.new_line_token_id,
                assistant_prefix_ids=self.assistant_prefix_ids,
                im_end_token_id=self.im_end_token_id,
            )
            self.converters[task_id] = ChatConverter(
                tokenizer=self.tokenizer,
                tool_call_parser=self.tool_call_parser,
            )

        state = self.tasks[task_id]
        converter = self.converters[task_id]

        pre_msg_length = state.get_num_recorded_messages()

        # Run CPU-bound tokenization in thread pool to avoid blocking the
        # event loop (512 concurrent tasks share this single async server).
        def _tokenize():
            new_items = converter.tokenize_new_messages(messages, pre_msg_length, tools=tools)
            state.add_message_items(new_items)
            normalized_messages = _normalize_tool_calls_for_template(messages)
            input_ids = self.tokenizer.apply_chat_template(
                normalized_messages, tokenize=True, add_generation_prompt=True, tools=tools
            )
            return input_ids

        input_ids = await asyncio.to_thread(_tokenize)

        # Log dropped parameters (H2 debug)
        forwarded_keys = {"temperature", "top_p", "max_tokens", "stop"}
        all_keys = set(request_data.keys()) - {"messages", "tools", "model", "stream"}
        dropped_keys = all_keys - forwarded_keys
        if dropped_keys:
            logger.warning(
                f"[TITO-PARAM-DROP] task={task_id} dropped_params={sorted(dropped_keys)} "
                f"dropped_values={{k: request_data[k] for k in dropped_keys if k in request_data}}"
            )

        sampling_params = {
            k: v for k, v in request_data.items()
            if k in ("temperature", "top_p", "max_tokens", "stop")
        }
        max_new_tokens = sampling_params.pop("max_tokens", 4096)

        generate_payload = {
            "input_ids": input_ids,
            "sampling_params": {
                **sampling_params,
                "max_new_tokens": max_new_tokens,
            },
            "return_logprob": True,
            "return_routed_experts": getattr(self.args,
                                            "use_rollout_routing_replay", False),
            "pre_recorded_experts_length": state.get_routed_experts_length(),
        }

        logger.info(
            f"[TITO-DEBUG] task={task_id} sending /generate "
            f"input_ids_len={len(input_ids)} "
            f"sampling_params={generate_payload['sampling_params']} "
            f"payload_keys={sorted(generate_payload.keys())} "
            f"return_routed_experts={generate_payload.get('return_routed_experts')} "
            f"pre_recorded_experts_length={generate_payload.get('pre_recorded_experts_length')} "
            f"url={self.sglang_base_url}/generate"
        )

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{self.sglang_base_url}/generate",
                json=generate_payload,
            )
            output = resp.json()

        logger.info(
            f"[TITO-DEBUG] task={task_id} "
            f"meta_info_keys={sorted(output.get('meta_info', {}).keys())} "
            f"has_routed_experts={'routed_experts' in output.get('meta_info', {})} "
            f"routed_experts_type={type(output.get('meta_info', {}).get('routed_experts'))} "
            f"routed_experts_len={len(str(output.get('meta_info', {}).get('routed_experts', '')))} "
            f"payload_return_routed_experts={generate_payload.get('return_routed_experts')} "
            f"payload_pre_recorded_experts_length={generate_payload.get('pre_recorded_experts_length')} "
            f"sglang_url={self.sglang_base_url}"
        )

        # Forward SGLang errors (e.g., context length exceeded) as OpenAI-format
        if "text" not in output:
            error_msg = output.get("error", output)
            logger.warning(f"[TITO] SGLang error for {task_id}: {error_msg}")
            return JSONResponse(
                content={
                    "error": {
                        "message": str(error_msg),
                        "type": "invalid_request_error",
                    }
                },
                status_code=resp.status_code if resp.status_code >= 400 else 400,
            )

        text = output["text"]
        meta_info = output["meta_info"]

        logger.info(
            f"[TITO-RESPONSE] task={task_id} "
            f"finish_reason={meta_info.get('finish_reason')} "
            f"text_len={len(text)} "
            f"has_tool_call_marker={'<tool_call>' in text} "
            f"has_function_marker={'<function=' in text} "
            f"output_tokens={len(meta_info.get('output_token_logprobs', []))}"
        )
        
        output_token_logprobs = meta_info.get("output_token_logprobs", [])
        output_ids = [item[1] for item in output_token_logprobs]
        output_logprobs = [item[0] for item in output_token_logprobs]

        routed_experts = None
        if experts_b64 := meta_info.get("routed_experts"):
            raw = np.frombuffer(
                pybase64.b64decode(experts_b64.encode("ascii")), dtype=np.int32
            )
            num_new_tokens = len(output_ids)
            num_layers = self.args.num_layers
            topk = self.args.moe_router_topk
            routed_experts = raw.reshape(num_new_tokens, num_layers, topk)

        logger.info(
            f"[TITO-DEBUG] task={task_id} "
            f"routed_experts_shape={routed_experts.shape if routed_experts is not None else None} "
            f"num_output_ids={len(output_ids)} "
            f"num_layers={getattr(self.args, 'num_layers', 'MISSING')} "
            f"topk={getattr(self.args, 'moe_router_topk', 'MISSING')}"
        )

        state.add_response(output_ids, output_logprobs, routed_experts)

        chat_response = converter.build_response(request_data, output, text)
        return JSONResponse(content=chat_response)
    
    def get_task_result(self, task_id: str) -> dict:
        state = self.tasks.pop(task_id, None)
        self.converters.pop(task_id, None)
        
        if state is None:
            logger.warning(f"No state found for task {task_id}")
            return None
        
        return state.finalize()

