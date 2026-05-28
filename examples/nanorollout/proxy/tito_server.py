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
from fastapi.responses import JSONResponse, StreamingResponse

from .tito_state import TaskState
from .tito_converter import ChatConverter

logger = logging.getLogger(__name__)

# DeepResearch (Qwen3.5) agent stop markers — mirrors QWEN_STOP_STRINGS in
# nanorollout/harness/agents/deepresearch/qwen35_agent.py. Used as the default
# `stop` for the /completions path when a caller omits it.
QWEN_STOP_STRINGS = ("\n<tool_response>", "<tool_response>")


def _normalize_tool_calls_for_template(messages: list[dict]) -> list[dict]:
    """Hermes tool call template expects tool_call arguments as dicts, not JSON strings."""
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

        @app.post("/v1/completions")
        async def completions(request: Request):
            return await self.handle_completion(request)

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
        # TODO(Junli): skip re-tokenization
        def _tokenize():
            new_items = converter.tokenize_new_messages(messages, pre_msg_length, tools=tools)
            state.add_message_items(new_items)
            normalized_messages = _normalize_tool_calls_for_template(messages)
            input_ids = self.tokenizer.apply_chat_template(
                normalized_messages, tokenize=True, add_generation_prompt=True, tools=tools
            )
            return input_ids

        input_ids = await asyncio.to_thread(_tokenize)

        forwarded_keys = {"temperature", "top_p", "max_tokens", "stop"}
        all_keys = set(request_data.keys()) - {"messages", "tools", "model", "stream"}
        dropped_keys = all_keys - forwarded_keys
        if dropped_keys:
            logger.debug(
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

        logger.debug(
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

        logger.debug(
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

        text, output_ids, output_logprobs, routed_experts = self._extract_generation(output)

        state.add_response(output_ids, output_logprobs, routed_experts)

        chat_response = converter.build_response(request_data, output, text)
        return JSONResponse(content=chat_response)

    def _extract_generation(self, output: dict):
        """Pull (text, token_ids, logprobs, routed_experts) out of an SGLang
        /generate response. Shared by the chat and completions handlers."""
        text = output["text"]
        meta_info = output["meta_info"]

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

        return text, output_ids, output_logprobs, routed_experts

    def _truncate_at_stop(self, output_ids, output_logprobs, routed_experts, stop):
        """Defensive: cut captured tokens at the first stop-string occurrence.

        SGLang already stops at `stop` (and trims it) when the param is set, so
        this is normally a no-op. It guards against partial-token boundaries and
        any future caller that omits `stop`, so captured tokens == the tokens the
        agent keeps (the agent does its own client-side stop-cut). We keep the
        largest token prefix whose decoded text stays strictly before the marker
        — a token straddling the boundary is dropped (we cannot split a token).
        """
        text = self.tokenizer.decode(output_ids)
        if not stop or not output_ids:
            return output_ids, output_logprobs, routed_experts, text
        positions = [text.find(s) for s in stop]
        cut = min((p for p in positions if p != -1), default=-1)
        if cut < 0:
            return output_ids, output_logprobs, routed_experts, text
        keep = 0
        for n in range(1, len(output_ids) + 1):
            if len(self.tokenizer.decode(output_ids[:n])) <= cut:
                keep = n
            else:
                break
        logger.warning(
            "[TITO-STOP-TRUNCATE] /completions output contained an un-trimmed "
            "stop string at char %d; truncating %d→%d tokens.",
            cut, len(output_ids), keep,
        )
        rex = routed_experts[:keep] if routed_experts is not None else None
        return (
            output_ids[:keep],
            output_logprobs[:keep],
            rex,
            self.tokenizer.decode(output_ids[:keep]),
        )

    async def handle_completion(self, request: Request):
        """Token-in/token-out capture for the DeepResearch (Qwen3.5) agent, which
        uses /completions: it renders the chat template itself, sends a raw prompt
        string, and parses <tool_call> manually. We capture the exact generated
        tokens + logprobs into the same TaskState the chat path uses, so
        finalize() (and miles' loss) is unchanged. See add_completion_round."""
        request_data = await request.json()
        prompt = request_data.get("prompt", "")
        if isinstance(prompt, list):
            # OpenAI permits a list of prompts; the DeepResearch agent always
            # sends a single string. Defensively take the first.
            prompt = prompt[0] if prompt else ""

        auth = request.headers.get("Authorization", "")
        task_id = auth.replace("Bearer ", "")

        if task_id not in self.tasks:
            self.tasks[task_id] = TaskState(
                new_line_token_id=self.new_line_token_id,
                assistant_prefix_ids=self.assistant_prefix_ids,
                im_end_token_id=self.im_end_token_id,
            )
            # No ChatConverter: completions needs no message-diff-by-template.
        state = self.tasks[task_id]

        stop = request_data.get("stop") or list(QWEN_STOP_STRINGS)
        if isinstance(stop, str):
            stop = [stop]

        # Tokenize the prompt EXACTLY as the agent does (qwen35_agent._tokenize:
        # encode(prompt, add_special_tokens=False)) so the recorded prefix lines
        # up with the agent's view; the append-only canary catches any drift.
        # CPU-bound → run off the event loop.
        prompt_ids = await asyncio.to_thread(
            self.tokenizer.encode, prompt, add_special_tokens=False
        )

        sampling_params = {"stop": stop}
        if "temperature" in request_data:
            sampling_params["temperature"] = request_data["temperature"]
        if "top_p" in request_data:
            sampling_params["top_p"] = request_data["top_p"]
        max_new_tokens = request_data.get("max_tokens", 4096)

        generate_payload = {
            "input_ids": prompt_ids,
            "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
            "return_routed_experts": getattr(
                self.args, "use_rollout_routing_replay", False
            ),
            "pre_recorded_experts_length": state.get_routed_experts_length(),
        }

        logger.debug(
            f"[TITO-DEBUG] task={task_id} /completions → /generate "
            f"input_ids_len={len(prompt_ids)} "
            f"sampling_params={generate_payload['sampling_params']}"
        )

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(
                f"{self.sglang_base_url}/generate", json=generate_payload
            )
            output = resp.json()

        if "text" not in output:
            error_msg = output.get("error", output)
            logger.warning(f"[TITO] SGLang error (/completions) for {task_id}: {error_msg}")
            # Plain 500 — deliberately NOT a 400/404 "not supported" body, which
            # would trip the agent's /chat/completions fallback and bypass capture.
            return JSONResponse(
                content={"error": {"message": str(error_msg), "type": "server_error"}},
                status_code=500,
            )

        text, output_ids, output_logprobs, routed_experts = self._extract_generation(output)
        output_ids, output_logprobs, routed_experts, text = self._truncate_at_stop(
            output_ids, output_logprobs, routed_experts, stop
        )

        state.add_completion_round(prompt_ids, output_ids, output_logprobs, routed_experts)

        completion_response = {
            "object": "text_completion",
            "model": request_data.get("model", "tito"),
            "choices": [
                {"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}
            ],
        }

        async def _sse():
            yield f"data: {json.dumps(completion_response)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    def get_task_result(self, task_id: str) -> dict:
        state = self.tasks.pop(task_id, None)
        self.converters.pop(task_id, None)

        if state is None:
            logger.warning(f"No state found for task {task_id}")
            return None

        return state.finalize()

    def get_task_rounds(self, task_id: str) -> list[dict] | None:
        """Per-round samples for the /completions (DeepResearch) path.

        Pops the task state and returns one self-consistent (prompt, completion)
        training sample per round (Strategy B). Returns None if the task made no
        recorded calls, or [] if it used the chat path (no completion rounds);
        callers fall back to message-retokenization in those cases.
        """
        state = self.tasks.pop(task_id, None)
        self.converters.pop(task_id, None)

        if state is None:
            logger.warning(f"No state found for task {task_id}")
            return None

        return state.finalize_rounds()

