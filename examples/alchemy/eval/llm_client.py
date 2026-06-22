"""Provider-abstracted LLM client for the Alchemy zero-shot eval.

Only the Anthropic adapter is implemented now (per the `claude-api` reference: official SDK,
`claude-opus-4-8`, prompt caching, refusal handling). OpenAI/Gemini adapters are the next step
(`make_client` raises NotImplementedError for them for now). Keys are read from `/home/qixinx/miles/.env`.
"""
from __future__ import annotations

import os
import threading

_DOTENV = "/home/qixinx/miles/.env"


def load_dotenv_keys(path: str = _DOTENV) -> None:
    """Load KEY=VALUE lines from .env into os.environ (does not overwrite already-set vars)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _with_cache_control(messages: list[dict]) -> list[dict]:
    """Mark the last user message for prompt caching so the growing chat prefix is cached
    (no-summary re-sends the whole history every turn → ~90% input-token savings)."""
    out: list[dict] = []
    for i, m in enumerate(messages):
        if i == len(messages) - 1 and m["role"] == "user":
            out.append({"role": "user", "content": [
                {"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class AnthropicClient:
    """Wraps the official `anthropic` SDK. `complete(system, messages)` -> assistant text."""

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 2048):
        import anthropic
        self.model = model
        self.max_tokens = max_tokens
        self._c = anthropic.Anthropic()                    # reads ANTHROPIC_API_KEY from env
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.refusals = 0
        self._lock = threading.Lock()                      # usage accumulation across parallel episodes

    def complete(self, system: str, messages: list[dict]) -> str:
        sys_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        resp = self._c.messages.create(                    # SDK client is safe for concurrent calls
            model=self.model, max_tokens=self.max_tokens, system=sys_blocks,
            messages=_with_cache_control(messages))
        u = resp.usage
        with self._lock:
            self.usage["input"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output"] += getattr(u, "output_tokens", 0) or 0
            self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            if resp.stop_reason == "refusal":
                self.refusals += 1
        return "" if resp.stop_reason == "refusal" else "".join(
            b.text for b in resp.content if b.type == "text")


class OpenAIClient:
    """OpenAI SDK `chat.completions` — one adapter for BOTH OpenAI cloud (GPT-5) and any
    OpenAI-compatible LOCAL endpoint (e.g. Qwen3.5-4B served by sglang) via `base_url`.

    - cloud (base_url=None): reads OPENAI_API_KEY; reasoning models (GPT-5/o-series) need
      `max_completion_tokens` (not `max_tokens`) and accept `reasoning_effort`.
    - local (base_url set, e.g. http://localhost:30000/v1): dummy key; sglang's OpenAI endpoint uses
      `max_tokens`. Prefix/prompt caching is automatic server-side for both, so no cache_control needed.
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 2048, reasoning_effort: str | None = None,
                 temperature: float | None = None, extra_body: dict | None = None):
        import openai
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.extra_body = extra_body          # e.g. {"chat_template_kwargs": {"enable_thinking": False}} for Qwen
        self._local = base_url is not None
        key = api_key or (os.environ.get("OPENAI_API_KEY") if not self._local else "EMPTY")
        self._c = openai.OpenAI(base_url=base_url, api_key=key)
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.refusals = 0
        self._lock = threading.Lock()

    def complete(self, system: str, messages: list[dict]) -> str:
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
        kw: dict = {"model": self.model, "messages": msgs}
        kw["max_tokens" if self._local else "max_completion_tokens"] = self.max_tokens
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        resp = self._c.chat.completions.create(**kw)
        u = getattr(resp, "usage", None)
        with self._lock:
            if u:
                self.usage["input"] += getattr(u, "prompt_tokens", 0) or 0
                self.usage["output"] += getattr(u, "completion_tokens", 0) or 0
                ptd = getattr(u, "prompt_tokens_details", None)        # OpenAI auto-cache hits
                self.usage["cache_read"] += (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            with self._lock:
                self.refusals += 1
            return ""
        return choice.message.content or ""


class GeminiClient:
    """Google Gemini via REST (httpx) — no SDK needed. `systemInstruction` + `contents` (assistant
    role is 'model'); response text from candidates[0].content.parts. Gemini 3 thinks adaptively by
    default; pass thinking_budget to cap it. Key from GEMINI_API_KEY."""

    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str, max_tokens: int = 2048, temperature: float | None = None,
                 thinking_budget: int | None = None):
        import httpx
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self._key = os.environ.get("GEMINI_API_KEY")
        self._http = httpx.Client(timeout=180.0)
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self.refusals = 0
        self._lock = threading.Lock()

    def complete(self, system: str, messages: list[dict]) -> str:
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in messages]
        gen: dict = {"maxOutputTokens": self.max_tokens}
        if self.temperature is not None:
            gen["temperature"] = self.temperature
        if self.thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": contents, "generationConfig": gen}
        resp = self._http.post(f"{self.BASE}/{self.model}:generateContent",
                               params={"key": self._key}, json=body)
        resp.raise_for_status()
        d = resp.json()
        um = d.get("usageMetadata", {})
        with self._lock:
            self.usage["input"] += um.get("promptTokenCount", 0) or 0
            self.usage["output"] += um.get("candidatesTokenCount", 0) or 0
            self.usage["cache_read"] += um.get("cachedContentTokenCount", 0) or 0
        cands = d.get("candidates", [])
        blocked = (not cands) or cands[0].get("finishReason") in (
            "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "RECITATION")
        if blocked:
            with self._lock:
                self.refusals += 1
            return ""
        return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))


def make_client(provider: str, model: str, max_tokens: int = 2048, base_url: str | None = None,
                reasoning_effort: str | None = None, temperature: float | None = None,
                extra_body: dict | None = None):
    """Factory. Loads .env keys first.
    provider: 'anthropic' | 'openai' (cloud GPT, or local/Qwen via base_url) | 'gemini' (REST)."""
    load_dotenv_keys()
    if provider == "anthropic":
        return AnthropicClient(model=model, max_tokens=max_tokens)
    if provider == "openai":          # Qwen3.5-4B: --provider openai --base-url http://localhost:PORT/v1
        return OpenAIClient(model=model, base_url=base_url, max_tokens=max_tokens,
                            reasoning_effort=reasoning_effort, temperature=temperature,
                            extra_body=extra_body)
    if provider == "gemini":          # reasoning_effort not directly used (Gemini thinks adaptively)
        return GeminiClient(model=model, max_tokens=max_tokens, temperature=temperature)
    raise NotImplementedError(f"provider {provider!r} not implemented")
