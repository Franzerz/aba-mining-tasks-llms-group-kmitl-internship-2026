#### NOT USE ####

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str


class LLMClient:
    def complete(self, *, model: str, prompt: str, temperature: float, top_p: float, max_output_tokens: int) -> LLMResponse:
        raise NotImplementedError


class OpenAIChatClient(LLMClient):
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from openai import OpenAI  

        self._client = OpenAI(api_key=api_key)

    def complete(self, *, model: str, prompt: str, temperature: float, top_p: float, max_output_tokens: int) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
        )
        return LLMResponse(text=resp.choices[0].message.content or "")


class VLLMOpenAICompatibleClient(LLMClient):
    """
    vLLM can expose an OpenAI-compatible API. We use OpenAI() but override base_url.
    Set:
      - VLLM_BASE_URL (e.g. http://127.0.0.1:8000/v1)
      - VLLM_API_KEY (can be 'EMPTY')
    """

    def __init__(self) -> None:
        base_url = os.getenv("VLLM_BASE_URL")
        if not base_url:
            raise RuntimeError("VLLM_BASE_URL is not set.")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")

        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, *, model: str, prompt: str, temperature: float, top_p: float, max_output_tokens: int) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_output_tokens,
        )
        return LLMResponse(text=resp.choices[0].message.content or "")


class OllamaClient(LLMClient):
    """
    Local Ollama backend (your existing setup in Topic_selection/LLM_Pick.py).
    Set OLLAMA_HOST (optional; defaults to http://127.0.0.1:11434).
    """

    def __init__(self, *, options: dict) -> None:
        import ollama

        host = os.getenv("OLLAMA_HOST")
        # python-ollama uses OLLAMA_HOST env var; set it if provided.
        if host:
            os.environ["OLLAMA_HOST"] = host
        self._ollama = ollama
        self._options = options or {}
        # Optional per-request timeout (seconds). The default client waits forever, and a local
        # runner that wedges then stalls a long sweep silently instead of failing so it can be
        # retried. Unset = old behaviour, since a large model can legitimately be slow.
        timeout = os.getenv("OLLAMA_REQUEST_TIMEOUT")
        self._client = ollama.Client(host=host, timeout=float(timeout)) if timeout else None

    def complete(self, *, model: str, prompt: str, temperature: float, top_p: float, max_output_tokens: int) -> LLMResponse:

        options = dict(self._options)
        options.setdefault("temperature", temperature)
        options.setdefault("top_p", top_p)

        options.setdefault("num_predict", max_output_tokens)

        # think=False: for "hybrid reasoning" models (e.g. the Qwen3 family), Ollama's
        # `thinking` field is separate from `response` (so parsing was never contaminated),
        # but the model still spends real generation time producing those reasoning tokens
        # before the final answer even though they land in a different field — that's where
        # the slowdown comes from. Disabling it skips that step entirely. Ollama silently
        # ignores this option for models that don't support thinking at all, so it's safe to
        # always pass regardless of which model is being called.
        generate = self._client.generate if self._client is not None else self._ollama.generate
        resp = generate(model=model, prompt=prompt, stream=False, think=False, options=options)
        return LLMResponse(text=(resp.get("response") or "").strip())


def build_client(provider: str, *, ollama_options: dict | None = None) -> LLMClient:
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAIChatClient()
    if provider in {"vllm", "vllm_openai_compatible", "vllm_openai_compatible"}:
        return VLLMOpenAICompatibleClient()
    if provider == "ollama":
        return OllamaClient(options=ollama_options or {})
    raise ValueError(f"Unsupported provider: {provider}")

