"""
Ollama LLM MCP Server.

Wraps Ollama's OpenAI-compatible /v1/chat/completions endpoint.
Swapping to vLLM or Claude = change the URL + API key in engines.yaml.
No code changes required.

Default endpoint: http://localhost:11434/v1
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

from mcp_servers.llm.base import (
    LLMChunk,
    LLMConfig,
    LLMEngine,
    LLMMessage,
    LLMResponse,
    ModelInfo,
)

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "qwen2.5:32b"
_CONNECT_TIMEOUT = 5.0    # seconds — fail fast if Ollama isn't running
_READ_TIMEOUT = 300.0     # seconds — large models can be slow


def _build_request_body(
    system_prompt: str,
    messages: list[LLMMessage],
    config: LLMConfig,
    stream: bool = False,
    keep_alive: str | int | None = None,
) -> dict[str, Any]:
    all_messages = [{"role": "system", "content": system_prompt}]
    all_messages += [{"role": m.role, "content": m.content} for m in messages]

    body: dict[str, Any] = {
        "model": config.model,
        "messages": all_messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "stream": stream,
    }
    if config.stop:
        body["stop"] = config.stop
    if config.seed is not None:
        body["seed"] = config.seed
    if config.response_format:
        body["response_format"] = config.response_format
    # keep_alive=0 tells Ollama to unload the model from VRAM immediately after
    # the response. Useful when sharing a GPU with WhisperX (large-v3 ~10 GB).
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    return body


class OllamaServer(LLMEngine):
    """
    LLM engine backed by a local Ollama instance.

    Ollama serves all models through an OpenAI-compatible API:
        POST {url}/chat/completions

    Supports both synchronous (generate_sync) and async (generate) modes.
    The note_node uses generate_sync; async methods are available for
    future streaming / WebSocket use cases.
    """

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        api_key: Optional[str] = None,
        model_overrides: Optional[dict[str, str]] = None,
        connect_timeout: float = _CONNECT_TIMEOUT,
        read_timeout: float = _READ_TIMEOUT,
        keep_alive: str | int | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.model_overrides = model_overrides or {}
        # keep_alive=0 unloads the model from VRAM after each response.
        # Useful when sharing GPU with WhisperX. None = Ollama default (5 min).
        self.keep_alive = keep_alive
        self._timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=30.0, pool=5.0)
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "OllamaServer":
        """Instantiate from a config block returned by config.loader.get_llm_config()."""
        return cls(
            url=cfg.get("url", _DEFAULT_URL),
            api_key=cfg.get("api_key"),
            model_overrides=cfg.get("models", {}),
        )

    def model_for_task(self, task: str, config: LLMConfig) -> str:
        """Resolve model name: explicit config > task override > config.model."""
        return self.model_overrides.get(task, config.model)

    # ── Sync path (used by note_node in LangGraph sync execution) ──────────

    def generate_sync(
        self,
        system_prompt: str,
        messages: list[LLMMessage],
        config: LLMConfig,
        task: str = "note_generation",
    ) -> LLMResponse:
        """
        Blocking HTTP call to Ollama. Safe to call from sync LangGraph nodes.

        Args:
            system_prompt: System instruction.
            messages:      Conversation messages.
            config:        Generation parameters.
            task:          Task name for model selection (e.g. "note_generation").

        Returns:
            LLMResponse on success.

        Raises:
            httpx.ConnectError:  Ollama is not running.
            httpx.TimeoutException: Model is too slow / request timed out.
            RuntimeError:        HTTP error from Ollama.
        """
        effective_model = self.model_overrides.get(task, config.model)
        effective_config = LLMConfig(**{**config.__dict__, "model": effective_model})

        body = _build_request_body(system_prompt, messages, effective_config, stream=False, keep_alive=self.keep_alive)
        endpoint = f"{self.url}/chat/completions"

        logger.debug("ollama: POST %s model=%s tokens_max=%d", endpoint, effective_model, config.max_tokens)

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(endpoint, json=body, headers=self._headers)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", effective_model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    # ── Async path (for future streaming / WebSocket endpoints) ───────────

    async def generate(
        self,
        system_prompt: str,
        messages: list[LLMMessage],
        config: LLMConfig,
        task: str = "note_generation",
    ) -> LLMResponse:
        effective_model = self.model_overrides.get(task, config.model)
        effective_config = LLMConfig(**{**config.__dict__, "model": effective_model})

        body = _build_request_body(system_prompt, messages, effective_config, stream=False, keep_alive=self.keep_alive)
        endpoint = f"{self.url}/chat/completions"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(endpoint, json=body, headers=self._headers)

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", effective_model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[LLMMessage],
        config: LLMConfig,
        task: str = "note_generation",
    ) -> AsyncIterator[LLMChunk]:
        effective_model = self.model_overrides.get(task, config.model)
        effective_config = LLMConfig(**{**config.__dict__, "model": effective_model})

        body = _build_request_body(system_prompt, messages, effective_config, stream=True, keep_alive=self.keep_alive)
        endpoint = f"{self.url}/chat/completions"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", endpoint, json=body, headers=self._headers) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama returned HTTP {resp.status_code}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        yield LLMChunk(delta="", is_final=True, finish_reason="stop")
                        return
                    try:
                        chunk = json.loads(raw)
                        delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        finish = chunk["choices"][0].get("finish_reason")
                        yield LLMChunk(delta=delta, is_final=bool(finish), finish_reason=finish)
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def get_model_info(self) -> ModelInfo:
        """Query Ollama's /api/tags to get available models and context size."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.url.replace('/v1', '')}/api/tags")
            models = resp.json().get("models", [])
            # Find context window for default model
            context = 32768  # Qwen 2.5 default
            return ModelInfo(
                model_name=self.model_overrides.get("note_generation", _DEFAULT_MODEL),
                context_window=context,
                supports_streaming=True,
                supports_json_mode=True,
            )
        except Exception:
            return ModelInfo(
                model_name=_DEFAULT_MODEL,
                context_window=32768,
                supports_streaming=True,
            )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                resp = await client.get(f"{self.url.replace('/v1', '')}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
