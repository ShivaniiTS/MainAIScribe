"""
LLM MCP Server Base Interface.

All LLM backends (Ollama, vLLM, Claude, OpenAI, …) implement LLMEngine.
The Note node calls only this interface — never the model server directly.

Implementations:
    ollama_server.py   — DEFAULT (Ollama OpenAI-compatible endpoint)
    vllm_server.py     — high-throughput production option
    claude_server.py   — cloud option (Anthropic API)
    openai_server.py   — cloud option (OpenAI API)

All servers POST to /v1/chat/completions — the only difference is the URL
and optional API key. Switching providers is a config change, not a code change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class LLMMessage:
    role: str    # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    """Per-request generation parameters."""
    model: str = "qwen2.5:32b"
    temperature: float = 0.1       # Low temperature for clinical accuracy
    max_tokens: int = 4096
    top_p: float = 0.9
    stop: list[str] = field(default_factory=list)
    # Provider-specific
    seed: Optional[int] = None
    response_format: Optional[dict] = None   # {"type": "json_object"} for structured output


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"
    # Structured output (when response_format=json_object)
    parsed_json: Optional[dict] = None


@dataclass
class LLMChunk:
    """Streaming response chunk."""
    delta: str
    is_final: bool
    finish_reason: Optional[str] = None


@dataclass
class ModelInfo:
    model_name: str
    context_window: int
    supports_streaming: bool = True
    supports_json_mode: bool = False
    capabilities: list[str] = field(default_factory=list)


class LLMEngine(ABC):
    """
    Abstract base class for all LLM MCP servers.

    Every LLM backend must implement these three methods.
    The note node calls only this interface.
    """

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResponse:
        """
        Generate a completion (non-streaming).

        Args:
            system_prompt: System/instruction message.
            messages:      Conversation history.
            config:        Generation parameters (model, temperature, …).

        Returns:
            LLMResponse with content and token counts.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[LLMChunk]:
        """
        Generate a streaming completion.

        Args:
            system_prompt: System/instruction message.
            messages:      Conversation history.
            config:        Generation parameters.

        Yields:
            LLMChunk deltas; last chunk has is_final=True.
        """
        ...

    @abstractmethod
    async def get_model_info(self) -> ModelInfo:
        """
        Return metadata about the current model.

        Used by the note node to budget context window and by the LLM router.
        """
        ...

    async def health_check(self) -> bool:
        """Verify the LLM server is reachable. Override for custom checks."""
        try:
            await self.get_model_info()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Server", "").lower()
