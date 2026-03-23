"""
MCP Engine Registry — central discovery, instantiation, and health checking.

All engine selection is driven by config/engines.yaml.
Nodes call ``registry.get_engine("llm")`` instead of importing servers directly.

The registry:
  1. Reads config/engines.yaml to discover configured engines.
  2. Maps (engine_type, server_name) → implementation class via _SERVER_MAP.
  3. Lazily instantiates servers on first request and caches them.
  4. Exposes health_check_all() for startup/monitoring.
  5. Supports failover: if the default server fails, tries the next configured one.

Adding a new engine implementation:
  1. Create mcp_servers/{type}/{name}_server.py implementing the base interface.
  2. Add an entry to _SERVER_MAP below.
  3. Add config to config/engines.yaml under the appropriate section.
  That's it — zero changes to pipeline nodes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from config.loader import get_asr_config, get_llm_config, load_engines_config

from mcp_servers.asr.base import ASREngine
from mcp_servers.llm.base import LLMEngine
from mcp_servers.ehr.base import EHRAdapter

log = logging.getLogger(__name__)

# Type alias for any engine base class
EngineBase = Union[ASREngine, LLMEngine, EHRAdapter]

# ─────────────────────────────────────────────────────────────────────────────
# Server map: (engine_type, config_type_field) → (module_path, class_name)
#
# When adding a new implementation, add one line here.
# ─────────────────────────────────────────────────────────────────────────────
_SERVER_MAP: dict[tuple[str, str], tuple[str, str]] = {
    # ASR engines
    ("asr", "whisperx"):        ("mcp_servers.asr.whisperx_server",      "WhisperXServer"),
    ("asr", "whisperx_lora"):   ("mcp_servers.asr.whisperx_lora_server", "WhisperXLoRAServer"),
    # ("asr", "nemo"):          ("mcp_servers.asr.nemo_streaming_server", "NemoStreamingServer"),
    # ("asr", "deepgram"):      ("mcp_servers.asr.deepgram_server",      "DeepgramServer"),

    # LLM engines
    ("llm", "ollama"):          ("mcp_servers.llm.ollama_server",        "OllamaServer"),
    # ("llm", "vllm"):          ("mcp_servers.llm.vllm_server",          "VLLMServer"),
    # ("llm", "openai_compatible"): ("mcp_servers.llm.openai_server",    "OpenAIServer"),

    # EHR adapters
    ("ehr", "stub"):            ("mcp_servers.ehr.stub_server",          "StubEHRServer"),
    ("ehr", "manual"):          ("mcp_servers.ehr.stub_server",          "StubEHRServer"),
    # ("ehr", "fhir"):          ("mcp_servers.ehr.fhir_server",          "FHIRServer"),
    # ("ehr", "browser_extension"): ("mcp_servers.ehr.extension_server", "ExtensionServer"),
}


@dataclass
class EngineStatus:
    """Health status of a single engine."""
    engine_type: str
    server_name: str
    healthy: bool
    error: Optional[str] = None


@dataclass
class RegistryStatus:
    """Aggregate health across all registered engines."""
    engines: list[EngineStatus] = field(default_factory=list)

    @property
    def all_healthy(self) -> bool:
        return all(e.healthy for e in self.engines)

    def summary(self) -> str:
        lines = ["=== ENGINE REGISTRY STATUS ==="]
        for e in self.engines:
            icon = "OK" if e.healthy else "FAIL"
            line = f"  [{icon}] {e.engine_type}/{e.server_name}"
            if e.error:
                line += f" — {e.error}"
            lines.append(line)
        return "\n".join(lines)


class EngineRegistry:
    """
    Central registry for all MCP engine servers.

    Usage:
        registry = EngineRegistry()          # reads config/engines.yaml
        llm = registry.get("llm")            # returns default LLM engine
        asr = registry.get("asr", "whisperx")# returns specific ASR engine
    """

    def __init__(self) -> None:
        self._config = load_engines_config()
        self._cache: dict[tuple[str, str], EngineBase] = {}

    def reload_config(self) -> None:
        """Re-read engines.yaml and clear cached instances."""
        from config.loader import invalidate_config_cache
        invalidate_config_cache()
        self._config = load_engines_config()
        self._cache.clear()
        log.info("registry: config reloaded, engine cache cleared")

    # ── Core API ─────────────────────────────────────────────────────────

    def get(
        self,
        engine_type: str,
        server_name: Optional[str] = None,
    ) -> EngineBase:
        """
        Get an engine instance by type and optional server name.

        Args:
            engine_type: "llm", "asr", or "ehr"
            server_name: Specific server (e.g. "ollama", "whisperx").
                         Defaults to the configured default for that type.

        Returns:
            A cached engine instance implementing the appropriate base interface.

        Raises:
            KeyError: if engine_type or server_name is not configured.
            ImportError: if the server implementation module is not found.
        """
        name = server_name or self._default_server(engine_type)
        cache_key = (engine_type, name)

        if cache_key not in self._cache:
            self._cache[cache_key] = self._instantiate(engine_type, name)
            log.info(f"registry: instantiated {engine_type}/{name}")

        return self._cache[cache_key]

    def get_llm(self, server_name: Optional[str] = None) -> LLMEngine:
        """Typed convenience: get an LLM engine."""
        return self.get("llm", server_name)  # type: ignore[return-value]

    def get_asr(self, server_name: Optional[str] = None) -> ASREngine:
        """Typed convenience: get an ASR engine."""
        return self.get("asr", server_name)  # type: ignore[return-value]

    def get_ehr(self, server_name: Optional[str] = None) -> EHRAdapter:
        """Typed convenience: get an EHR adapter."""
        return self.get("ehr", server_name)  # type: ignore[return-value]

    def get_asr_for_provider(
        self,
        provider_id: str,
        use_lora: bool = False,
    ) -> ASREngine:
        """
        Return the best available ASR engine for a given provider.

        LoRA fine-tuning is opt-in (use_lora=True) — it is NOT loaded
        automatically even when an adapter exists.  This is intentional:

        - Ambient LoRA shows ~12% WER improvement but was trained on SOAP
          note text (summaries), not verbatim transcripts.  Dictation LoRA
          degrades WER by ~24% for the same reason.
        - Per-provider LoRA adapters require ≥30 min of *verbatim* transcript
          data to be reliable.  Until that data is accumulated via the
          correction-capture loop (see docs/architecture.md §9 and §10.3),
          the base WhisperX model produces better results for dictation.
        - Once a provider has sufficient verified data, set use_lora=True in
          their profile (provider.asr_lora_enabled: true) and call this method
          with use_lora=True from transcribe_node.py.

        Args:
            provider_id: Provider identifier.
            use_lora:    If True, load the LoRA adapter when one exists.
                         Defaults to False (base model always used unless
                         explicitly requested).
        """
        if use_lora:
            from mcp_servers.asr.whisperx_lora_server import (
                WhisperXLoRAServer,
                adapter_exists,
            )

            cache_key = ("asr", f"whisperx_lora/{provider_id}")
            if cache_key in self._cache:
                return self._cache[cache_key]  # type: ignore[return-value]

            if adapter_exists(provider_id):
                log.info(
                    "registry: LoRA requested for provider '%s' — loading WhisperXLoRAServer",
                    provider_id,
                )
                asr_cfg = get_asr_config()
                base_cfg = asr_cfg.get("servers", {}).get("whisperx", {})
                engine = WhisperXLoRAServer.for_provider(
                    provider_id=provider_id,
                    device=base_cfg.get("device", "cuda"),
                    compute_type=base_cfg.get("compute_type", "float16"),
                    diarization=base_cfg.get("diarization", True),
                    hf_token=base_cfg.get("hf_token"),
                )
                self._cache[cache_key] = engine
                return engine

            log.warning(
                "registry: use_lora=True requested for provider '%s' but no adapter found — "
                "falling back to base WhisperX",
                provider_id,
            )

        # Default path: always use base WhisperX
        return self.get_asr()

    # ── Health checks ────────────────────────────────────────────────────

    async def health_check(
        self,
        engine_type: str,
        server_name: Optional[str] = None,
    ) -> EngineStatus:
        """Run health check on a single engine."""
        name = server_name or self._default_server(engine_type)
        try:
            engine = self.get(engine_type, name)
            healthy = await engine.health_check()
            return EngineStatus(
                engine_type=engine_type,
                server_name=name,
                healthy=healthy,
                error=None if healthy else "health_check returned False",
            )
        except Exception as exc:
            return EngineStatus(
                engine_type=engine_type,
                server_name=name,
                healthy=False,
                error=str(exc),
            )

    async def health_check_all(self) -> RegistryStatus:
        """Run health checks on all instantiated engines."""
        status = RegistryStatus()
        for (engine_type, name) in list(self._cache.keys()):
            result = await self.health_check(engine_type, name)
            status.engines.append(result)
        return status

    async def health_check_defaults(self) -> RegistryStatus:
        """Run health checks on default engines (llm, asr, ehr)."""
        status = RegistryStatus()
        for engine_type in ("llm", "asr", "ehr"):
            try:
                name = self._default_server(engine_type)
                result = await self.health_check(engine_type, name)
                status.engines.append(result)
            except KeyError:
                status.engines.append(EngineStatus(
                    engine_type=engine_type,
                    server_name="(not configured)",
                    healthy=False,
                    error="no configuration found in engines.yaml",
                ))
        return status

    # ── Failover ─────────────────────────────────────────────────────────

    def get_with_failover(
        self,
        engine_type: str,
        server_name: Optional[str] = None,
    ) -> EngineBase:
        """
        Try the requested (or default) server; if instantiation fails,
        fall through to other configured servers of the same type.
        """
        name = server_name or self._default_server(engine_type)
        servers = self._configured_servers(engine_type)

        # Try requested server first
        try:
            return self.get(engine_type, name)
        except Exception as exc:
            log.warning(f"registry: {engine_type}/{name} failed: {exc}")

        # Try remaining servers in config order
        for fallback in servers:
            if fallback == name:
                continue
            try:
                engine = self.get(engine_type, fallback)
                log.info(f"registry: failover {engine_type}/{name} → {fallback}")
                return engine
            except Exception as exc:
                log.warning(f"registry: failover {engine_type}/{fallback} also failed: {exc}")

        raise RuntimeError(
            f"No available {engine_type} engine. Tried: {[name] + [s for s in servers if s != name]}"
        )

    # ── Introspection ────────────────────────────────────────────────────

    def list_configured(self, engine_type: str) -> list[str]:
        """List all server names configured for an engine type."""
        return self._configured_servers(engine_type)

    def list_available(self, engine_type: str) -> list[str]:
        """List server names that have implementations in _SERVER_MAP."""
        return [
            name for name in self._configured_servers(engine_type)
            if (engine_type, self._server_type(engine_type, name)) in _SERVER_MAP
        ]

    def unload_engine(self, engine_type: str, server_name: str | None = None) -> None:
        """Unload a cached engine instance and free its resources (e.g., GPU memory).

        If the engine has an ``unload_model`` method (e.g., WhisperXServer), it is
        called before removing the instance from the cache.
        """
        name = server_name or self._default_server(engine_type)
        cache_key = (engine_type, name)
        engine = self._cache.pop(cache_key, None)
        if engine is None:
            return
        if hasattr(engine, "unload_model"):
            engine.unload_model()
        log.info("registry: unloaded %s/%s", engine_type, name)

    def list_cached(self) -> list[tuple[str, str]]:
        """List currently instantiated (engine_type, server_name) pairs."""
        return list(self._cache.keys())

    # ── Internals ────────────────────────────────────────────────────────

    def _default_server(self, engine_type: str) -> str:
        """Get the default server name for an engine type from config."""
        section = self._config.get(engine_type, {})
        # LLM/ASR use "default_server", EHR uses "default_adapter"
        return (
            section.get("default_server")
            or section.get("default_adapter")
            or next(iter(self._configured_servers(engine_type)), "")
        )

    def _configured_servers(self, engine_type: str) -> list[str]:
        """List all server names in engines.yaml for a given type."""
        section = self._config.get(engine_type, {})
        servers = section.get("servers") or section.get("adapters") or {}
        return list(servers.keys())

    def _server_config(self, engine_type: str, server_name: str) -> dict[str, Any]:
        """Get the raw config dict for a specific server."""
        section = self._config.get(engine_type, {})
        servers = section.get("servers") or section.get("adapters") or {}
        if server_name not in servers:
            raise KeyError(
                f"Server '{server_name}' not found in engines.yaml [{engine_type}]. "
                f"Available: {list(servers.keys())}"
            )
        return {"name": server_name, **servers[server_name]}

    def _server_type(self, engine_type: str, server_name: str) -> str:
        """Get the 'type' field from a server's config."""
        cfg = self._server_config(engine_type, server_name)
        return cfg.get("type", server_name)

    def _instantiate(self, engine_type: str, server_name: str) -> EngineBase:
        """Import the server class and instantiate it from config."""
        cfg = self._server_config(engine_type, server_name)
        server_type = cfg.get("type", server_name)

        map_key = (engine_type, server_type)
        if map_key not in _SERVER_MAP:
            raise KeyError(
                f"No implementation registered for {engine_type}/{server_type}. "
                f"Add an entry to _SERVER_MAP in mcp_servers/registry.py"
            )

        module_path, class_name = _SERVER_MAP[map_key]
        mod = _import_module(module_path)
        cls = getattr(mod, class_name)
        return cls.from_config(cfg)


def _import_module(path: str):
    """Dynamic import helper."""
    import importlib
    return importlib.import_module(path)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
_registry: Optional[EngineRegistry] = None


def get_registry() -> EngineRegistry:
    """Get the module-level EngineRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the singleton (for tests)."""
    global _registry
    _registry = None
