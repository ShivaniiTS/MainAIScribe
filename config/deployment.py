"""
config/deployment.py — Deployment configuration loader.

Reads config/deployment.yaml and provides typed access to deployment settings.
Server role, network addresses, feature flags, and sync configuration are all
determined by this module.

Usage:
    from config.deployment import get_deployment_config, ServerRole

    cfg = get_deployment_config()
    if cfg.is_provider_facing:
        # Enable EHR access, disable admin operations
        ...
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ServerRole(str, Enum):
    PROVIDER_FACING = "provider-facing"
    PROCESSING_PIPELINE = "processing-pipeline"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class NetworkEndpoint(BaseModel):
    host: str = "0.0.0.0"
    api_port: int = 8000
    web_port: int = 3000
    public_url: str = "http://localhost:8000"
    web_url: str = "http://localhost:3000"


class PipelineEndpoint(BaseModel):
    host: str = "0.0.0.0"
    api_port: int = 8100
    web_port: int = 3100
    internal_url: str = "http://localhost:8100"
    admin_url: str = "http://localhost:3100"


class NetworkConfig(BaseModel):
    provider_facing: NetworkEndpoint = Field(default_factory=NetworkEndpoint)
    processing_pipeline: PipelineEndpoint = Field(default_factory=PipelineEndpoint)


class DataPaths(BaseModel):
    data_dir: str = "ai-scribe-data"
    output_dir: str = "output"
    config_dir: str = "config"


class DataConfig(BaseModel):
    provider_facing: DataPaths = Field(default_factory=DataPaths)
    processing_pipeline: DataPaths = Field(default_factory=lambda: DataPaths(
        data_dir="pipeline-data",
        output_dir="pipeline-output",
        config_dir="config",
    ))


class ConfigSyncSettings(BaseModel):
    enabled: bool = True
    interval_seconds: int = 7200
    items: list[str] = Field(default_factory=lambda: [
        "providers", "templates", "dictionaries", "ehr_stub"
    ])


class OutputSyncSettings(BaseModel):
    enabled: bool = True
    overwrite_policy: str = "newer_only"


class BatchSyncSettings(BaseModel):
    max_concurrent: int = 4
    chunk_size: int = 10_485_760
    timeout_seconds: int = 300


class SyncConfig(BaseModel):
    config_sync: ConfigSyncSettings = Field(default_factory=ConfigSyncSettings)
    output_sync: OutputSyncSettings = Field(default_factory=OutputSyncSettings)
    batch: BatchSyncSettings = Field(default_factory=BatchSyncSettings)


class FeatureFlags(BaseModel):
    dashboard: bool = True
    view_encounters: bool = True
    view_providers: bool = True
    view_specialties: bool = True
    view_templates: bool = True
    view_quality: bool = True
    record_audio: bool = False
    trigger_pipeline: bool = False
    run_pipeline: bool = False
    batch_processing: bool = False
    ehr_access: bool = False
    patient_search: bool = False
    create_providers: bool = False
    edit_providers: bool = False
    create_templates: bool = False
    edit_templates: bool = False
    create_specialties: bool = False
    edit_specialties: bool = False


class FeaturesConfig(BaseModel):
    provider_facing: FeatureFlags = Field(default_factory=lambda: FeatureFlags(
        record_audio=True, trigger_pipeline=True,
        ehr_access=True, patient_search=True,
    ))
    processing_pipeline: FeatureFlags = Field(default_factory=lambda: FeatureFlags(
        run_pipeline=True, batch_processing=True,
        create_providers=True, edit_providers=True,
        create_templates=True, edit_templates=True,
        create_specialties=True, edit_specialties=True,
    ))


class SecurityConfig(BaseModel):
    phi_fields: list[str] = Field(default_factory=lambda: [
        "patient_demographics.json", "patient_context.yaml", "final_soap_note.md",
    ])
    safe_for_pipeline: list[str] = Field(default_factory=lambda: [
        "audio files", "encounter_details.json",
    ])


class InterServerAuth(BaseModel):
    enabled: bool = False
    secret_env_var: str = "AI_SCRIBE_INTER_SERVER_SECRET"

    @property
    def secret(self) -> str | None:
        if not self.enabled:
            return None
        return os.environ.get(self.secret_env_var)


class GPUConfig(BaseModel):
    asr_device: str = "cuda"
    vram_budget_gb: int = 23
    ollama_url: str = "http://localhost:11434"
    keep_alive: str = "0"
    two_pass_batch: bool = True


# ---------------------------------------------------------------------------
# Top-level deployment config
# ---------------------------------------------------------------------------
class DeploymentConfig(BaseModel):
    role: ServerRole = ServerRole.PROVIDER_FACING
    instance_id: str = "ai-scribe-dev-01"
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    inter_server_auth: InterServerAuth = Field(default_factory=InterServerAuth)
    gpu: GPUConfig = Field(default_factory=GPUConfig)

    # --- Convenience properties ---

    @property
    def is_provider_facing(self) -> bool:
        return self.role == ServerRole.PROVIDER_FACING

    @property
    def is_processing_pipeline(self) -> bool:
        return self.role == ServerRole.PROCESSING_PIPELINE

    @property
    def active_features(self) -> FeatureFlags:
        """Return feature flags for the current server role."""
        if self.role == ServerRole.PROVIDER_FACING:
            return self.features.provider_facing
        else:
            return self.features.processing_pipeline

    @property
    def api_port(self) -> int:
        """Return the API port for this server's role."""
        if self.role == ServerRole.PROCESSING_PIPELINE:
            return self.network.processing_pipeline.api_port
        return self.network.provider_facing.api_port

    @property
    def web_port(self) -> int:
        """Return the web port for this server's role."""
        if self.role == ServerRole.PROCESSING_PIPELINE:
            return self.network.processing_pipeline.web_port
        return self.network.provider_facing.web_port

    @property
    def pipeline_api_url(self) -> str:
        """URL to reach the pipeline API (used by provider-facing to proxy)."""
        return self.network.processing_pipeline.internal_url


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------
_config: DeploymentConfig | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _flatten_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert nested YAML structure into flat DeploymentConfig fields."""
    result: dict[str, Any] = {}

    server = raw.get("server", {})
    result["role"] = server.get("role", "provider-facing")
    result["instance_id"] = server.get("instance_id", "ai-scribe-dev-01")

    if "network" in raw:
        result["network"] = raw["network"]
    if "data" in raw:
        result["data"] = raw["data"]
    if "sync" in raw:
        result["sync"] = raw["sync"]
    if "features" in raw:
        result["features"] = raw["features"]

    security = raw.get("security", {})
    if "phi_fields" in security or "safe_for_pipeline" in security:
        result["security"] = {
            k: v for k, v in security.items()
            if k in ("phi_fields", "safe_for_pipeline")
        }
    auth = security.get("inter_server_auth", {})
    if auth:
        result["inter_server_auth"] = auth

    gpu = raw.get("gpu", {})
    if gpu:
        result["gpu"] = {
            "asr_device": gpu.get("asr", {}).get("device", "cuda"),
            "vram_budget_gb": gpu.get("asr", {}).get("vram_budget_gb", 23),
            "ollama_url": gpu.get("llm", {}).get("ollama_url", "http://localhost:11434"),
            "keep_alive": str(gpu.get("llm", {}).get("keep_alive", "0")),
            "two_pass_batch": gpu.get("two_pass_batch", True),
        }

    return result


def get_deployment_config(reload: bool = False) -> DeploymentConfig:
    """Load and cache the deployment configuration.

    Config file location (in priority order):
    1. AI_SCRIBE_DEPLOYMENT_CONFIG env var
    2. config/deployment.yaml relative to project root
    """
    global _config
    if _config is not None and not reload:
        return _config

    # Find config file
    config_path_env = os.environ.get("AI_SCRIBE_DEPLOYMENT_CONFIG")
    if config_path_env:
        config_path = Path(config_path_env)
    else:
        from config.paths import CONFIG_DIR
        config_path = CONFIG_DIR / "deployment.yaml"

    raw = _load_yaml(config_path)

    # Environment variable overrides
    role_override = os.environ.get("AI_SCRIBE_SERVER_ROLE")
    if role_override:
        raw.setdefault("server", {})["role"] = role_override

    flat = _flatten_yaml(raw)
    _config = DeploymentConfig(**flat)
    return _config


def require_feature(feature_name: str) -> None:
    """Raise an error if the given feature is not enabled for this server role."""
    cfg = get_deployment_config()
    flags = cfg.active_features
    if not getattr(flags, feature_name, False):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail=f"Feature '{feature_name}' is not enabled on this server (role={cfg.role.value})",
        )
