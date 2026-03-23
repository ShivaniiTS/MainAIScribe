"""
tests/unit/test_deployment_config.py — Tests for deployment configuration.

Tests the config loader, server roles, feature flags, and role-based
access control.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config.deployment import (
    DeploymentConfig,
    FeatureFlags,
    ServerRole,
    get_deployment_config,
    require_feature,
    _flatten_yaml,
)


# ---------------------------------------------------------------------------
# DeploymentConfig model tests
# ---------------------------------------------------------------------------
class TestDeploymentConfig:
    def test_default_role_is_provider_facing(self):
        cfg = DeploymentConfig()
        assert cfg.role == ServerRole.PROVIDER_FACING

    def test_is_provider_facing_only(self):
        cfg = DeploymentConfig(role=ServerRole.PROVIDER_FACING)
        assert cfg.is_provider_facing is True
        assert cfg.is_processing_pipeline is False

    def test_is_processing_pipeline_only(self):
        cfg = DeploymentConfig(role=ServerRole.PROCESSING_PIPELINE)
        assert cfg.is_provider_facing is False
        assert cfg.is_processing_pipeline is True

    def test_api_port_provider_facing(self):
        cfg = DeploymentConfig(role=ServerRole.PROVIDER_FACING)
        assert cfg.api_port == 8000

    def test_api_port_processing_pipeline(self):
        cfg = DeploymentConfig(role=ServerRole.PROCESSING_PIPELINE)
        assert cfg.api_port == 8100

    def test_pipeline_api_url(self):
        cfg = DeploymentConfig()
        assert cfg.pipeline_api_url == "http://localhost:8100"


# ---------------------------------------------------------------------------
# Feature flags tests
# ---------------------------------------------------------------------------
class TestFeatureFlags:
    def test_provider_facing_features(self):
        cfg = DeploymentConfig(role=ServerRole.PROVIDER_FACING)
        flags = cfg.active_features
        assert flags.ehr_access is True
        assert flags.patient_search is True
        assert flags.record_audio is True
        assert flags.trigger_pipeline is True
        # Admin features disabled
        assert flags.create_providers is False
        assert flags.edit_templates is False
        assert flags.create_specialties is False

    def test_processing_pipeline_features(self):
        cfg = DeploymentConfig(role=ServerRole.PROCESSING_PIPELINE)
        flags = cfg.active_features
        assert flags.run_pipeline is True
        assert flags.batch_processing is True
        assert flags.create_providers is True
        assert flags.edit_templates is True
        # EHR disabled
        assert flags.ehr_access is False
        assert flags.patient_search is False

    def test_provider_facing_does_not_have_pipeline_features(self):
        cfg = DeploymentConfig(role=ServerRole.PROVIDER_FACING)
        flags = cfg.active_features
        assert flags.ehr_access is True
        assert flags.run_pipeline is False
        assert flags.create_providers is False

    def test_processing_pipeline_does_not_have_ehr_features(self):
        cfg = DeploymentConfig(role=ServerRole.PROCESSING_PIPELINE)
        flags = cfg.active_features
        assert flags.ehr_access is False
        assert flags.run_pipeline is True
        assert flags.create_providers is True


# ---------------------------------------------------------------------------
# YAML loading tests
# ---------------------------------------------------------------------------
class TestFlattenYaml:
    def test_empty_yaml(self):
        result = _flatten_yaml({})
        cfg = DeploymentConfig(**result)
        assert cfg.role == ServerRole.PROVIDER_FACING

    def test_role_from_yaml(self):
        raw = {"server": {"role": "provider-facing"}}
        result = _flatten_yaml(raw)
        cfg = DeploymentConfig(**result)
        assert cfg.role == ServerRole.PROVIDER_FACING

    def test_instance_id(self):
        raw = {"server": {"instance_id": "test-server-01"}}
        result = _flatten_yaml(raw)
        cfg = DeploymentConfig(**result)
        assert cfg.instance_id == "test-server-01"

    def test_gpu_config(self):
        raw = {
            "gpu": {
                "asr": {"device": "cpu", "vram_budget_gb": 0},
                "llm": {"ollama_url": "http://gpu-host:11434"},
                "two_pass_batch": False,
            }
        }
        result = _flatten_yaml(raw)
        cfg = DeploymentConfig(**result)
        assert cfg.gpu.asr_device == "cpu"
        assert cfg.gpu.vram_budget_gb == 0
        assert cfg.gpu.ollama_url == "http://gpu-host:11434"
        assert cfg.gpu.two_pass_batch is False


# ---------------------------------------------------------------------------
# Config loader tests
# ---------------------------------------------------------------------------
class TestGetDeploymentConfig:
    def test_loads_from_default_path(self):
        """Test that the config loads from config/deployment.yaml."""
        cfg = get_deployment_config(reload=True)
        assert cfg.role == ServerRole.PROVIDER_FACING  # default in deployment.yaml

    def test_env_var_role_override(self):
        """Test that AI_SCRIBE_SERVER_ROLE env var overrides YAML."""
        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "provider-facing"}):
            cfg = get_deployment_config(reload=True)
            assert cfg.role == ServerRole.PROVIDER_FACING

        # Reset to default
        get_deployment_config(reload=True)

    def test_caches_config(self):
        cfg1 = get_deployment_config(reload=True)
        cfg2 = get_deployment_config()
        assert cfg1 is cfg2

    def test_reload_refreshes(self):
        cfg1 = get_deployment_config(reload=True)
        cfg2 = get_deployment_config(reload=True)
        # New instance created
        assert cfg1 is not cfg2


# ---------------------------------------------------------------------------
# require_feature tests
# ---------------------------------------------------------------------------
class TestRequireFeature:
    def test_allowed_feature_passes(self):
        """In provider-facing mode, view features should be enabled."""
        get_deployment_config(reload=True)  # Reset to provider-facing
        # Should not raise
        require_feature("dashboard")
        require_feature("view_encounters")

    def test_disabled_feature_raises(self):
        """When running as provider-facing, admin features should be blocked."""
        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "provider-facing"}):
            get_deployment_config(reload=True)
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                require_feature("create_providers")
            assert exc_info.value.status_code == 403

        # Reset
        get_deployment_config(reload=True)


# ---------------------------------------------------------------------------
# Sync config tests
# ---------------------------------------------------------------------------
class TestSyncConfig:
    def test_default_sync_interval(self):
        cfg = DeploymentConfig()
        assert cfg.sync.config_sync.interval_seconds == 7200

    def test_default_overwrite_policy(self):
        cfg = DeploymentConfig()
        assert cfg.sync.output_sync.overwrite_policy == "newer_only"

    def test_sync_items(self):
        cfg = DeploymentConfig()
        items = cfg.sync.config_sync.items
        assert "providers" in items
        assert "templates" in items
        assert "dictionaries" in items


# ---------------------------------------------------------------------------
# Security config tests
# ---------------------------------------------------------------------------
class TestSecurityConfig:
    def test_phi_fields(self):
        cfg = DeploymentConfig()
        assert "patient_demographics.json" in cfg.security.phi_fields
        assert "patient_context.yaml" in cfg.security.phi_fields

    def test_inter_server_auth_disabled_by_default(self):
        cfg = DeploymentConfig()
        assert cfg.inter_server_auth.enabled is False
        assert cfg.inter_server_auth.secret is None
