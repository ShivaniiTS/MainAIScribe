"""
tests/unit/test_paths.py — Tests for role-aware path resolution.

Ensures that DATA_DIR and OUTPUT_DIR resolve correctly based on:
1. Environment variables (highest priority)
2. Deployment config + AI_SCRIBE_SERVER_ROLE
3. Defaults (ai-scribe-data/ and output/)

These tests caught the bug where both servers shared the same directories.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env():
    """Remove path-related env vars before/after each test."""
    keys = [
        "AI_SCRIBE_DATA_DIR",
        "AI_SCRIBE_OUTPUT_DIR",
        "AI_SCRIBE_SERVER_ROLE",
        "AI_SCRIBE_ROOT",
        "AI_SCRIBE_CONFIG_DIR",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def _reimport_paths():
    """Force re-import of config.paths to re-evaluate _resolve_data_dirs()."""
    import config.paths as paths_mod
    importlib.reload(paths_mod)
    return paths_mod


class TestDefaultPaths:
    def test_defaults_to_ai_scribe_data(self):
        paths = _reimport_paths()
        assert paths.DATA_DIR.name == "ai-scribe-data"

    def test_defaults_to_output(self):
        paths = _reimport_paths()
        assert paths.OUTPUT_DIR.name == "output"

    def test_root_is_project_dir(self):
        paths = _reimport_paths()
        # ROOT should be the repo root (parent of config/)
        assert (paths.ROOT / "config").is_dir()


class TestEnvVarOverride:
    def test_env_vars_override_both(self, tmp_path):
        data_dir = str(tmp_path / "my-data")
        output_dir = str(tmp_path / "my-output")
        with patch.dict(os.environ, {
            "AI_SCRIBE_DATA_DIR": data_dir,
            "AI_SCRIBE_OUTPUT_DIR": output_dir,
        }):
            paths = _reimport_paths()
            assert str(paths.DATA_DIR) == data_dir
            assert str(paths.OUTPUT_DIR) == output_dir

    def test_env_vars_take_priority_over_role(self, tmp_path):
        """Even with a role set, env vars should win."""
        data_dir = str(tmp_path / "custom-data")
        output_dir = str(tmp_path / "custom-output")
        with patch.dict(os.environ, {
            "AI_SCRIBE_DATA_DIR": data_dir,
            "AI_SCRIBE_OUTPUT_DIR": output_dir,
            "AI_SCRIBE_SERVER_ROLE": "processing-pipeline",
        }):
            paths = _reimport_paths()
            assert str(paths.DATA_DIR) == data_dir
            assert str(paths.OUTPUT_DIR) == output_dir


class TestRoleAwarePaths:
    def test_provider_facing_uses_ai_scribe_data(self):
        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "provider-facing"}):
            paths = _reimport_paths()
            assert paths.DATA_DIR.name == "ai-scribe-data"
            assert paths.OUTPUT_DIR.name == "output"

    def test_processing_pipeline_uses_pipeline_data(self):
        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "processing-pipeline"}):
            paths = _reimport_paths()
            assert paths.DATA_DIR.name == "pipeline-data"
            assert paths.OUTPUT_DIR.name == "pipeline-output"

    def test_different_roles_get_different_dirs(self):
        """This is the critical test — both servers MUST NOT share directories."""
        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "provider-facing"}):
            provider_paths = _reimport_paths()
            provider_data = provider_paths.DATA_DIR
            provider_output = provider_paths.OUTPUT_DIR

        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "processing-pipeline"}):
            pipeline_paths = _reimport_paths()
            pipeline_data = pipeline_paths.DATA_DIR
            pipeline_output = pipeline_paths.OUTPUT_DIR

        assert provider_data != pipeline_data, \
            f"Provider and pipeline should use different DATA_DIR, both got {provider_data}"
        assert provider_output != pipeline_output, \
            f"Provider and pipeline should use different OUTPUT_DIR, both got {provider_output}"

    def test_unknown_role_falls_back_to_defaults(self):
        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "unknown-role"}):
            paths = _reimport_paths()
            assert paths.DATA_DIR.name == "ai-scribe-data"
            assert paths.OUTPUT_DIR.name == "output"

    def test_empty_role_uses_defaults(self):
        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": ""}):
            paths = _reimport_paths()
            assert paths.DATA_DIR.name == "ai-scribe-data"
            assert paths.OUTPUT_DIR.name == "output"
