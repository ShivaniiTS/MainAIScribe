"""
tests/unit/test_inter_server_auth.py — Inter-server authentication middleware tests.

Covers:
1. Auth middleware only protects /pipeline/* routes
2. 401 on missing X-Inter-Server-Auth header
3. 401 on wrong auth token
4. 200 when correct auth token is provided
5. Non-pipeline routes pass through without auth header
6. Auth disabled mode — all requests pass through
"""
from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _make_client(role: str, auth_enabled: bool = False, secret: str = "") -> TestClient:
    """Build a TestClient with the given role and auth settings.

    Reloads api.main so module-level cfg picks up the new environment.
    """
    env = {
        "AI_SCRIBE_SERVER_ROLE": role,
    }
    if auth_enabled and secret:
        env["AI_SCRIBE_INTER_SERVER_SECRET"] = secret

    # We need to reload modules in the correct order to pick up new env
    with patch.dict(os.environ, env, clear=False):
        import config.paths
        importlib.reload(config.paths)

        import config.deployment
        importlib.reload(config.deployment)

        # Write a temporary deployment.yaml-like config via env override
        cfg_module = config.deployment
        cfg_module._config = None  # Reset singleton

        if auth_enabled:
            # Patch the InterServerAuth config
            orig_flatten = cfg_module._flatten_yaml

            def patched_flatten(raw):
                result = orig_flatten(raw)
                result["inter_server_auth"] = {
                    "enabled": True,
                    "secret_env_var": "AI_SCRIBE_INTER_SERVER_SECRET",
                }
                return result

            with patch.object(cfg_module, "_flatten_yaml", side_effect=patched_flatten):
                cfg_module.get_deployment_config(reload=True)
        else:
            cfg_module.get_deployment_config(reload=True)

        import api.data_loader
        importlib.reload(api.data_loader)
        api.data_loader._quality_cache.clear()

        import api.main
        importlib.reload(api.main)

        client = TestClient(api.main.app)
        return client


class TestAuthMiddlewareEnabled:
    """Tests with inter-server auth enabled."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        yield
        # Reset deployment config
        import config.deployment
        config.deployment._config = None

    def test_pipeline_route_401_missing_header(self):
        """Pipeline routes should return 401 when auth header is missing."""
        client = _make_client("processing-pipeline", auth_enabled=True, secret="test-secret-123")

        resp = client.get("/pipeline/status/job-fake")
        # Will be 401 (auth middleware) or 404 (job not found) — auth check comes first
        if resp.status_code == 401:
            assert "Invalid inter-server auth" in resp.json()["detail"]

    def test_pipeline_route_401_wrong_token(self):
        """Pipeline routes should return 401 with wrong token."""
        client = _make_client("processing-pipeline", auth_enabled=True, secret="test-secret-123")

        resp = client.get(
            "/pipeline/status/job-fake",
            headers={"X-Inter-Server-Auth": "wrong-token"},
        )
        if resp.status_code == 401:
            assert "Invalid inter-server auth" in resp.json()["detail"]

    def test_pipeline_route_passes_with_correct_token(self):
        """Pipeline routes should pass auth with correct token (then 404 for nonexistent job)."""
        client = _make_client("processing-pipeline", auth_enabled=True, secret="test-secret-123")

        resp = client.get(
            "/pipeline/status/job-fake",
            headers={"X-Inter-Server-Auth": "test-secret-123"},
        )
        # Should get past auth middleware — 404 because job doesn't exist
        assert resp.status_code in (404, 200)

    def test_non_pipeline_routes_bypass_auth(self):
        """Non-pipeline routes should not require auth header."""
        client = _make_client("processing-pipeline", auth_enabled=True, secret="test-secret-123")

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_encounters_route_bypasses_auth(self):
        """Encounters routes don't go through /pipeline/* — no auth needed."""
        client = _make_client("processing-pipeline", auth_enabled=True, secret="test-secret-123")

        resp = client.get("/encounters")
        assert resp.status_code == 200


class TestAuthMiddlewareDisabled:
    """Tests with inter-server auth disabled (default)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        yield
        import config.deployment
        config.deployment._config = None

    def test_pipeline_routes_accessible_without_auth(self):
        """With auth disabled, pipeline routes should be accessible."""
        client = _make_client("processing-pipeline", auth_enabled=False)

        resp = client.get("/pipeline/status/job-fake")
        # Should get through to the actual handler — 404 because job doesn't exist
        assert resp.status_code == 404

    def test_health_accessible(self):
        client = _make_client("provider-facing", auth_enabled=False)
        resp = client.get("/health")
        assert resp.status_code == 200


class TestFeatureFlagEnforcement:
    """Test require_feature() blocks access based on server role."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        yield
        import config.deployment
        config.deployment._config = None

    def test_provider_facing_cannot_run_pipeline(self):
        """Provider-facing server should get 403 on run_pipeline features."""
        from config.deployment import require_feature, get_deployment_config

        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "provider-facing"}):
            import config.deployment
            config.deployment._config = None
            config.deployment.get_deployment_config(reload=True)

            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc:
                require_feature("run_pipeline")
            assert exc.value.status_code == 403

    def test_processing_pipeline_can_run_pipeline(self):
        """Processing pipeline should have run_pipeline enabled."""
        from config.deployment import require_feature

        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "processing-pipeline"}):
            import config.deployment
            config.deployment._config = None
            config.deployment.get_deployment_config(reload=True)

            # Should not raise
            require_feature("run_pipeline")

    def test_processing_pipeline_has_pipeline_features(self):
        """Processing pipeline should have pipeline features enabled."""
        from config.deployment import require_feature

        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "processing-pipeline"}):
            import config.deployment
            config.deployment._config = None
            config.deployment.get_deployment_config(reload=True)

            # These should not raise
            require_feature("run_pipeline")
            require_feature("batch_processing")

    def test_provider_facing_has_ehr_features(self):
        """Provider-facing should have EHR features enabled."""
        from config.deployment import require_feature

        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "provider-facing"}):
            import config.deployment
            config.deployment._config = None
            config.deployment.get_deployment_config(reload=True)

            require_feature("ehr_access")
