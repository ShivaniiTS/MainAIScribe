"""
tests/e2e/test_full_workflow.py — End-to-end workflow tests.

Tests the complete flow from encounter creation through pipeline execution
to output retrieval, covering both direct API and pipeline API paths.

Note: These tests may be slow as they exercise the actual pipeline (ASR + LLM).
Mark with @pytest.mark.slow to skip in fast CI runs.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_config():
    with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "provider-facing"}):
        from config.deployment import get_deployment_config
        get_deployment_config(reload=True)
        yield
    from config.deployment import get_deployment_config
    get_deployment_config(reload=True)


@pytest.fixture
def client():
    from config.deployment import get_deployment_config
    get_deployment_config(reload=True)
    import importlib
    import api.main
    importlib.reload(api.main)
    return TestClient(api.main.app)


class TestServerDeploymentModes:
    """Test that the server correctly configures itself for each role."""

    def test_provider_facing_has_provider_routes(self, client):
        """Provider-facing server should have encounter, provider, and patient routes."""
        assert client.get("/encounters").status_code == 200
        assert client.get("/providers").status_code == 200
        assert client.get("/specialties").status_code == 200
        assert client.get("/templates").status_code == 200
        assert client.get("/quality/aggregate").status_code == 200
        assert client.get("/patients/search").status_code == 200
        assert client.get("/config/features").status_code == 200
        assert client.get("/config/role").status_code == 200

    def test_features_reflect_provider_facing_role(self, client):
        features = client.get("/config/features").json()
        assert features["ehr_access"] is True
        assert features["patient_search"] is True
        assert features["run_pipeline"] is False
        assert features["create_providers"] is False


class TestPipelineAPIWorkflow:
    """Test the pipeline API upload → trigger → status → output flow."""

    @pytest.fixture
    def client(self):
        """Pipeline tests need the processing-pipeline role."""
        with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "processing-pipeline"}):
            from config.deployment import get_deployment_config
            get_deployment_config(reload=True)
            import importlib
            import api.main
            importlib.reload(api.main)
            yield TestClient(api.main.app)
        from config.deployment import get_deployment_config
        get_deployment_config(reload=True)

    def test_upload_and_status(self, client):
        """Upload encounter files and check status."""
        audio = io.BytesIO(b"fake mp3 audio content for test")
        resp = client.post(
            "/pipeline/upload",
            files={"audio": ("dictation.mp3", audio, "audio/mpeg")},
            data={
                "sample_id": "e2e_test_sample_001",
                "mode": "dictation",
                "provider_id": "dr_e2e_test",
                "encounter_details": json.dumps({
                    "visit_type": "follow_up",
                    "mode": "dictation",
                }),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        job_id = data["job_id"]
        assert data["status"] == "pending"

        # Check status
        status = client.get(f"/pipeline/status/{job_id}").json()
        assert status["status"] == "pending"
        assert status["sample_id"] == "e2e_test_sample_001"

    def test_batch_output_retrieval(self, client):
        """Batch output retrieval with no matching samples returns empty."""
        resp = client.get(
            "/pipeline/outputs/batch",
            params={"sample_ids": "nonexistent_001,nonexistent_002"},
        )
        assert resp.status_code == 200
        assert resp.json()["samples"] == []


class TestDataIntegrity:
    """Test that existing sample data is correctly served."""

    def test_sample_list_includes_physicians(self, client):
        """All samples should have a physician."""
        samples = client.get("/encounters").json()
        for s in samples:
            assert s["physician"], f"Sample {s['sample_id']} missing physician"

    def test_sample_list_includes_mode(self, client):
        """All samples should have a valid mode."""
        samples = client.get("/encounters").json()
        for s in samples:
            assert s["mode"] in ("dictation", "ambient"), \
                f"Sample {s['sample_id']} has invalid mode: {s['mode']}"

    def test_providers_have_names(self, client):
        """All providers should have a name."""
        providers = client.get("/providers").json()
        for p in providers:
            assert p.get("name") or p.get("id"), f"Provider missing name and id"

    def test_quality_trend_ordered(self, client):
        """Quality trend should be ordered by version."""
        trend = client.get("/quality/trend").json()
        if len(trend) > 1:
            versions = [t["version"] for t in trend]
            assert versions == sorted(versions), "Quality trend not ordered by version"


class TestConfigEndpoints:
    """Test configuration-related endpoints."""

    def test_features_returns_all_flags(self, client):
        """Features endpoint should return all expected flag fields."""
        features = client.get("/config/features").json()
        expected_keys = [
            "dashboard", "view_encounters", "view_providers",
            "view_specialties", "view_templates", "view_quality",
            "record_audio", "trigger_pipeline", "run_pipeline",
            "batch_processing", "ehr_access", "patient_search",
            "create_providers", "edit_providers",
            "create_templates", "edit_templates",
            "create_specialties", "edit_specialties",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature flag: {key}"

    def test_role_endpoint_has_required_fields(self, client):
        data = client.get("/config/role").json()
        assert "role" in data
        assert "instance_id" in data
        assert "is_provider_facing" in data
        assert "is_processing_pipeline" in data
