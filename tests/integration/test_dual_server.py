"""
tests/integration/test_dual_server.py — Dual-server integration tests.

Verifies that the provider-facing and processing-pipeline server roles:
1. Resolve different data/output directories
2. Mount the correct API routes for their role
3. Feature flags restrict access appropriately
4. Config endpoints report correct role information
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def provider_client(tmp_path):
    """Create a TestClient simulating provider-facing server."""
    data_dir = tmp_path / "provider-data"
    output_dir = tmp_path / "provider-output"
    data_dir.mkdir()
    output_dir.mkdir()

    env = {
        "AI_SCRIBE_DATA_DIR": str(data_dir),
        "AI_SCRIBE_OUTPUT_DIR": str(output_dir),
        "AI_SCRIBE_SERVER_ROLE": "provider-facing",
    }
    with patch.dict(os.environ, env):
        import config.paths
        importlib.reload(config.paths)
        import config.deployment
        importlib.reload(config.deployment)
        config.deployment.get_deployment_config(reload=True)
        import api.data_loader
        importlib.reload(api.data_loader)
        api.data_loader._quality_cache.clear()
        import api.main
        importlib.reload(api.main)

        client = TestClient(api.main.app)
        yield client

    # Reset
    config.deployment.get_deployment_config(reload=True)


@pytest.fixture
def pipeline_client(tmp_path):
    """Create a TestClient simulating processing-pipeline server."""
    data_dir = tmp_path / "pipeline-data"
    output_dir = tmp_path / "pipeline-output"
    data_dir.mkdir()
    output_dir.mkdir()

    env = {
        "AI_SCRIBE_DATA_DIR": str(data_dir),
        "AI_SCRIBE_OUTPUT_DIR": str(output_dir),
        "AI_SCRIBE_SERVER_ROLE": "processing-pipeline",
    }
    with patch.dict(os.environ, env):
        import config.paths
        importlib.reload(config.paths)
        import config.deployment
        importlib.reload(config.deployment)
        config.deployment.get_deployment_config(reload=True)
        import api.data_loader
        importlib.reload(api.data_loader)
        api.data_loader._quality_cache.clear()
        import api.main
        importlib.reload(api.main)

        client = TestClient(api.main.app)
        yield client

    config.deployment.get_deployment_config(reload=True)


class TestRoleReporting:
    def test_provider_reports_correct_role(self, provider_client):
        resp = provider_client.get("/config/role")
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "provider-facing"
        assert data["is_provider_facing"] is True
        assert data["is_processing_pipeline"] is False

    def test_pipeline_reports_correct_role(self, pipeline_client):
        resp = pipeline_client.get("/config/role")
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "processing-pipeline"
        assert data["is_provider_facing"] is False
        assert data["is_processing_pipeline"] is True


class TestFeatureFlags:
    def test_provider_has_ehr_access(self, provider_client):
        resp = provider_client.get("/config/features")
        data = resp.json()
        assert data["ehr_access"] is True
        assert data["patient_search"] is True

    def test_provider_no_admin_features(self, provider_client):
        resp = provider_client.get("/config/features")
        data = resp.json()
        assert data["create_providers"] is False
        assert data["edit_templates"] is False

    def test_pipeline_has_admin_features(self, pipeline_client):
        resp = pipeline_client.get("/config/features")
        data = resp.json()
        assert data["create_providers"] is True
        assert data["edit_templates"] is True
        assert data["run_pipeline"] is True

    def test_pipeline_no_ehr_access(self, pipeline_client):
        resp = pipeline_client.get("/config/features")
        data = resp.json()
        assert data["ehr_access"] is False
        assert data["patient_search"] is False


class TestSharedEndpoints:
    def test_both_serve_encounters(self, provider_client, pipeline_client):
        resp1 = provider_client.get("/encounters")
        resp2 = pipeline_client.get("/encounters")
        assert resp1.status_code == 200
        assert resp2.status_code == 200

    def test_both_serve_quality(self, provider_client, pipeline_client):
        resp1 = provider_client.get("/quality/aggregate", params={"version": "latest"})
        resp2 = pipeline_client.get("/quality/aggregate", params={"version": "latest"})
        assert resp1.status_code == 200
        assert resp2.status_code == 200

    def test_both_serve_health(self, provider_client, pipeline_client):
        resp1 = provider_client.get("/health")
        resp2 = pipeline_client.get("/health")
        assert resp1.json()["status"] == "ok"
        assert resp2.json()["status"] == "ok"


class TestDataIntegrity:
    """Verify that both server roles return populated data, not just 200 OK.

    These tests catch the class of bug where an endpoint returns 200 but
    with empty quality scores, missing fields, or null data that the UI
    needs to render correctly.
    """

    def _populate_server(self, data_dir: Path, output_dir: Path, sample_id: str = "s1"):
        """Create a sample with notes AND quality report in a server's directories."""
        import json
        mode = "dictation"
        physician = "dr_test_integrity"

        # Output dir: generated note + transcript
        out_enc = output_dir / mode / physician / sample_id
        out_enc.mkdir(parents=True, exist_ok=True)
        (out_enc / "generated_note_v9.md").write_text("# Note v9\nSOAP content")
        (out_enc / "audio_transcript_v9.txt").write_text("Transcript text")
        (out_enc / "comparison_v9.md").write_text("# Comparison\nGold vs Generated")

        # Quality report at output root
        quality_lines = [
            "# Report v9",
            "",
            "## Per-Sample Scores",
            "",
            "| Sample | Overall | Accuracy | Complete | No Halluc | Structure | Language | Overlap | Status |",
            "|--------|---------|----------|----------|-----------|-----------|----------|---------|--------|",
            f"| {sample_id} | 4.30 | 4.0 | 4.5 | 5.0 | 4.5 | 3.5 | 42% | ok |",
        ]
        (output_dir / "quality_report_v9.md").write_text("\n".join(quality_lines))

        # Data dir: gold note + demographics
        data_enc = data_dir / mode / physician / sample_id
        data_enc.mkdir(parents=True, exist_ok=True)
        (data_enc / "final_soap_note.md").write_text("# Gold Standard")
        (data_enc / "encounter_details.json").write_text(json.dumps({
            "mode": "dictation", "provider_id": physician, "visit_type": "follow_up",
        }))

    def test_pipeline_encounters_have_quality_scores(self, tmp_path):
        """Pipeline server must return quality scores when quality_report exists.

        This is the exact bug that was missed: pipeline-output/ had notes
        but no quality_report_v9.md, so all scores were null.
        """
        data_dir = tmp_path / "pl-data"
        output_dir = tmp_path / "pl-output"
        data_dir.mkdir()
        output_dir.mkdir()

        self._populate_server(data_dir, output_dir, "integrity_s1")

        env = {
            "AI_SCRIBE_DATA_DIR": str(data_dir),
            "AI_SCRIBE_OUTPUT_DIR": str(output_dir),
            "AI_SCRIBE_SERVER_ROLE": "processing-pipeline",
        }
        with patch.dict(os.environ, env):
            import config.paths, config.deployment, api.data_loader, api.main
            importlib.reload(config.paths)
            importlib.reload(config.deployment)
            config.deployment.get_deployment_config(reload=True)
            importlib.reload(api.data_loader)
            api.data_loader._quality_cache.clear()
            importlib.reload(api.main)

            client = TestClient(api.main.app)
            resp = client.get("/encounters")
            assert resp.status_code == 200
            samples = resp.json()
            assert len(samples) > 0, "Pipeline server should list samples"
            s = samples[0]
            assert s["quality"] is not None, (
                "Pipeline server encounters must have quality scores when "
                "quality_report exists in OUTPUT_DIR"
            )
            assert s["quality"]["overall"] == 4.3
            assert s["quality"]["accuracy"] == 4.0

        config.deployment.get_deployment_config(reload=True)

    def test_provider_encounters_have_quality_scores(self, tmp_path):
        """Provider server must also return quality scores."""
        data_dir = tmp_path / "pf-data"
        output_dir = tmp_path / "pf-output"
        data_dir.mkdir()
        output_dir.mkdir()

        self._populate_server(data_dir, output_dir, "integrity_s2")

        env = {
            "AI_SCRIBE_DATA_DIR": str(data_dir),
            "AI_SCRIBE_OUTPUT_DIR": str(output_dir),
            "AI_SCRIBE_SERVER_ROLE": "provider-facing",
        }
        with patch.dict(os.environ, env):
            import config.paths, config.deployment, api.data_loader, api.main
            importlib.reload(config.paths)
            importlib.reload(config.deployment)
            config.deployment.get_deployment_config(reload=True)
            importlib.reload(api.data_loader)
            api.data_loader._quality_cache.clear()
            importlib.reload(api.main)

            client = TestClient(api.main.app)
            resp = client.get("/encounters")
            samples = resp.json()
            s = samples[0]
            assert s["quality"] is not None, (
                "Provider server encounters must have quality scores"
            )
            assert s["quality"]["overall"] == 4.3

        config.deployment.get_deployment_config(reload=True)


class TestLatestVersionEndpoint:
    def test_provider_returns_latest_version(self, provider_client):
        resp = provider_client.get("/config/latest-version")
        assert resp.status_code == 200
        data = resp.json()
        assert "latest" in data
        assert "versions" in data

    def test_pipeline_returns_latest_version(self, pipeline_client):
        resp = pipeline_client.get("/config/latest-version")
        assert resp.status_code == 200
        data = resp.json()
        assert "latest" in data
