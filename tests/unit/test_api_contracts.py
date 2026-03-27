"""
tests/unit/test_api_contracts.py — API endpoint contract tests.

Tests every API endpoint to verify:
- Correct HTTP status codes for success and error cases
- Response shape matches expected schema
- Edge cases: empty data, missing samples, missing versions
- Version="latest" parameter resolution
- Graceful handling of empty quality data (dashboard bug fix)

Uses FastAPI TestClient with mocked data directories.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_env(tmp_path):
    """Set up isolated data directories and reload modules."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()

    env = {
        "AI_SCRIBE_DATA_DIR": str(data_dir),
        "AI_SCRIBE_OUTPUT_DIR": str(output_dir),
        "AI_SCRIBE_SERVER_ROLE": "provider-facing",
    }
    with patch.dict(os.environ, env):
        # Reload path-dependent modules
        import config.paths
        importlib.reload(config.paths)
        import api.data_loader
        importlib.reload(api.data_loader)
        api.data_loader._quality_cache.clear()
        yield data_dir, output_dir


@pytest.fixture
def client(mock_env):
    """Create a FastAPI test client."""
    from api.main import app
    return TestClient(app)


def _create_sample(output_dir: Path, data_dir: Path, mode: str, physician: str,
                   sample_id: str, versions: list[str], has_gold: bool = True):
    """Create a sample in mock directories."""
    out_enc = output_dir / mode / physician / sample_id
    out_enc.mkdir(parents=True, exist_ok=True)
    for v in versions:
        (out_enc / f"generated_note_{v}.md").write_text(f"# Note {v}\nContent here")
        (out_enc / f"audio_transcript_{v}.txt").write_text(f"Transcript for {v}")
        (out_enc / f"comparison_{v}.md").write_text(f"# Comparison {v}\nGold vs Generated")

    data_enc = data_dir / mode / physician / sample_id
    data_enc.mkdir(parents=True, exist_ok=True)
    if has_gold:
        (data_enc / "final_soap_note.md").write_text("# Gold Standard Note")
    (data_enc / "patient_demographics.json").write_text(json.dumps({
        "first_name": "John", "last_name": "Doe",
        "date_of_birth": "1990-05-20",
    }))
    (data_enc / "encounter_details.json").write_text(json.dumps({
        "encounter_id": "e001", "mode": "dictation" if mode == "dictation" else "ambient",
        "provider": {"full_name": "Dr. Smith"}, "visit_type": "follow_up",
    }))


# ── Health endpoints ──

class TestHealthEndpoints:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "AI Scribe Enterprise API"
        assert "role" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_config_features(self, client):
        resp = client.get("/config/features")
        assert resp.status_code == 200
        data = resp.json()
        assert "dashboard" in data

    def test_config_role(self, client):
        resp = client.get("/config/role")
        assert resp.status_code == 200
        assert "role" in resp.json()

    def test_config_latest_version(self, client):
        resp = client.get("/config/latest-version")
        assert resp.status_code == 200
        data = resp.json()
        assert "latest" in data
        assert "versions" in data


# ── Encounters endpoints ──

class TestEncountersEndpoints:
    def test_list_encounters_empty(self, client):
        resp = client.get("/encounters")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_encounters_with_data(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v9"])
        resp = client.get("/encounters")
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) == 1
        assert samples[0]["sample_id"] == "s1"
        assert samples[0]["mode"] == "dictation"
        assert "v9" in samples[0]["versions"]

    def test_get_encounter_not_found(self, client):
        resp = client.get("/encounters/nonexistent")
        assert resp.status_code == 404

    def test_get_encounter_with_data(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v9"])
        resp = client.get("/encounters/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sample_id"] == "s1"
        assert data["patient_context"] is not None
        assert data["patient_context"]["patient"]["name"] == "John Doe"

    def test_get_note_not_found(self, client):
        resp = client.get("/encounters/missing/note", params={"version": "v1"})
        assert resp.status_code == 404

    def test_get_note_with_data(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v9"])
        resp = client.get("/encounters/s1/note", params={"version": "v9"})
        assert resp.status_code == 200
        assert "content" in resp.json()

    def test_get_note_latest_version(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v5", "v9"])
        resp = client.get("/encounters/s1/note", params={"version": "latest"})
        assert resp.status_code == 200
        assert "Note v9" in resp.json()["content"]

    def test_get_note_wrong_version(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v9"])
        resp = client.get("/encounters/s1/note", params={"version": "v1"})
        assert resp.status_code == 404

    def test_get_comparison_not_found(self, client):
        resp = client.get("/encounters/missing/comparison", params={"version": "v1"})
        assert resp.status_code == 404

    def test_get_gold_not_found(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v9"], has_gold=False)
        resp = client.get("/encounters/s1/gold")
        assert resp.status_code == 404

    def test_get_gold_with_data(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v9"], has_gold=True)
        resp = client.get("/encounters/s1/gold")
        assert resp.status_code == 200
        assert "Gold Standard Note" in resp.json()["content"]

    def test_get_transcript(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v9"])
        resp = client.get("/encounters/s1/transcript", params={"version": "v9"})
        assert resp.status_code == 200
        assert "content" in resp.json()
        assert "versions" in resp.json()

    def test_get_quality_not_found(self, client):
        resp = client.get("/encounters/missing/quality", params={"version": "v1"})
        assert resp.status_code == 404

    def test_get_versions(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v5", "v9"])
        resp = client.get("/encounters/s1/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sample_id"] == "s1"
        assert "v9" in data["versions"]
        assert "v5" in data["versions"]

    def test_versions_not_found(self, client):
        resp = client.get("/encounters/nonexistent/versions")
        assert resp.status_code == 404

    def test_is_test_flag_on_test_samples(self, client, mock_env):
        """Test samples must have is_test=True so the UI can badge them."""
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_test", "test_sample_001", ["v9"])
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "real_sample_001", ["v9"])
        resp = client.get("/encounters")
        assert resp.status_code == 200
        samples = resp.json()
        test_sample = next(s for s in samples if s["sample_id"] == "test_sample_001")
        real_sample = next(s for s in samples if s["sample_id"] == "real_sample_001")
        assert test_sample["is_test"] is True, "Test sample must have is_test=True"
        assert real_sample["is_test"] is False, "Real sample must have is_test=False"


# ── Quality endpoints ──

class TestQualityEndpoints:
    def _write_quality_report(self, output_dir: Path, version: str):
        lines = [
            f"# Report {version}",
            "",
            "## Per-Sample Scores",
            "",
            "| Sample | Overall | Accuracy | Complete | No Halluc | Structure | Language | Overlap | Status |",
            "|--------|---------|----------|----------|-----------|-----------|----------|---------|--------|",
            "| s1 | 4.50 | 4.5 | 4.0 | 5.0 | 4.5 | 4.0 | 45% | ✓ |",
        ]
        (output_dir / f"quality_report_{version}.md").write_text("\n".join(lines))

    def test_aggregate_empty(self, client):
        """Dashboard should NOT crash when no quality data exists."""
        resp = client.get("/quality/aggregate", params={"version": "latest"})
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_aggregate_with_data(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v9"])
        self._write_quality_report(output_dir, "v9")
        # Clear cache
        import api.data_loader
        api.data_loader._quality_cache.clear()
        resp = client.get("/quality/aggregate", params={"version": "v9"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["average"] == 4.5
        assert data["sample_count"] == 1

    def test_quality_scores_populated_on_samples(self, client, mock_env):
        """Samples with a quality report must have quality scores in the list response.
        This catches the bug where pipeline-output/ had notes but no quality report."""
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v9"])
        self._write_quality_report(output_dir, "v9")
        import api.data_loader
        api.data_loader._quality_cache.clear()
        resp = client.get("/encounters")
        assert resp.status_code == 200
        samples = resp.json()
        s1 = next(s for s in samples if s["sample_id"] == "s1")
        assert s1["quality"] is not None, "Sample with quality report must have quality scores"
        assert s1["quality"]["overall"] == 4.5, "Quality overall score must match report"
        assert s1["quality"]["accuracy"] == 4.5
        assert s1["quality"]["no_hallucination"] == 5.0

    def test_quality_scores_none_without_report(self, client, mock_env):
        """Samples without a quality report must have quality=None, not empty dict."""
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v9"])
        # No quality report written
        import api.data_loader
        api.data_loader._quality_cache.clear()
        resp = client.get("/encounters")
        samples = resp.json()
        s1 = next(s for s in samples if s["sample_id"] == "s1")
        assert s1["quality"] is None, "Sample without quality report must have quality=None"

    def test_aggregate_latest_resolves(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v9"])
        self._write_quality_report(output_dir, "v9")
        import api.data_loader
        api.data_loader._quality_cache.clear()
        resp = client.get("/quality/aggregate", params={"version": "latest"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data.get("average"), (int, float))

    def test_trend_empty(self, client):
        resp = client.get("/quality/trend")
        assert resp.status_code == 200
        assert resp.json() == {"trend": []}

    def test_dimensions_empty(self, client):
        resp = client.get("/quality/dimensions", params={"version": "latest"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # All scores should be None when no data
        for dim in data:
            assert dim["score"] is None

    def test_by_provider_empty(self, client):
        resp = client.get("/quality/by-provider", params={"version": "latest"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_by_mode_empty(self, client):
        resp = client.get("/quality/by-mode", params={"version": "latest"})
        assert resp.status_code == 200
        assert resp.json() == {}


# ── Create encounter endpoints ──

class TestCreateEncounter:
    def test_create_encounter(self, client):
        resp = client.post("/encounters", json={
            "provider_id": "dr_test",
            "patient_id": "pat_001",
            "visit_type": "follow_up",
            "mode": "dictation",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "pending"
        assert "encounter_id" in data

    def test_get_status_not_found(self, client):
        resp = client.get("/encounters/nonexistent/status")
        assert resp.status_code == 404


# ── Providers endpoints ──

class TestProvidersEndpoints:
    def test_list_providers_empty(self, client):
        resp = client.get("/providers")
        assert resp.status_code == 200
        # May return providers from YAML profiles

    def test_list_providers_discovers_from_data(self, client, mock_env):
        data_dir, output_dir = mock_env
        _create_sample(output_dir, data_dir, "dictation", "dr_new_physician", "s1", ["v9"])
        resp = client.get("/providers")
        assert resp.status_code == 200
        providers = resp.json()
        ids = [p["id"] for p in providers]
        assert "dr_new_physician" in ids
