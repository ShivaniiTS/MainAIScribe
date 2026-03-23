"""
tests/integration/test_api_encounters.py — Integration tests for encounter API.

Tests the full encounter lifecycle: list, detail, notes, transcripts, quality.
Uses the actual data directories (requires test data to be present).
"""
from __future__ import annotations

import os
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


class TestEncounterList:
    def test_list_returns_list(self, client):
        resp = client.get("/encounters")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_list_filter_by_mode(self, client):
        resp = client.get("/encounters", params={"mode": "dictation"})
        assert resp.status_code == 200
        for item in resp.json():
            assert item["mode"] == "dictation"

    def test_list_sample_has_expected_fields(self, client):
        resp = client.get("/encounters")
        data = resp.json()
        if data:
            sample = data[0]
            assert "sample_id" in sample
            assert "mode" in sample
            assert "physician" in sample
            assert "versions" in sample


class TestEncounterDetail:
    def test_detail_not_found(self, client):
        resp = client.get("/encounters/nonexistent_sample_xyz")
        assert resp.status_code == 404

    def test_detail_returns_context(self, client):
        # First get a valid sample_id
        list_resp = client.get("/encounters")
        samples = list_resp.json()
        if not samples:
            pytest.skip("No samples available")

        sample_id = samples[0]["sample_id"]
        resp = client.get(f"/encounters/{sample_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sample_id"] == sample_id


class TestEncounterNote:
    def test_note_not_found(self, client):
        resp = client.get("/encounters/nonexistent/note")
        assert resp.status_code == 404

    def test_note_with_valid_sample(self, client):
        list_resp = client.get("/encounters")
        samples = [s for s in list_resp.json() if s.get("latest_version")]
        if not samples:
            pytest.skip("No samples with notes available")

        sample = samples[0]
        resp = client.get(f"/encounters/{sample['sample_id']}/note",
                         params={"version": sample["latest_version"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert len(data["content"]) > 0


class TestEncounterTranscript:
    def test_transcript_with_valid_sample(self, client):
        list_resp = client.get("/encounters")
        samples = [s for s in list_resp.json() if s.get("latest_version")]
        if not samples:
            pytest.skip("No samples with transcripts available")

        sample = samples[0]
        resp = client.get(f"/encounters/{sample['sample_id']}/transcript",
                         params={"version": sample["latest_version"]})
        # Transcript may or may not exist
        assert resp.status_code in (200, 404)


class TestEncounterAudio:
    def test_audio_streaming(self, client):
        list_resp = client.get("/encounters")
        samples = list_resp.json()
        if not samples:
            pytest.skip("No samples available")

        resp = client.get(f"/encounters/{samples[0]['sample_id']}/audio")
        # Audio may or may not exist
        assert resp.status_code in (200, 404)


class TestQualityRoutes:
    def test_aggregate(self, client):
        resp = client.get("/quality/aggregate")
        assert resp.status_code == 200

    def test_trend(self, client):
        resp = client.get("/quality/trend")
        assert resp.status_code == 200
        data = resp.json()
        # Endpoint returns {"trend": [...]} dict
        assert isinstance(data, (list, dict))

    def test_dimensions(self, client):
        resp = client.get("/quality/dimensions")
        assert resp.status_code == 200

    def test_by_mode(self, client):
        resp = client.get("/quality/by-mode")
        assert resp.status_code == 200

    def test_by_provider(self, client):
        resp = client.get("/quality/by-provider")
        assert resp.status_code == 200
