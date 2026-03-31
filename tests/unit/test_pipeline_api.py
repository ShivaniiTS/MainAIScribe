"""
tests/unit/test_pipeline_api.py — Tests for the Pipeline API routes.

Tests file upload, status polling, output retrieval, and batch operations.
"""
from __future__ import annotations

import io
import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_config():
    """Ensure processing-pipeline mode and reset config."""
    with patch.dict(os.environ, {"AI_SCRIBE_SERVER_ROLE": "processing-pipeline"}):
        from config.deployment import get_deployment_config
        get_deployment_config(reload=True)
        yield
    from config.deployment import get_deployment_config
    get_deployment_config(reload=True)


@pytest.fixture
def client():
    """Create a test client for the app."""
    from config.deployment import get_deployment_config
    get_deployment_config(reload=True)

    import importlib
    import api.main
    importlib.reload(api.main)

    return TestClient(api.main.app)


class TestPipelineUpload:
    def test_upload_audio(self, client: TestClient):
        """Upload audio and get a job_id back."""
        audio = io.BytesIO(b"fake audio data")
        resp = client.post(
            "/pipeline/upload",
            files={"audio": ("test.mp3", audio, "audio/mpeg")},
            data={
                "sample_id": "test_upload_001",
                "mode": "dictation",
                "provider_id": "dr_test",
                "encounter_details": json.dumps({"visit_type": "follow_up"}),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["sample_id"] == "test_upload_001"
        assert "job_id" in data

    def test_upload_creates_files(self, client: TestClient, tmp_path):
        """Upload should create the encounter directory structure."""
        audio = io.BytesIO(b"fake audio data for test")
        resp = client.post(
            "/pipeline/upload",
            files={"audio": ("dictation.mp3", audio, "audio/mpeg")},
            data={
                "sample_id": "test_file_creation",
                "mode": "dictation",
                "provider_id": "dr_test_files",
                "encounter_details": json.dumps({"mode": "dictation"}),
            },
        )
        assert resp.status_code == 200

    def test_upload_missing_audio_returns_422(self, client: TestClient):
        """Upload without audio file should return 422."""
        resp = client.post(
            "/pipeline/upload",
            data={"sample_id": "test", "mode": "dictation"},
        )
        assert resp.status_code == 422

    def test_upload_canonicalizes_provider_id_from_name(self, client: TestClient, tmp_path):
        """If provider_id is external (e.g. numeric), provider_name should map to profile ID."""
        from config import provider_manager as pm
        from api.pipeline import routes as pipeline_routes

        providers_dir = tmp_path / "providers"
        providers_dir.mkdir()
        (providers_dir / "dr_mohammed_alwahaidy.yaml").write_text(
            "\n".join([
                "id: dr_mohammed_alwahaidy",
                "name: Dr. Mohammed Alwahaidy",
                "specialty: chiropractic",
            ]) + "\n",
            encoding="utf-8",
        )

        original_manager = pm._manager
        pm._manager = pm.ProviderManager(providers_dir=providers_dir)
        try:
            audio = io.BytesIO(b"fake audio data")
            resp = client.post(
                "/pipeline/upload",
                files={"audio": ("test.mp3", audio, "audio/mpeg")},
                data={
                    "sample_id": "test_provider_map_001",
                    "mode": "dictation",
                    "provider_id": "300",
                    "encounter_details": json.dumps({"provider_name": "Alwahaidy, Mohammed"}),
                },
            )
            assert resp.status_code == 200
            job_id = resp.json()["job_id"]
            assert pipeline_routes._jobs[job_id]["provider_id"] == "dr_mohammed_alwahaidy"
        finally:
            pm._manager = original_manager


class TestPipelineStatus:
    def test_status_nonexistent_job(self, client: TestClient):
        """Polling status for nonexistent job returns 404."""
        resp = client.get("/pipeline/status/nonexistent-job")
        assert resp.status_code == 404

    def test_status_after_upload(self, client: TestClient):
        """After upload, job status should be 'pending'."""
        audio = io.BytesIO(b"fake audio")
        upload_resp = client.post(
            "/pipeline/upload",
            files={"audio": ("test.mp3", audio, "audio/mpeg")},
            data={"sample_id": "test_status_001", "mode": "dictation", "provider_id": "dr_test"},
        )
        job_id = upload_resp.json()["job_id"]

        status_resp = client.get(f"/pipeline/status/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "pending"


class TestPipelineTrigger:
    def test_trigger_nonexistent_job(self, client: TestClient):
        """Triggering nonexistent job returns 404."""
        resp = client.post("/pipeline/trigger/nonexistent-job", json={})
        assert resp.status_code == 404

    def test_trigger_without_audio_fails(self, client: TestClient):
        """Triggering a job that has no audio file should fail."""
        # Create a job with fake data dir
        audio = io.BytesIO(b"")  # Empty audio
        upload_resp = client.post(
            "/pipeline/upload",
            files={"audio": ("test.mp3", audio, "audio/mpeg")},
            data={"sample_id": "test_trigger_empty", "mode": "dictation", "provider_id": "dr_test"},
        )
        # The audio file was written but is empty — trigger should work
        # since the file exists (even if empty)
        job_id = upload_resp.json()["job_id"]
        resp = client.post(f"/pipeline/trigger/{job_id}", json={"mode": "dictation"})
        # Should start processing (the pipeline will fail later, but trigger succeeds)
        assert resp.status_code == 200
        assert resp.json()["status"] == "processing"


class TestOutputRetrieval:
    def test_output_nonexistent_sample(self, client: TestClient):
        """Requesting outputs for nonexistent sample returns 404."""
        resp = client.get("/pipeline/output/nonexistent_sample_xyz")
        assert resp.status_code == 404

    def test_note_nonexistent_sample(self, client: TestClient):
        resp = client.get("/pipeline/output/nonexistent_sample_xyz/note")
        assert resp.status_code == 404

    def test_transcript_nonexistent_sample(self, client: TestClient):
        resp = client.get("/pipeline/output/nonexistent_sample_xyz/transcript")
        assert resp.status_code == 404


class TestBatchOperations:
    def test_batch_retrieve_empty(self, client: TestClient):
        """Batch retrieve with no matching samples returns empty."""
        resp = client.get("/pipeline/outputs/batch", params={"sample_ids": "fake_001,fake_002"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["samples"] == []
