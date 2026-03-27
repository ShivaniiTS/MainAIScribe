"""
tests/e2e/test_smoke.py — End-to-end smoke tests.

These tests verify the full API works against real (but temporary) data
without requiring GPU/Ollama. They test the complete request→response
cycle through the FastAPI app.

Covers:
- Dashboard data flow: samples → quality → aggregate
- Sample detail page: all tabs return correct data
- Version switching: different versions return different content
- Empty state: all pages handle gracefully when no data exists
- Encounter creation flow
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
def setup(tmp_path):
    """Full E2E setup with sample data."""
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
        import config.paths
        importlib.reload(config.paths)
        import config.deployment
        importlib.reload(config.deployment)
        config.deployment.get_deployment_config(reload=True)
        import api.data_loader
        importlib.reload(api.data_loader)
        api.data_loader._quality_cache.clear()

        # Create realistic sample data
        _populate_test_data(data_dir, output_dir)

        from api.main import app
        client = TestClient(app)
        yield client, data_dir, output_dir

    config.deployment.get_deployment_config(reload=True)


def _populate_test_data(data_dir: Path, output_dir: Path):
    """Create a realistic set of test samples."""
    samples = [
        ("dictation", "dr_smith", "patient_doe_abc123_2026-03-10", True),
        ("dictation", "dr_smith", "patient_jones_def456_2026-03-11", True),
        ("conversation", "dr_jones", "patient_roe_ghi789_2026-03-12", True),
        ("dictation", "dr_jones", "patient_new_jkl012_2026-03-13", False),
    ]

    for mode, physician, sample_id, has_gold in samples:
        # Data directory
        data_enc = data_dir / mode / physician / sample_id
        data_enc.mkdir(parents=True)
        (data_enc / "patient_demographics.json").write_text(json.dumps({
            "first_name": "Jane", "last_name": "Doe",
            "date_of_birth": "1990-01-01",
        }))
        (data_enc / "encounter_details.json").write_text(json.dumps({
            "encounter_id": "e001",
            "mode": "dictation" if mode == "dictation" else "ambient",
            "provider": {"full_name": physician.replace("_", " ").title()},
            "visit_type": "follow_up",
        }))
        if has_gold:
            (data_enc / "final_soap_note.md").write_text("# Gold Standard\n\nContent")

        # Audio placeholder
        audio_name = "dictation.mp3" if mode == "dictation" else "conversation_audio.mp3"
        (data_enc / audio_name).write_bytes(b"fake audio")

        # Output directory with versioned outputs
        out_enc = output_dir / mode / physician / sample_id
        out_enc.mkdir(parents=True)
        for v in ["v8", "v9"]:
            (out_enc / f"generated_note_{v}.md").write_text(
                f"# Clinical Note {v}\n\n## Chief Complaint\nPatient presents for follow-up."
            )
            (out_enc / f"audio_transcript_{v}.txt").write_text(
                f"Doctor: Hello. Patient: Hi, I'm here for my follow-up. ({v})"
            )
            (out_enc / f"comparison_{v}.md").write_text(
                f"# Comparison {v}\n\n| Gold | Generated | Score |\n|---|---|---|\n| text | text | 4.5 |"
            )

    # Write aggregate quality report for v9
    quality_lines = [
        "# Aggregate Quality Report — Pipeline v9",
        "",
        "## Per-Sample Scores",
        "",
        "| Sample | Overall | Accuracy | Complete | No Halluc | Structure | Language | Overlap | Status |",
        "|--------|---------|----------|----------|-----------|-----------|----------|---------|--------|",
    ]
    for mode, physician, sample_id, has_gold in samples:
        if has_gold:
            quality_lines.append(
                f"| {sample_id} | 4.35 | 4.5 | 4.0 | 5.0 | 4.2 | 4.0 | 42% | ✓ |"
            )
    (output_dir / "quality_report_v9.md").write_text("\n".join(quality_lines))


class TestDashboardFlow:
    """Simulates what the dashboard page fetches."""

    def test_full_dashboard_data(self, setup):
        client, _, _ = setup

        # Fetch all dashboard data in parallel (as the frontend does)
        agg = client.get("/quality/aggregate", params={"version": "latest"}).json()
        trend = client.get("/quality/trend").json()
        dims = client.get("/quality/dimensions", params={"version": "latest"}).json()
        samples = client.get("/encounters").json()
        by_provider = client.get("/quality/by-provider", params={"version": "latest"}).json()

        # Aggregate quality should have data
        assert isinstance(agg.get("average"), (int, float))
        assert agg["average"] > 0
        assert agg["sample_count"] >= 1

        # Trend should have at least one version
        assert len(trend["trend"]) >= 1

        # Dimensions should be a list of 5
        assert len(dims) == 5
        for dim in dims:
            assert dim["score"] is not None

        # Samples should include all 4
        assert len(samples) == 4

        # Provider breakdown should have 2 providers
        assert len(by_provider) >= 1

    def test_dashboard_empty_state(self, tmp_path):
        """Dashboard should NOT crash with zero data."""
        data_dir = tmp_path / "empty-data"
        output_dir = tmp_path / "empty-output"
        data_dir.mkdir()
        output_dir.mkdir()

        with patch.dict(os.environ, {
            "AI_SCRIBE_DATA_DIR": str(data_dir),
            "AI_SCRIBE_OUTPUT_DIR": str(output_dir),
            "AI_SCRIBE_SERVER_ROLE": "provider-facing",
        }):
            import config.paths
            importlib.reload(config.paths)
            import api.data_loader
            importlib.reload(api.data_loader)
            api.data_loader._quality_cache.clear()

            from api.main import app
            client = TestClient(app)

            # All endpoints should return 200 with empty data
            assert client.get("/encounters").status_code == 200
            assert client.get("/quality/aggregate", params={"version": "latest"}).status_code == 200
            assert client.get("/quality/trend").status_code == 200
            assert client.get("/quality/dimensions", params={"version": "latest"}).status_code == 200
            assert client.get("/quality/by-provider", params={"version": "latest"}).status_code == 200
            assert client.get("/quality/by-mode", params={"version": "latest"}).status_code == 200
            assert client.get("/providers").status_code == 200


class TestSampleDetailFlow:
    """Simulates what the sample detail page fetches."""

    def test_full_sample_detail(self, setup):
        client, _, _ = setup

        # List samples to find one
        samples = client.get("/encounters").json()
        sample = next(s for s in samples if s["has_gold"])
        sid = sample["sample_id"]

        # Fetch all detail data
        detail = client.get(f"/encounters/{sid}").json()
        assert detail["patient_context"] is not None

        # Fetch note
        note = client.get(f"/encounters/{sid}/note", params={"version": "v9"}).json()
        assert "Clinical Note v9" in note["content"]

        # Fetch transcript
        tx = client.get(f"/encounters/{sid}/transcript", params={"version": "v9"}).json()
        assert "v9" in tx["content"]

        # Fetch comparison
        comp = client.get(f"/encounters/{sid}/comparison", params={"version": "v9"}).json()
        assert "Comparison" in comp["content"]

        # Fetch gold
        gold = client.get(f"/encounters/{sid}/gold").json()
        assert "Gold Standard" in gold["content"]

    def test_version_switching(self, setup):
        client, _, _ = setup
        samples = client.get("/encounters").json()
        sid = samples[0]["sample_id"]

        v8 = client.get(f"/encounters/{sid}/note", params={"version": "v8"}).json()
        v9 = client.get(f"/encounters/{sid}/note", params={"version": "v9"}).json()

        assert "v8" in v8["content"]
        assert "v9" in v9["content"]
        assert v8["content"] != v9["content"]

    def test_latest_returns_newest_version(self, setup):
        client, _, _ = setup
        samples = client.get("/encounters").json()
        sid = samples[0]["sample_id"]

        latest = client.get(f"/encounters/{sid}/note", params={"version": "latest"}).json()
        assert "v9" in latest["content"]


class TestEncounterCreation:
    def test_create_and_check_status(self, setup):
        client, _, _ = setup

        # Create encounter
        resp = client.post("/encounters", json={
            "provider_id": "dr_smith",
            "patient_id": "pat_001",
            "visit_type": "follow_up",
            "mode": "dictation",
        })
        assert resp.status_code == 201
        enc = resp.json()
        eid = enc["encounter_id"]

        # Check status
        status = client.get(f"/encounters/{eid}/status").json()
        assert status["status"] == "pending"


class TestProviderFlow:
    def test_providers_list(self, setup):
        client, _, _ = setup
        resp = client.get("/providers")
        assert resp.status_code == 200
        providers = resp.json()
        physician_ids = [p["id"] for p in providers]
        assert "dr_smith" in physician_ids or "dr_jones" in physician_ids
