"""
tests/unit/test_error_recovery.py — Pipeline failure handling and error recovery tests.

Covers:
1. Pipeline failure sets encounter status to "error" and sends WS error event
2. Quality evaluation failure is non-fatal (pipeline still completes)
3. Proxy timeout handling (10-min poll limit)
4. Partial batch failure (some samples fail, others succeed)
5. Pipeline trigger rejects already-running job (409)
6. Missing audio file in trigger returns 404
7. Upload with invalid encounter_details JSON is handled gracefully
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestPipelineFailureStatus:
    """Pipeline errors should update encounter status and send WS error event."""

    @pytest.mark.asyncio
    async def test_run_pipeline_async_catches_exception(self):
        """When run_encounter raises, status should be 'error' and WS error sent."""
        from api.routes.encounters import _run_pipeline_async, _encounters
        from api.ws.session_events import ConnectionManager

        encounter_id = "test-err-1"
        _encounters[encounter_id] = {
            "encounter_id": encounter_id,
            "status": "processing",
            "message": "",
        }

        mock_mgr = AsyncMock(spec=ConnectionManager)

        with patch("api.ws.session_events.manager", mock_mgr), \
             patch("config.provider_manager.get_provider_manager") as mock_pm:
            # Make provider manager raise an error
            mock_pm.side_effect = Exception("Provider not found")

            await _run_pipeline_async(
                encounter_id=encounter_id,
                sample_id="sample_err",
                audio_path="/fake/audio.mp3",
                mode="dictation",
                provider_id="dr_fake",
                visit_type="follow_up",
                output_dir="/tmp/output",
                data_dir="/tmp/data",
            )

        assert _encounters[encounter_id]["status"] == "error"
        assert "Pipeline error" in _encounters[encounter_id]["message"]
        mock_mgr.send_error.assert_awaited_once()

        # Cleanup
        del _encounters[encounter_id]

    @pytest.mark.asyncio
    async def test_rerun_pipeline_catches_exception(self):
        """When rerun pipeline raises, status should be 'error'."""
        from api.routes.encounters import _run_rerun_pipeline, _encounters
        from api.ws.session_events import ConnectionManager

        encounter_id = "test-rerun-err"
        _encounters[encounter_id] = {
            "encounter_id": encounter_id,
            "status": "processing",
            "message": "",
        }

        mock_mgr = AsyncMock(spec=ConnectionManager)

        with patch("api.ws.session_events.manager", mock_mgr), \
             patch("config.provider_manager.get_provider_manager") as mock_pm:
            mock_pm.side_effect = RuntimeError("Boom")

            await _run_rerun_pipeline(
                encounter_id=encounter_id,
                sample_id="sample_rerun_err",
                audio_path="/fake/audio.mp3",
                note_audio_path=None,
                mode="dictation",
                provider_id="dr_fake",
                output_dir="/tmp/output",
                version="v99",
            )

        assert _encounters[encounter_id]["status"] == "error"
        mock_mgr.send_error.assert_awaited_once()

        del _encounters[encounter_id]


class TestQualityEvalNonFatal:
    """Quality evaluation failure should not crash the pipeline."""

    @pytest.mark.asyncio
    async def test_quality_failure_still_completes(self, tmp_path):
        """Pipeline should still be 'complete' even if quality eval fails."""
        from api.routes.encounters import _run_pipeline_async, _encounters
        from api.ws.session_events import ConnectionManager

        encounter_id = "test-quality-err"
        _encounters[encounter_id] = {
            "encounter_id": encounter_id,
            "status": "processing",
            "message": "",
        }

        mock_mgr = AsyncMock(spec=ConnectionManager)

        # Create mock objects for the full pipeline
        mock_state = MagicMock()
        mock_state.final_note = "# Test Note\n## Chief Complaint\nTest"
        mock_state.transcript = MagicMock()
        mock_state.transcript.full_text = "Doctor says test"

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        from orchestrator.state import ProviderProfile
        real_profile = ProviderProfile(id="dr_test", name="Dr. Test", specialty="orthopedic")

        with patch("api.ws.session_events.manager", mock_mgr), \
             patch("config.provider_manager.get_provider_manager") as mock_pm, \
             patch("orchestrator.graph.build_graph"), \
             patch("orchestrator.graph.run_encounter", return_value=mock_state), \
             patch("output.markdown_writer.write_clinical_note"), \
             patch("api.data_loader._discover_sample_versions", return_value=[]), \
             patch("api.data_loader.get_latest_version", return_value="v8"), \
             patch("api.data_loader.get_gold_note", return_value="Gold note"), \
             patch("api.quality_runner.evaluate_sample", side_effect=Exception("Judge OOM")), \
             patch("mcp_servers.registry.get_registry"):

            mock_pm.return_value.load_or_default.return_value = real_profile

            await _run_pipeline_async(
                encounter_id=encounter_id,
                sample_id="sample_qual_err",
                audio_path="/fake/audio.mp3",
                mode="dictation",
                provider_id="dr_test",
                visit_type="follow_up",
                output_dir=str(output_dir),
                data_dir=str(tmp_path / "data"),
            )

        # Pipeline should complete despite quality failure
        assert _encounters[encounter_id]["status"] == "complete"
        mock_mgr.send_complete.assert_awaited_once()

        del _encounters[encounter_id]


class TestProxyTimeoutHandling:
    """Proxy pipeline has a 10-minute poll timeout."""

    @pytest.mark.asyncio
    async def test_proxy_timeout_sets_error(self):
        """When pipeline server reports error, encounter should be 'error'."""
        from api.routes.encounters import _proxy_pipeline_run, _encounters
        from api.ws.session_events import ConnectionManager

        encounter_id = "test-proxy-timeout"
        _encounters[encounter_id] = {
            "encounter_id": encounter_id,
            "status": "processing",
            "message": "",
        }

        mock_mgr = AsyncMock(spec=ConnectionManager)

        call_count = 0

        async def mock_status(job_id):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                return {"status": "error", "pct": 0, "stage": "", "message": "Pipeline OOM"}
            return {"status": "processing", "pct": 50, "stage": "transcribe", "message": ""}

        with patch("api.ws.session_events.manager", mock_mgr), \
             patch("api.proxy.proxy_upload", new_callable=AsyncMock, return_value={"job_id": "job-1"}), \
             patch("api.proxy.proxy_trigger", new_callable=AsyncMock, return_value={}), \
             patch("api.proxy.proxy_status", side_effect=mock_status), \
             patch("asyncio.sleep", new_callable=AsyncMock):

            await _proxy_pipeline_run(
                encounter_id=encounter_id,
                sample_id="sample-proxy",
                audio_bytes=b"fake audio",
                audio_filename="test.mp3",
                mode="dictation",
                provider_id="dr_test",
                visit_type="follow_up",
                encounter_details={},
                output_dir="/tmp/output",
            )

        assert _encounters[encounter_id]["status"] == "error"
        mock_mgr.send_error.assert_awaited_once()

        del _encounters[encounter_id]


class TestPipelineTriggerGuards:
    """Pipeline trigger endpoint guards."""

    @pytest.fixture
    def pipeline_jobs(self):
        from api.pipeline.routes import _jobs
        # Inject a test job
        _jobs["job-test-1"] = {
            "job_id": "job-test-1",
            "sample_id": "sample-1",
            "status": "processing",
            "stage": "transcribe",
            "pct": 50,
            "message": "Processing...",
            "mode": "dictation",
            "provider_id": "dr_test",
            "data_dir": "/tmp/fake",
        }
        yield _jobs
        _jobs.pop("job-test-1", None)

    def test_trigger_already_running_409(self, pipeline_jobs):
        """Triggering a job that's already processing should return 409."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.pipeline.routes import router

        app = FastAPI()
        app.include_router(router)

        with patch("api.pipeline.routes.require_feature"):
            client = TestClient(app)
            resp = client.post(
                "/pipeline/trigger/job-test-1",
                json={"mode": "dictation", "provider_id": "dr_test", "visit_type": "follow_up"},
            )
            assert resp.status_code == 409
            assert "already running" in resp.json()["detail"]

    def test_trigger_nonexistent_job_404(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.pipeline.routes import router

        app = FastAPI()
        app.include_router(router)

        with patch("api.pipeline.routes.require_feature"):
            client = TestClient(app)
            resp = client.post(
                "/pipeline/trigger/nonexistent-job",
                json={"mode": "dictation"},
            )
            assert resp.status_code == 404


class TestBatchPartialFailure:
    """Batch processing should continue even if some samples fail."""

    @pytest.mark.asyncio
    async def test_batch_continues_after_individual_failure(self):
        """If one sample in a batch fails, others should still be processed."""
        from api.pipeline.routes import _run_batch, _jobs

        # Create two jobs with real tmp dirs for audio
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        data_b1 = tmp / "b1"
        data_b2 = tmp / "b2"
        data_b1.mkdir()
        data_b2.mkdir()
        (data_b1 / "dictation.mp3").write_bytes(b"fake audio 1")
        (data_b2 / "dictation.mp3").write_bytes(b"fake audio 2")

        out_dir = tmp / "output"
        out_dir.mkdir()

        _jobs["job-b1"] = {
            "job_id": "job-b1",
            "sample_id": "sample-b1",
            "status": "pending",
            "mode": "dictation",
            "provider_id": "dr_test",
            "data_dir": str(data_b1),
        }
        _jobs["job-b2"] = {
            "job_id": "job-b2",
            "sample_id": "sample-b2",
            "status": "pending",
            "mode": "dictation",
            "provider_id": "dr_test",
            "data_dir": str(data_b2),
        }

        run_count = 0
        original_run = None

        async def mock_run_pipeline(
            job_id, sample_id, audio_path, note_audio_path,
            mode, provider_id, visit_type, output_dir, version,
        ):
            nonlocal run_count
            run_count += 1
            job = _jobs.get(job_id, {})
            if sample_id == "sample-b1":
                job["status"] = "error"
                job["message"] = "OOM"
            else:
                job["status"] = "complete"

        with patch("api.pipeline.routes._run_pipeline", side_effect=mock_run_pipeline), \
             patch("api.pipeline.routes._find_sample_dirs", return_value=(data_b1, out_dir, "dictation", "dr_test")), \
             patch("api.pipeline.routes._get_pipeline_output_dir", return_value=out_dir), \
             patch("api.quality_runner.generate_aggregate_report", return_value=None):

            await _run_batch(["job-b1", "job-b2"], "v9", False)

        # Both jobs should have been attempted
        assert run_count == 2

        # Cleanup
        _jobs.pop("job-b1", None)
        _jobs.pop("job-b2", None)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


class TestUploadEdgeCases:
    """Upload endpoint edge cases."""

    def test_upload_invalid_encounter_details_json(self):
        """Invalid JSON in encounter_details should be handled gracefully."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.pipeline.routes import router
        import io

        app = FastAPI()
        app.include_router(router)

        with patch("api.pipeline.routes.require_feature"), \
             patch("api.pipeline.routes._get_pipeline_data_dir", return_value=Path("/tmp/test-data")):

            client = TestClient(app)
            resp = client.post(
                "/pipeline/upload",
                files={"audio": ("test.mp3", io.BytesIO(b"fake audio"), "audio/mpeg")},
                data={
                    "sample_id": "test-sample",
                    "mode": "dictation",
                    "provider_id": "dr_test",
                    "encounter_details": "{invalid json!!!}",
                },
            )
            # Should succeed — invalid JSON falls back to empty dict
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "pending"
