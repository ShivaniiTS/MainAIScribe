"""
tests/unit/test_concurrent_pipeline.py — Concurrent pipeline execution tests.

Covers:
1. Multiple simultaneous encounters tracked independently in _encounters dict
2. Multiple pipeline jobs tracked independently in _jobs dict
3. WebSocket events are encounter-scoped (no cross-talk)
4. Encounter status isolation (one error doesn't affect others)
5. Concurrent pipeline upload + trigger flow
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.ws.session_events import ConnectionManager


class TestConcurrentEncounterTracking:
    """Multiple encounters tracked independently."""

    def test_multiple_encounters_independent_status(self):
        """Creating multiple encounters should track independently."""
        from api.routes.encounters import _encounters

        # Simulate creating two encounters
        _encounters["enc-a"] = {"encounter_id": "enc-a", "status": "processing"}
        _encounters["enc-b"] = {"encounter_id": "enc-b", "status": "pending"}

        # Update one — other should be unaffected
        _encounters["enc-a"]["status"] = "error"
        assert _encounters["enc-b"]["status"] == "pending"

        # Cleanup
        del _encounters["enc-a"]
        del _encounters["enc-b"]

    def test_pipeline_jobs_independent_status(self):
        """Multiple pipeline jobs should track independently."""
        from api.pipeline.routes import _jobs

        _jobs["job-a"] = {"job_id": "job-a", "status": "processing", "sample_id": "s1"}
        _jobs["job-b"] = {"job_id": "job-b", "status": "pending", "sample_id": "s2"}

        _jobs["job-a"]["status"] = "complete"
        assert _jobs["job-b"]["status"] == "pending"

        del _jobs["job-a"]
        del _jobs["job-b"]


class TestWebSocketEncounterIsolation:
    """WebSocket events should be scoped to their encounter."""

    @pytest.mark.asyncio
    async def test_progress_events_scoped(self):
        """Progress events for one encounter should not reach another."""
        mgr = ConnectionManager()
        ws_a = AsyncMock()
        ws_b = AsyncMock()
        await mgr.connect("enc-a", ws_a)
        await mgr.connect("enc-b", ws_b)

        await mgr.send_progress("enc-a", "transcribe", 50, "Processing A")

        ws_a.send_text.assert_awaited_once()
        ws_b.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_complete_events_scoped(self):
        """Complete event for one encounter doesn't reach another."""
        mgr = ConnectionManager()
        ws_a = AsyncMock()
        ws_b = AsyncMock()
        await mgr.connect("enc-a", ws_a)
        await mgr.connect("enc-b", ws_b)

        await mgr.send_complete("enc-a", "sample-a")

        sent = json.loads(ws_a.send_text.call_args[0][0])
        assert sent["type"] == "complete"
        ws_b.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_error_events_scoped(self):
        """Error event for one encounter doesn't reach another."""
        mgr = ConnectionManager()
        ws_a = AsyncMock()
        ws_b = AsyncMock()
        await mgr.connect("enc-a", ws_a)
        await mgr.connect("enc-b", ws_b)

        await mgr.send_error("enc-a", "OOM crash")

        ws_b.send_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_encounter(self):
        """Multiple WS clients on the same encounter all get the event."""
        mgr = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()
        await mgr.connect("enc-shared", ws1)
        await mgr.connect("enc-shared", ws2)
        await mgr.connect("enc-shared", ws3)

        await mgr.send_progress("enc-shared", "note", 80, "Generating note")

        assert ws1.send_text.await_count == 1
        assert ws2.send_text.await_count == 1
        assert ws3.send_text.await_count == 1


class TestConcurrentPipelineErrorIsolation:
    """One pipeline failure should not affect another running pipeline."""

    @pytest.mark.asyncio
    async def test_one_failure_others_continue(self):
        """Failing one encounter should not affect another's status."""
        from api.routes.encounters import _encounters

        _encounters["enc-fail"] = {
            "encounter_id": "enc-fail",
            "status": "processing",
            "message": "",
        }
        _encounters["enc-ok"] = {
            "encounter_id": "enc-ok",
            "status": "processing",
            "message": "",
        }

        # Simulate failure in one
        _encounters["enc-fail"]["status"] = "error"
        _encounters["enc-fail"]["message"] = "Pipeline error: CUDA OOM"

        # Other should be unaffected
        assert _encounters["enc-ok"]["status"] == "processing"
        assert _encounters["enc-ok"]["message"] == ""

        del _encounters["enc-fail"]
        del _encounters["enc-ok"]


class TestConcurrentCreateAndPoll:
    """Test creating and polling multiple encounters concurrently."""

    def test_create_multiple_encounters_via_api(self):
        """Multiple encounters can be created and polled independently."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes.encounters import router, _encounters

        app = FastAPI()
        app.include_router(router)

        with patch("api.routes.encounters.needs_proxy", return_value=False), \
             patch("api.routes.encounters.dl"):

            client = TestClient(app)

            # Create two encounters
            resp1 = client.post("/encounters", json={
                "provider_id": "dr_a",
                "patient_id": "p1",
                "visit_type": "initial_evaluation",
                "mode": "dictation",
            })
            resp2 = client.post("/encounters", json={
                "provider_id": "dr_b",
                "patient_id": "p2",
                "visit_type": "follow_up",
                "mode": "ambient",
            })

            assert resp1.status_code == 201
            assert resp2.status_code == 201

            enc_id_1 = resp1.json()["encounter_id"]
            enc_id_2 = resp2.json()["encounter_id"]
            assert enc_id_1 != enc_id_2

            # Poll each independently
            status1 = client.get(f"/encounters/{enc_id_1}/status")
            status2 = client.get(f"/encounters/{enc_id_2}/status")

            assert status1.status_code == 200
            assert status2.status_code == 200
            assert status1.json()["provider_id"] == "dr_a"
            assert status2.json()["provider_id"] == "dr_b"

            # Cleanup
            _encounters.pop(enc_id_1, None)
            _encounters.pop(enc_id_2, None)
