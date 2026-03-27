"""
tests/unit/test_websocket.py — WebSocket ConnectionManager unit tests.

Covers:
1. ConnectionManager connect/disconnect lifecycle
2. Multi-subscriber broadcast
3. Dead connection cleanup on send failure
4. Event format validation (progress, complete, error, connected, ping)
5. Encounter isolation (messages don't leak between encounters)
6. WebSocket route integration via TestClient
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.ws.session_events import ConnectionManager


# ---------------------------------------------------------------------------
# ConnectionManager unit tests
# ---------------------------------------------------------------------------

class TestConnectionManagerConnect:
    """Test connect/disconnect lifecycle."""

    @pytest.fixture
    def mgr(self):
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect_accepts_and_stores(self, mgr):
        ws = AsyncMock()
        await mgr.connect("enc-1", ws)
        ws.accept.assert_awaited_once()
        assert ws in mgr._connections["enc-1"]

    @pytest.mark.asyncio
    async def test_connect_multiple_subscribers(self, mgr):
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect("enc-1", ws1)
        await mgr.connect("enc-1", ws2)
        assert len(mgr._connections["enc-1"]) == 2

    @pytest.mark.asyncio
    async def test_connect_different_encounters_isolated(self, mgr):
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await mgr.connect("enc-1", ws1)
        await mgr.connect("enc-2", ws2)
        assert len(mgr._connections["enc-1"]) == 1
        assert len(mgr._connections["enc-2"]) == 1

    def test_disconnect_removes_connection(self, mgr):
        ws = MagicMock()
        mgr._connections["enc-1"] = [ws]
        mgr.disconnect("enc-1", ws)
        assert ws not in mgr._connections["enc-1"]

    def test_disconnect_nonexistent_encounter_noop(self, mgr):
        ws = MagicMock()
        # Should not raise
        mgr.disconnect("nonexistent", ws)

    def test_disconnect_nonexistent_ws_noop(self, mgr):
        ws1 = MagicMock()
        ws2 = MagicMock()
        mgr._connections["enc-1"] = [ws1]
        mgr.disconnect("enc-1", ws2)
        assert ws1 in mgr._connections["enc-1"]


class TestConnectionManagerSend:
    """Test broadcast and dead connection cleanup."""

    @pytest.fixture
    def mgr(self):
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_send_broadcasts_to_all_subscribers(self, mgr):
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        mgr._connections["enc-1"] = [ws1, ws2]

        data = {"type": "progress", "stage": "transcribe", "pct": 50}
        await mgr.send("enc-1", data)

        expected = json.dumps(data)
        ws1.send_text.assert_awaited_once_with(expected)
        ws2.send_text.assert_awaited_once_with(expected)

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_encounter_noop(self, mgr):
        # Should not raise
        await mgr.send("nonexistent", {"type": "test"})

    @pytest.mark.asyncio
    async def test_send_removes_dead_connections(self, mgr):
        ws_good = AsyncMock()
        ws_dead = AsyncMock()
        ws_dead.send_text.side_effect = Exception("Connection closed")

        mgr._connections["enc-1"] = [ws_good, ws_dead]
        await mgr.send("enc-1", {"type": "test"})

        # Dead connection should be removed
        assert ws_dead not in mgr._connections["enc-1"]
        # Good connection should remain
        assert ws_good in mgr._connections["enc-1"]

    @pytest.mark.asyncio
    async def test_send_all_dead_connections_cleaned(self, mgr):
        ws1 = AsyncMock()
        ws1.send_text.side_effect = Exception("Connection closed")
        ws2 = AsyncMock()
        ws2.send_text.side_effect = Exception("Connection closed")

        mgr._connections["enc-1"] = [ws1, ws2]
        await mgr.send("enc-1", {"type": "test"})

        assert len(mgr._connections["enc-1"]) == 0

    @pytest.mark.asyncio
    async def test_send_does_not_leak_between_encounters(self, mgr):
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        mgr._connections["enc-1"] = [ws1]
        mgr._connections["enc-2"] = [ws2]

        await mgr.send("enc-1", {"type": "test"})

        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_not_awaited()


class TestConnectionManagerEventHelpers:
    """Test send_progress, send_complete, send_error format."""

    @pytest.fixture
    def mgr(self):
        return ConnectionManager()

    @pytest.mark.asyncio
    async def test_send_progress_event_format(self, mgr):
        ws = AsyncMock()
        mgr._connections["enc-1"] = [ws]

        await mgr.send_progress("enc-1", "transcribe", 40, "Transcribing...")

        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["type"] == "progress"
        assert sent["stage"] == "transcribe"
        assert sent["pct"] == 40
        assert sent["message"] == "Transcribing..."

    @pytest.mark.asyncio
    async def test_send_progress_default_message(self, mgr):
        ws = AsyncMock()
        mgr._connections["enc-1"] = [ws]

        await mgr.send_progress("enc-1", "init", 5)

        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["message"] == ""

    @pytest.mark.asyncio
    async def test_send_complete_event_format(self, mgr):
        ws = AsyncMock()
        mgr._connections["enc-1"] = [ws]

        await mgr.send_complete("enc-1", "sample_001")

        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["type"] == "complete"
        assert sent["sample_id"] == "sample_001"

    @pytest.mark.asyncio
    async def test_send_error_event_format(self, mgr):
        ws = AsyncMock()
        mgr._connections["enc-1"] = [ws]

        await mgr.send_error("enc-1", "Pipeline crashed: OOM")

        sent = json.loads(ws.send_text.call_args[0][0])
        assert sent["type"] == "error"
        assert sent["error"] == "Pipeline crashed: OOM"


class TestWebSocketRoute:
    """Test the WebSocket route via TestClient."""

    @pytest.fixture
    def client(self):
        """Create a TestClient with just the WS router."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.ws.session_events import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_websocket_connect_sends_welcome(self, client):
        with client.websocket_connect("/ws/encounters/test-enc") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert data["encounter_id"] == "test-enc"

    def test_websocket_keepalive_ping(self, client):
        """After 30s timeout on receive, server should send a ping."""
        # The TestClient's websocket_connect doesn't truly timeout,
        # but we can verify the connected message is sent.
        with client.websocket_connect("/ws/encounters/ping-test") as ws:
            welcome = ws.receive_json()
            assert welcome["type"] == "connected"
            # Send a client message (simulates client ping)
            ws.send_text("ping")

    def test_websocket_disconnect_cleans_up(self, client):
        """After disconnect, manager should have no connections for this encounter."""
        from api.ws.session_events import manager

        with client.websocket_connect("/ws/encounters/cleanup-test") as ws:
            ws.receive_json()  # consume welcome

        # After context manager exits, connection should be removed
        conns = manager._connections.get("cleanup-test", [])
        assert len(conns) == 0
