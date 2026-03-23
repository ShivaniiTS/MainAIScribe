"""
api/ws/session_events.py

WebSocket endpoint for real-time pipeline progress events.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    def __init__(self):
        # encounter_id → list of active WebSocket connections
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, encounter_id: str, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(encounter_id, []).append(ws)

    def disconnect(self, encounter_id: str, ws: WebSocket):
        conns = self._connections.get(encounter_id, [])
        if ws in conns:
            conns.remove(ws)

    async def send(self, encounter_id: str, data: dict[str, Any]):
        """Broadcast a progress event to all subscribers of an encounter."""
        message = json.dumps(data)
        dead = []
        for ws in self._connections.get(encounter_id, []):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(encounter_id, ws)

    async def send_progress(
        self,
        encounter_id: str,
        stage: str,
        pct: int,
        message: str = "",
    ):
        await self.send(
            encounter_id,
            {"type": "progress", "stage": stage, "pct": pct, "message": message},
        )

    async def send_complete(self, encounter_id: str, sample_id: str):
        await self.send(
            encounter_id,
            {"type": "complete", "sample_id": sample_id},
        )

    async def send_error(self, encounter_id: str, error: str):
        await self.send(
            encounter_id,
            {"type": "error", "error": error},
        )


manager = ConnectionManager()


@router.websocket("/ws/encounters/{encounter_id}")
async def encounter_ws(encounter_id: str, websocket: WebSocket):
    """
    WebSocket for real-time pipeline progress.
    Clients subscribe by connecting here; server broadcasts stage events.

    Event shape:
      {"type": "progress", "stage": "transcribe", "pct": 40, "message": "..."}
      {"type": "complete", "sample_id": "..."}
      {"type": "error", "error": "..."}
    """
    await manager.connect(encounter_id, websocket)
    try:
        # Send a welcome ping
        await websocket.send_text(
            json.dumps({"type": "connected", "encounter_id": encounter_id})
        )
        # Keep the connection alive, waiting for server-push events
        while True:
            try:
                # Accept any client ping (ignore content)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(encounter_id, websocket)
