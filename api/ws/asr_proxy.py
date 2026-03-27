"""
api/ws/asr_proxy.py — Provider-facing server proxy for streaming ASR endpoints.

Proxies /asr/preload, /asr/status, and /ws/asr/{encounter_id} to the
pipeline server which runs the GPU-based NeMo model.

This keeps the Capture page on the provider-facing server (port 8000)
while streaming ASR runs on the pipeline server (port 8100).
"""
from __future__ import annotations

import asyncio
import json
import logging

import httpx
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from config.deployment import get_deployment_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["asr"])


def _pipeline_url() -> str:
    cfg = get_deployment_config()
    return cfg.pipeline_api_url


def _get_client() -> httpx.AsyncClient:
    cfg = get_deployment_config()
    headers = {}
    if cfg.inter_server_auth.enabled:
        secret = cfg.inter_server_auth.secret
        if secret:
            headers["X-Inter-Server-Auth"] = secret
    return httpx.AsyncClient(
        base_url=_pipeline_url(),
        headers=headers,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )


@router.post("/asr/preload")
async def proxy_asr_preload(mode: str = "dictation"):
    """Proxy ASR model preload request to pipeline server."""
    async with _get_client() as client:
        try:
            resp = await client.post("/asr/preload", params={"mode": mode})
            return resp.json()
        except Exception as exc:
            logger.warning("asr_proxy: preload failed — %s", exc)
            return {"status": "unavailable", "message": f"Pipeline server unreachable: {exc}"}


@router.get("/asr/status")
async def proxy_asr_status():
    """Proxy ASR model status check to pipeline server."""
    async with _get_client() as client:
        try:
            resp = await client.get("/asr/status")
            return resp.json()
        except Exception as exc:
            logger.warning("asr_proxy: status failed — %s", exc)
            return {"status": "unavailable", "message": f"Pipeline server unreachable: {exc}"}


@router.websocket("/ws/asr/{encounter_id}")
async def proxy_asr_websocket(
    encounter_id: str,
    websocket: WebSocket,
    mode: str = Query("dictation"),
    format: str = Query("webm"),
):
    """
    Bidirectional WebSocket proxy: client ↔ provider server ↔ pipeline server.

    Audio chunks from the client are forwarded to the pipeline server's
    /ws/asr/ endpoint. Transcript events from the pipeline are forwarded
    back to the client.
    """
    await websocket.accept()

    pipeline_ws_url = _pipeline_url().replace("http://", "ws://").replace("https://", "wss://")
    pipeline_ws_url += f"/ws/asr/{encounter_id}?mode={mode}&format={format}"

    try:
        import websockets
        async with websockets.connect(
            pipeline_ws_url,
            max_size=10_000_000,
            ping_interval=None,
            close_timeout=10,
        ) as pipeline_ws:

            async def client_to_pipeline():
                """Forward audio chunks from client to pipeline."""
                try:
                    while True:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in message and message["bytes"]:
                            await pipeline_ws.send(message["bytes"])
                        elif "text" in message and message["text"]:
                            await pipeline_ws.send(message["text"])
                except WebSocketDisconnect:
                    pass
                except Exception:
                    pass

            async def pipeline_to_client():
                """Forward transcript events from pipeline to client."""
                try:
                    async for msg in pipeline_ws:
                        if isinstance(msg, str):
                            await websocket.send_text(msg)
                        else:
                            await websocket.send_bytes(msg)
                except Exception:
                    pass

            await asyncio.gather(
                client_to_pipeline(),
                pipeline_to_client(),
                return_exceptions=True,
            )

    except ImportError:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "websockets package not installed on provider server",
        }))
    except Exception as exc:
        logger.warning("asr_proxy: WebSocket proxy failed — %s", exc)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Pipeline server WebSocket unreachable: {exc}",
            }))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
