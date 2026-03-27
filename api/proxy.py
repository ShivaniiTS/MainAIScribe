"""
api/proxy.py — Provider-facing server proxy to the processing pipeline.

When the provider-facing server receives pipeline trigger requests
(create encounter, upload audio, rerun), it proxies them to the
processing pipeline server's API. This keeps the pipeline execution
on the GPU server while the provider-facing server handles the
client-facing interface.

In processing-pipeline mode, no proxying is needed — the request runs locally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import structlog

from config.deployment import get_deployment_config, ServerRole

logger = structlog.get_logger()

# Persistent async client (reused across requests)
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        cfg = get_deployment_config()
        headers = {}
        if cfg.inter_server_auth.enabled:
            secret = cfg.inter_server_auth.secret
            if secret:
                headers["X-Inter-Server-Auth"] = secret
        _client = httpx.AsyncClient(
            base_url=cfg.pipeline_api_url,
            headers=headers,
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
    return _client


def needs_proxy() -> bool:
    """Return True if this server should proxy to a remote pipeline server."""
    cfg = get_deployment_config()
    return cfg.role == ServerRole.PROVIDER_FACING


async def proxy_upload(
    audio_bytes: bytes,
    audio_filename: str,
    note_audio_bytes: bytes | None,
    note_audio_filename: str | None,
    sample_id: str,
    mode: str,
    provider_id: str,
    encounter_details: dict[str, Any],
) -> dict:
    """Upload encounter files to the pipeline server."""
    client = _get_client()
    target_url = f"{client._base_url}/pipeline/upload"
    logger.info("proxy_upload_start", target=target_url, sample_id=sample_id, provider_id=provider_id, audio_size=len(audio_bytes))
    files: dict[str, tuple] = {
        "audio": (audio_filename, audio_bytes, "audio/mpeg"),
    }
    if note_audio_bytes and note_audio_filename:
        files["note_audio"] = (note_audio_filename, note_audio_bytes, "audio/mpeg")
    resp = await client.post(
        "/pipeline/upload",
        files=files,
        data={
            "sample_id": sample_id,
            "mode": mode,
            "provider_id": provider_id,
            "encounter_details": json.dumps(encounter_details),
        },
    )
    if resp.status_code >= 400:
        body = resp.text
        raise RuntimeError(f"Pipeline upload failed ({resp.status_code}): {body}")
    return resp.json()


async def proxy_trigger(job_id: str, mode: str, provider_id: str, visit_type: str) -> dict:
    """Trigger pipeline execution on the pipeline server."""
    client = _get_client()
    resp = await client.post(
        f"/pipeline/trigger/{job_id}",
        json={
            "mode": mode,
            "provider_id": provider_id,
            "visit_type": visit_type,
        },
    )
    resp.raise_for_status()
    return resp.json()


async def proxy_status(job_id: str) -> dict:
    """Poll pipeline status from the pipeline server.

    Retries up to 3 times on transient connection errors (the pipeline
    server may drop idle connections between polls).
    """
    import asyncio as _asyncio

    client = _get_client()
    last_exc: Exception | None = None

    for attempt in range(3):
        try:
            resp = await client.get(f"/pipeline/status/{job_id}")
            resp.raise_for_status()
            return resp.json()
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as exc:
            last_exc = exc
            logger.warning(
                "proxy_status_retry",
                job_id=job_id,
                attempt=attempt + 1,
                error=f"{type(exc).__name__}: {exc}",
            )
            await _asyncio.sleep(2)

    raise last_exc or RuntimeError(f"proxy_status failed for {job_id}")


async def proxy_get_note(sample_id: str, version: str = "latest") -> dict:
    """Fetch generated note from the pipeline server."""
    client = _get_client()
    resp = await client.get(
        f"/pipeline/output/{sample_id}/note",
        params={"version": version},
    )
    resp.raise_for_status()
    return resp.json()


async def proxy_get_transcript(sample_id: str, version: str = "latest") -> dict:
    """Fetch transcript from the pipeline server."""
    client = _get_client()
    resp = await client.get(
        f"/pipeline/output/{sample_id}/transcript",
        params={"version": version},
    )
    resp.raise_for_status()
    return resp.json()


async def proxy_batch_outputs(sample_ids: list[str], since: str = "") -> dict:
    """Batch retrieve output manifests from the pipeline server."""
    client = _get_client()
    resp = await client.get(
        "/pipeline/outputs/batch",
        params={"sample_ids": ",".join(sample_ids), "since": since},
    )
    resp.raise_for_status()
    return resp.json()


async def close() -> None:
    """Close the persistent HTTP client."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
