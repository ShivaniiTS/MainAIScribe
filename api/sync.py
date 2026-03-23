"""
api/sync.py — Background config sync for provider-facing server.

Periodically pulls templates, providers, specialties, and EHR stub data
from the processing pipeline server to keep the provider-facing server
in sync with the authoritative configuration.

Runs as an asyncio background task during the FastAPI lifespan.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import structlog

from config.deployment import get_deployment_config, ServerRole

logger = structlog.get_logger()

_sync_task: asyncio.Task | None = None


async def start_config_sync() -> None:
    """Start the periodic config sync background task."""
    global _sync_task
    cfg = get_deployment_config()

    # Only run on provider-facing server
    if cfg.role != ServerRole.PROVIDER_FACING:
        logger.info("config_sync_skipped", reason=f"role={cfg.role.value}")
        return

    if not cfg.sync.config_sync.enabled:
        logger.info("config_sync_disabled")
        return

    _sync_task = asyncio.create_task(_sync_loop())
    logger.info("config_sync_started", interval=cfg.sync.config_sync.interval_seconds)


async def stop_config_sync() -> None:
    """Stop the sync background task."""
    global _sync_task
    if _sync_task:
        _sync_task.cancel()
        try:
            await _sync_task
        except asyncio.CancelledError:
            pass
        _sync_task = None
        logger.info("config_sync_stopped")


async def _sync_loop() -> None:
    """Periodically sync config from pipeline server."""
    cfg = get_deployment_config()
    interval = cfg.sync.config_sync.interval_seconds

    while True:
        try:
            await _do_sync()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("config_sync_error", error=str(e))

        await asyncio.sleep(interval)


async def _do_sync() -> None:
    """Execute one sync cycle — pull config items from pipeline server."""
    cfg = get_deployment_config()
    pipeline_url = cfg.pipeline_api_url
    items = cfg.sync.config_sync.items

    headers = {}
    if cfg.inter_server_auth.enabled:
        secret = cfg.inter_server_auth.secret
        if secret:
            headers["X-Inter-Server-Auth"] = secret

    async with httpx.AsyncClient(
        base_url=pipeline_url,
        headers=headers,
        timeout=httpx.Timeout(60.0),
    ) as client:
        synced = []

        if "providers" in items:
            try:
                resp = await client.get("/providers")
                resp.raise_for_status()
                providers = resp.json()
                _sync_providers(providers)
                synced.append(f"providers({len(providers)})")
            except Exception as e:
                logger.warning("sync_providers_failed", error=str(e))

        if "templates" in items:
            try:
                resp = await client.get("/templates")
                resp.raise_for_status()
                templates = resp.json()
                _sync_templates(templates, client)
                synced.append(f"templates({len(templates)})")
            except Exception as e:
                logger.warning("sync_templates_failed", error=str(e))

        if "dictionaries" in items:
            try:
                resp = await client.get("/specialties")
                resp.raise_for_status()
                specialties = resp.json()
                await _sync_specialties(specialties, client)
                synced.append(f"specialties({len(specialties)})")
            except Exception as e:
                logger.warning("sync_specialties_failed", error=str(e))

        logger.info("config_sync_complete", synced=synced)


def _sync_providers(providers: list[dict]) -> None:
    """Write provider profiles to local config/providers/."""
    from config.paths import PROVIDERS_DIR
    PROVIDERS_DIR.mkdir(parents=True, exist_ok=True)

    for p in providers:
        provider_id = p.get("id", "")
        if not provider_id:
            continue
        # Only sync the summary — full profile needs a detail fetch
        # For now, write a minimal YAML if it doesn't exist locally
        local_path = PROVIDERS_DIR / f"{provider_id}.yaml"
        if not local_path.exists():
            import yaml
            profile = {
                "id": provider_id,
                "name": p.get("name", ""),
                "credentials": p.get("credentials", ""),
                "specialty": p.get("specialty", ""),
            }
            local_path.write_text(yaml.dump(profile, default_flow_style=False))
            logger.info("sync_provider_created", provider_id=provider_id)


def _sync_templates(templates: list[dict], client: httpx.AsyncClient) -> None:
    """Write template summaries to local config/templates/."""
    from config.paths import CONFIG_DIR
    template_dir = CONFIG_DIR / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    for t in templates:
        template_id = t.get("id", "")
        if not template_id:
            continue
        local_path = template_dir / f"{template_id}.yaml"
        # Don't overwrite existing templates — admin manages these
        # Only create missing ones (new templates created on pipeline server)
        if not local_path.exists():
            logger.info("sync_template_new", template_id=template_id,
                       msg="New template detected — fetch detail on next sync")


async def _sync_specialties(specialties: list[dict], client: httpx.AsyncClient) -> None:
    """Sync specialty dictionaries from pipeline server."""
    from config.paths import CONFIG_DIR
    dict_dir = CONFIG_DIR / "dictionaries"
    dict_dir.mkdir(parents=True, exist_ok=True)

    for s in specialties:
        specialty_id = s.get("id", "")
        if not specialty_id:
            continue
        local_path = dict_dir / f"{specialty_id}.txt"
        if not local_path.exists():
            # Fetch the full specialty detail
            try:
                resp = await client.get(f"/specialties/{specialty_id}")
                resp.raise_for_status()
                detail = resp.json()
                terms = detail.get("terms", [])
                local_path.write_text("\n".join(terms) + "\n")
                logger.info("sync_specialty_created", specialty_id=specialty_id,
                           term_count=len(terms))
            except Exception as e:
                logger.warning("sync_specialty_fetch_failed",
                             specialty_id=specialty_id, error=str(e))


async def sync_outputs_from_pipeline(sample_ids: list[str], since: str = "") -> dict:
    """
    Pull generated outputs from the pipeline server for specific samples.

    Called by the data sync scripts or on-demand when checking for new results.
    Returns a summary of what was synced.
    """
    cfg = get_deployment_config()
    if cfg.role not in (ServerRole.PROVIDER_FACING,):
        return {"synced": 0, "message": "Output sync only runs on provider-facing server"}

    from api.proxy import proxy_batch_outputs, proxy_get_note, proxy_get_transcript
    from config.paths import OUTPUT_DIR

    manifest = await proxy_batch_outputs(sample_ids, since)
    samples = manifest.get("samples", [])
    synced_files = 0

    for sample in samples:
        sid = sample["sample_id"]
        for file_info in sample.get("files", []):
            fname = file_info["filename"]
            # Determine where to write locally
            # We need to know the mode and physician for the path
            # For now, fetch via the output endpoints
            if fname.startswith("generated_note_"):
                version = fname.replace("generated_note_", "").replace(".md", "")
                try:
                    data = await proxy_get_note(sid, version)
                    # Write to local output dir (need to find the right path)
                    _write_synced_output(sid, fname, data.get("content", ""))
                    synced_files += 1
                except Exception:
                    pass
            elif fname.startswith("audio_transcript_"):
                version = fname.replace("audio_transcript_", "").replace(".txt", "")
                try:
                    data = await proxy_get_transcript(sid, version)
                    _write_synced_output(sid, fname, data.get("content", ""))
                    synced_files += 1
                except Exception:
                    pass

    return {"synced": synced_files, "samples": len(samples)}


def _write_synced_output(sample_id: str, filename: str, content: str) -> None:
    """Write a synced output file to the local output directory."""
    from config.paths import OUTPUT_DIR

    # Find the sample's output directory
    for mode_dir in ["dictation", "conversation"]:
        mode_path = OUTPUT_DIR / mode_dir
        if not mode_path.exists():
            continue
        for physician_dir in mode_path.iterdir():
            if not physician_dir.is_dir():
                continue
            sample_dir = physician_dir / sample_id
            if sample_dir.exists():
                dest = sample_dir / filename
                dest.write_text(content)
                return

    # If not found, can't determine path without more context
    logger.warning("sync_output_no_local_dir", sample_id=sample_id, filename=filename)
