#!/usr/bin/env python3
"""
scripts/sync_from_pipeline.py — Pull generated outputs from the pipeline server.

Retrieves generated notes, transcripts, and comparisons from the processing
pipeline server to the provider-facing server's output directory.

Usage:
    # Sync all outputs
    python scripts/sync_from_pipeline.py

    # Sync specific samples
    python scripts/sync_from_pipeline.py --sample-ids sample_001,sample_002

    # Only sync files newer than a timestamp
    python scripts/sync_from_pipeline.py --since 2026-03-15T00:00:00

    # Dry run
    python scripts/sync_from_pipeline.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from config.deployment import get_deployment_config
from config.paths import DATA_DIR, OUTPUT_DIR


def discover_local_samples(output_dir: Path, data_dir: Path) -> list[str]:
    """Discover all known sample IDs from both output/ and data/ directories."""
    sample_ids = set()
    for root_dir in [output_dir, data_dir]:
        for mode_dir in ["dictation", "conversation"]:
            mode_path = root_dir / mode_dir
            if not mode_path.exists():
                continue
            for physician_dir in mode_path.iterdir():
                if not physician_dir.is_dir():
                    continue
                for sample_dir in physician_dir.iterdir():
                    if sample_dir.is_dir():
                        sample_ids.add(sample_dir.name)
    return sorted(sample_ids)


def find_local_output_dir(sample_id: str) -> Path | None:
    """Find the local output directory for a sample."""
    for mode_dir in ["dictation", "conversation"]:
        mode_path = OUTPUT_DIR / mode_dir
        if not mode_path.exists():
            continue
        for physician_dir in mode_path.iterdir():
            if not physician_dir.is_dir():
                continue
            sample_dir = physician_dir / sample_id
            if sample_dir.exists():
                return sample_dir
    return None


def find_data_mode_physician(sample_id: str) -> tuple[str, str] | None:
    """Find mode and physician for a sample from data directory."""
    for mode_dir in ["dictation", "conversation"]:
        mode_path = DATA_DIR / mode_dir
        if not mode_path.exists():
            continue
        for physician_dir in mode_path.iterdir():
            if not physician_dir.is_dir():
                continue
            if (physician_dir / sample_id).exists():
                return mode_dir, physician_dir.name
    return None


def sync_outputs(
    client: httpx.Client,
    sample_ids: list[str],
    since: str = "",
    dry_run: bool = False,
    overwrite_policy: str = "newer_only",
) -> dict:
    """Sync output files from the pipeline server."""
    # Get manifest of available outputs
    try:
        resp = client.get(
            "/pipeline/outputs/batch",
            params={"sample_ids": ",".join(sample_ids), "since": since},
            timeout=60.0,
        )
        resp.raise_for_status()
        manifest = resp.json()
    except Exception as e:
        return {"error": str(e), "synced": 0}

    samples = manifest.get("samples", [])
    synced = 0
    skipped = 0

    for sample in samples:
        sid = sample["sample_id"]
        files = sample.get("files", [])

        # Find or create local output directory
        local_dir = find_local_output_dir(sid)
        if local_dir is None:
            # Try to determine from data dir
            info = find_data_mode_physician(sid)
            if info:
                mode_dir, physician = info
                local_dir = OUTPUT_DIR / mode_dir / physician / sid
                local_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"  WARNING: Cannot find local path for {sid}, skipping")
                continue

        for file_info in files:
            fname = file_info["filename"]
            remote_modified = file_info.get("modified", "")

            local_path = local_dir / fname

            # Check overwrite policy
            if local_path.exists() and overwrite_policy == "newer_only":
                local_mtime = datetime.fromtimestamp(local_path.stat().st_mtime)
                if remote_modified:
                    try:
                        remote_dt = datetime.fromisoformat(remote_modified)
                        if remote_dt <= local_mtime:
                            skipped += 1
                            continue
                    except ValueError:
                        pass

            if dry_run:
                print(f"  Would sync: {sid}/{fname} ({file_info.get('size', 0)} bytes)")
                synced += 1
                continue

            # Fetch the actual content
            try:
                if fname.startswith("generated_note_"):
                    version = fname.replace("generated_note_", "").replace(".md", "")
                    resp = client.get(
                        f"/pipeline/output/{sid}/note",
                        params={"version": version},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    content = resp.json().get("content", "")
                    local_path.write_text(content)
                    synced += 1
                elif fname.startswith("audio_transcript_"):
                    version = fname.replace("audio_transcript_", "").replace(".txt", "")
                    resp = client.get(
                        f"/pipeline/output/{sid}/transcript",
                        params={"version": version},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    content = resp.json().get("content", "")
                    local_path.write_text(content)
                    synced += 1
            except Exception as e:
                print(f"  ERROR syncing {sid}/{fname}: {e}")

    return {"synced": synced, "skipped": skipped, "samples": len(samples)}


def main():
    parser = argparse.ArgumentParser(description="Sync outputs from pipeline server")
    parser.add_argument("--sample-ids", type=str, help="Comma-separated sample IDs")
    parser.add_argument("--since", type=str, default="", help="ISO timestamp filter")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    args = parser.parse_args()

    cfg = get_deployment_config()
    pipeline_url = cfg.pipeline_api_url
    overwrite_policy = cfg.sync.output_sync.overwrite_policy

    print(f"Pipeline server: {pipeline_url}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Discover samples
    if args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",")]
    else:
        sample_ids = discover_local_samples(OUTPUT_DIR, DATA_DIR)

    print(f"Checking {len(sample_ids)} samples for new outputs...")

    headers = {}
    if cfg.inter_server_auth.enabled:
        secret = cfg.inter_server_auth.secret
        if secret:
            headers["X-Inter-Server-Auth"] = secret

    with httpx.Client(base_url=pipeline_url, headers=headers) as client:
        result = sync_outputs(
            client, sample_ids,
            since=args.since,
            dry_run=args.dry_run,
            overwrite_policy=overwrite_policy,
        )

    print()
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Sync complete. Files synced: {result['synced']}, "
              f"Skipped (unchanged): {result.get('skipped', 0)}, "
              f"Samples with outputs: {result.get('samples', 0)}")


if __name__ == "__main__":
    main()
