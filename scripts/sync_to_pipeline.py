#!/usr/bin/env python3
"""
scripts/sync_to_pipeline.py — Push audio + encounter data to the pipeline server.

Transfers audio files and de-identified encounter metadata from the
provider-facing server to the processing pipeline server for pipeline execution.

PHI isolation: patient_demographics.json, patient_context.yaml, and
final_soap_note.md are NEVER sent. Only audio files and encounter_details.json
(which contains mode, visit_type, provider_id — no patient names) are transferred.

Usage:
    # Sync all samples
    python scripts/sync_to_pipeline.py

    # Sync specific samples
    python scripts/sync_to_pipeline.py --sample-ids sample_001,sample_002

    # Dry run (show what would be synced)
    python scripts/sync_to_pipeline.py --dry-run

    # Sync only new/modified files
    python scripts/sync_to_pipeline.py --incremental
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from config.deployment import get_deployment_config
from config.paths import DATA_DIR


# Files safe to send to the pipeline server
SAFE_FILES = {
    "dictation.mp3",
    "conversation_audio.mp3",
    "note_audio.mp3",
    "notes.mp3",
    "encounter_details.json",
}

# Files that contain PHI — NEVER send
PHI_FILES = {
    "patient_demographics.json",
    "patient_context.yaml",
    "final_soap_note.md",
}


def discover_samples(data_dir: Path, sample_ids: list[str] | None = None) -> list[dict]:
    """Discover all samples in the data directory."""
    samples = []
    for mode_dir in ["dictation", "conversation"]:
        mode_path = data_dir / mode_dir
        if not mode_path.exists():
            continue
        for physician_dir in mode_path.iterdir():
            if not physician_dir.is_dir():
                continue
            for sample_dir in physician_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                sid = sample_dir.name
                if sample_ids and sid not in sample_ids:
                    continue

                # Collect transferable files
                files = []
                for f in sample_dir.iterdir():
                    if f.name in SAFE_FILES and f.is_file():
                        files.append(f)

                if files:
                    samples.append({
                        "sample_id": sid,
                        "mode": "ambient" if mode_dir == "conversation" else "dictation",
                        "provider_id": physician_dir.name,
                        "data_dir": sample_dir,
                        "files": files,
                    })

    return samples


def sync_sample(
    client: httpx.Client,
    sample: dict,
    dry_run: bool = False,
    incremental: bool = False,
) -> dict:
    """Upload a single sample to the pipeline server."""
    sid = sample["sample_id"]
    mode = sample["mode"]
    provider_id = sample["provider_id"]

    if dry_run:
        file_names = [f.name for f in sample["files"]]
        total_size = sum(f.stat().st_size for f in sample["files"])
        return {
            "sample_id": sid,
            "status": "dry_run",
            "files": file_names,
            "total_bytes": total_size,
        }

    # Find audio file
    audio_file = None
    for f in sample["files"]:
        if f.suffix == ".mp3":
            audio_file = f
            break

    if not audio_file:
        return {"sample_id": sid, "status": "skipped", "reason": "no audio file"}

    # Read encounter details
    encounter_details = {}
    details_file = sample["data_dir"] / "encounter_details.json"
    if details_file.exists():
        encounter_details = json.loads(details_file.read_text())

    # Upload via pipeline API
    try:
        with open(audio_file, "rb") as af:
            resp = client.post(
                "/pipeline/upload",
                files={"audio": (audio_file.name, af, "audio/mpeg")},
                data={
                    "sample_id": sid,
                    "mode": mode,
                    "provider_id": provider_id,
                    "encounter_details": json.dumps(encounter_details),
                },
                timeout=300.0,
            )
        resp.raise_for_status()
        result = resp.json()
        return {
            "sample_id": sid,
            "status": "uploaded",
            "job_id": result.get("job_id"),
        }
    except Exception as e:
        return {"sample_id": sid, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Sync data to pipeline server")
    parser.add_argument("--sample-ids", type=str, help="Comma-separated sample IDs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--incremental", action="store_true", help="Only sync new/modified files")
    parser.add_argument("--trigger", action="store_true", help="Trigger pipeline after upload")
    args = parser.parse_args()

    cfg = get_deployment_config()
    pipeline_url = cfg.pipeline_api_url

    sample_ids = None
    if args.sample_ids:
        sample_ids = [s.strip() for s in args.sample_ids.split(",")]

    print(f"Pipeline server: {pipeline_url}")
    print(f"Data directory: {DATA_DIR}")
    print()

    samples = discover_samples(DATA_DIR, sample_ids)
    print(f"Found {len(samples)} samples to sync")

    if not samples:
        print("Nothing to sync.")
        return

    headers = {}
    if cfg.inter_server_auth.enabled:
        secret = cfg.inter_server_auth.secret
        if secret:
            headers["X-Inter-Server-Auth"] = secret

    uploaded = 0
    errors = 0
    job_ids = []

    with httpx.Client(base_url=pipeline_url, headers=headers) as client:
        for i, sample in enumerate(samples, 1):
            result = sync_sample(client, sample, dry_run=args.dry_run, incremental=args.incremental)
            status = result["status"]
            sid = result["sample_id"]

            if status == "uploaded":
                uploaded += 1
                jid = result.get("job_id", "")
                job_ids.append(jid)
                print(f"  [{i}/{len(samples)}] {sid} → uploaded (job: {jid})")
            elif status == "dry_run":
                files = result.get("files", [])
                size_kb = result.get("total_bytes", 0) / 1024
                print(f"  [{i}/{len(samples)}] {sid} → would upload {len(files)} files ({size_kb:.0f} KB)")
            elif status == "skipped":
                print(f"  [{i}/{len(samples)}] {sid} → skipped ({result.get('reason', '')})")
            else:
                errors += 1
                print(f"  [{i}/{len(samples)}] {sid} → ERROR: {result.get('error', '')}")

    print()
    if args.dry_run:
        print(f"Dry run complete. {len(samples)} samples would be synced.")
    else:
        print(f"Sync complete. Uploaded: {uploaded}, Errors: {errors}")

    # Optionally trigger pipeline for all uploaded samples
    if args.trigger and job_ids and not args.dry_run:
        print()
        print("Triggering pipeline for uploaded samples...")
        with httpx.Client(base_url=pipeline_url, headers=headers) as client:
            for jid in job_ids:
                try:
                    resp = client.post(f"/pipeline/trigger/{jid}", json={})
                    resp.raise_for_status()
                    print(f"  Triggered: {jid}")
                except Exception as e:
                    print(f"  Error triggering {jid}: {e}")


if __name__ == "__main__":
    main()
