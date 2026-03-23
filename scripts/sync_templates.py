#!/usr/bin/env python3
"""
scripts/sync_templates.py — Periodic config sync from pipeline server.

Pulls templates, providers, specialties from the authoritative pipeline
server to keep the provider-facing server's config in sync.

This is the CLI equivalent of the background sync task in api/sync.py.
Useful for:
  - Initial setup of a new provider-facing server
  - Manual sync trigger
  - Cron job alternative to the in-process background task

Usage:
    python scripts/sync_templates.py
    python scripts/sync_templates.py --items providers,templates
    python scripts/sync_templates.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import yaml

from config.deployment import get_deployment_config
from config.paths import CONFIG_DIR, PROVIDERS_DIR


def sync_providers(client: httpx.Client, dry_run: bool = False) -> int:
    """Sync provider profiles from pipeline server."""
    resp = client.get("/providers")
    resp.raise_for_status()
    providers = resp.json()

    PROVIDERS_DIR.mkdir(parents=True, exist_ok=True)
    synced = 0

    for p in providers:
        pid = p.get("id", "")
        if not pid:
            continue

        local_path = PROVIDERS_DIR / f"{pid}.yaml"
        if local_path.exists():
            # Provider already exists locally — skip (admin manages on pipeline server)
            continue

        if dry_run:
            print(f"  Would create provider: {pid}")
            synced += 1
            continue

        profile = {
            "id": pid,
            "name": p.get("name", ""),
            "credentials": p.get("credentials", ""),
            "specialty": p.get("specialty", ""),
        }
        local_path.write_text(yaml.dump(profile, default_flow_style=False))
        print(f"  Created provider: {pid}")
        synced += 1

    return synced


def sync_templates(client: httpx.Client, dry_run: bool = False) -> int:
    """Sync note templates from pipeline server."""
    resp = client.get("/templates")
    resp.raise_for_status()
    templates = resp.json()

    template_dir = CONFIG_DIR / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    synced = 0

    for t in templates:
        tid = t.get("id", "")
        if not tid:
            continue

        local_path = template_dir / f"{tid}.yaml"
        if local_path.exists():
            continue

        # Fetch full template detail
        try:
            detail_resp = client.get(f"/templates/{tid}")
            detail_resp.raise_for_status()
            detail = detail_resp.json()
        except Exception as e:
            print(f"  ERROR fetching template {tid}: {e}")
            continue

        if dry_run:
            print(f"  Would create template: {tid}")
            synced += 1
            continue

        local_path.write_text(yaml.dump(detail, default_flow_style=False))
        print(f"  Created template: {tid}")
        synced += 1

    return synced


def sync_specialties(client: httpx.Client, dry_run: bool = False) -> int:
    """Sync specialty dictionaries from pipeline server."""
    resp = client.get("/specialties")
    resp.raise_for_status()
    specialties = resp.json()

    dict_dir = CONFIG_DIR / "dictionaries"
    dict_dir.mkdir(parents=True, exist_ok=True)
    synced = 0

    for s in specialties:
        sid = s.get("id", "")
        if not sid:
            continue

        local_path = dict_dir / f"{sid}.txt"
        if local_path.exists():
            continue

        # Fetch full specialty detail with terms
        try:
            detail_resp = client.get(f"/specialties/{sid}")
            detail_resp.raise_for_status()
            detail = detail_resp.json()
            terms = detail.get("terms", [])
        except Exception as e:
            print(f"  ERROR fetching specialty {sid}: {e}")
            continue

        if dry_run:
            print(f"  Would create specialty: {sid} ({len(terms)} terms)")
            synced += 1
            continue

        local_path.write_text("\n".join(terms) + "\n")
        print(f"  Created specialty: {sid} ({len(terms)} terms)")
        synced += 1

    return synced


def main():
    parser = argparse.ArgumentParser(description="Sync config from pipeline server")
    parser.add_argument("--items", type=str, default="providers,templates,dictionaries",
                       help="Comma-separated items to sync")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    args = parser.parse_args()

    cfg = get_deployment_config()
    pipeline_url = cfg.pipeline_api_url
    items = [i.strip() for i in args.items.split(",")]

    print(f"Pipeline server: {pipeline_url}")
    print(f"Config directory: {CONFIG_DIR}")
    print(f"Items to sync: {items}")
    print()

    headers = {}
    if cfg.inter_server_auth.enabled:
        secret = cfg.inter_server_auth.secret
        if secret:
            headers["X-Inter-Server-Auth"] = secret

    total = 0

    with httpx.Client(base_url=pipeline_url, headers=headers, timeout=60.0) as client:
        if "providers" in items:
            print("Syncing providers...")
            count = sync_providers(client, dry_run=args.dry_run)
            total += count
            print(f"  → {count} providers {'would be ' if args.dry_run else ''}synced")
            print()

        if "templates" in items:
            print("Syncing templates...")
            count = sync_templates(client, dry_run=args.dry_run)
            total += count
            print(f"  → {count} templates {'would be ' if args.dry_run else ''}synced")
            print()

        if "dictionaries" in items:
            print("Syncing specialties...")
            count = sync_specialties(client, dry_run=args.dry_run)
            total += count
            print(f"  → {count} specialties {'would be ' if args.dry_run else ''}synced")
            print()

    print(f"Total items synced: {total}")


if __name__ == "__main__":
    main()
