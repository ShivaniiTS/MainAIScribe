#!/usr/bin/env python3
"""
Backfill standalone transcript files for pipeline versions that predate
the audio_transcript_v{N}.txt convention.

For each sample that has a transcript in the cache, writes:
  output/{dictation,conversations}/{id}/audio_transcript_v{N}.txt

Usage:
    python scripts/backfill_transcripts.py
    python scripts/backfill_transcripts.py --versions v1 v2 v3 v4 v5
    python scripts/backfill_transcripts.py --cache output/transcript_cache_v7q14.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import OUTPUT_DIR
DEFAULT_CACHE = str(OUTPUT_DIR / "transcript_cache_v7q14.json")
DEFAULT_VERSIONS = ["v1", "v2", "v3", "v4", "v5"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="Path to transcript cache JSON (from two-pass batch_eval)")
    parser.add_argument("--versions", nargs="+", default=DEFAULT_VERSIONS,
                        help="Versions to backfill")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        sys.exit(1)

    with open(cache_path) as f:
        cache: dict[str, dict] = json.load(f)

    written = 0
    skipped = 0

    for sample_id, data in cache.items():
        full_text = data.get("full_text", "").strip()
        if not full_text:
            print(f"  {sample_id}: no transcript text, skipping")
            skipped += 1
            continue

        # Locate the output directory for this sample
        sample_out_dir = None
        for subdir in ("dictation", "conversations"):
            candidate = OUTPUT_DIR / subdir / sample_id
            if candidate.exists():
                sample_out_dir = candidate
                break

        if sample_out_dir is None:
            print(f"  {sample_id}: no output directory found, skipping")
            skipped += 1
            continue

        for version in args.versions:
            dest = sample_out_dir / f"audio_transcript_{version}.txt"
            dest.write_text(full_text)
            written += 1

        print(f"  {sample_id}: wrote {len(args.versions)} transcript files")

    print(f"\nDone — {written} files written, {skipped} samples skipped")


if __name__ == "__main__":
    main()
