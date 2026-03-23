#!/usr/bin/env python3
# Uses only stdlib: zipfile, pathlib, shutil, argparse
"""
Unzip a zip file from the data folder into tmp, then flatten the hierarchy.

Example:
  Original: 16245/7274335/4/151150/ containing conversation.mp3, notes.mp3, soap_initial.docx
  Flattened: 16245_7274335_4_151150/ containing those same files
"""

import argparse
import shutil
import zipfile
from pathlib import Path


def unzip_to_tmp(zip_path: Path, tmp_dir: Path) -> Path:
    """Extract zip file to tmp directory. Returns path to extracted root."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)
    # Find the extracted root (first subdirectory if zip has a single top-level folder)
    extracted = tmp_dir
    items = list(tmp_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        extracted = items[0]
    return extracted


def flatten_hierarchy(source_root: Path, output_dir: Path) -> None:
    """
    Flatten nested folder hierarchy into single folders named by path parts joined with underscores.

    Leaf folders (those containing files) become folders like: 16245_7274335_4_151150
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Walk the tree and find leaf folders (folders that contain files, not just subdirs)
    for path in source_root.rglob("*"):
        if path.is_file():
            # Get the leaf folder containing this file
            leaf_folder = path.parent
            # Compute relative path from source_root to leaf_folder
            try:
                rel = leaf_folder.relative_to(source_root)
            except ValueError:
                continue
            # Flatten: 16245/7274335/4/151150 -> 16245_7274335_4_151150
            flat_name = "_".join(rel.parts)
            flat_dir = output_dir / flat_name
            flat_dir.mkdir(parents=True, exist_ok=True)
            dest = flat_dir / path.name
            shutil.copy2(path, dest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unzip data zip file and flatten folder hierarchy"
    )
    parser.add_argument(
        "--zip",
        default=None,
        help="Path to zip file (default: data/*.zip in workspace)",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Temporary directory for extracted files (default: tmp/unzipped)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for flattened files (default: tmp/flattened)",
    )
    parser.add_argument(
        "--flatten-only",
        action="store_true",
        help="Skip unzip; only flatten from existing tmp dir",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    data_dir = workspace / "data"

    if args.zip:
        zip_path = Path(args.zip)
    else:
        # Find first zip in data folder
        zips = list(data_dir.glob("*.zip"))
        if not zips:
            raise SystemExit(f"No zip file found in {data_dir}")
        zip_path = zips[0]

    if not zip_path.exists():
        raise SystemExit(f"Zip file not found: {zip_path}")

    tmp_base = Path(args.tmp_dir) if args.tmp_dir else workspace / "tmp"
    tmp_extracted = tmp_base / "unzipped"
    output_dir = Path(args.output) if args.output else tmp_base / "flattened"

    if not args.flatten_only:
        print(f"Unzipping {zip_path} to {tmp_extracted}...")
        source_root = unzip_to_tmp(zip_path, tmp_extracted)
        print(f"Extracted to {source_root}")
    else:
        # Find extracted root
        items = list(tmp_extracted.iterdir())
        if not items:
            raise SystemExit(f"No extracted content in {tmp_extracted}. Run without --flatten-only first.")
        source_root = items[0] if items[0].is_dir() else tmp_extracted

    print(f"Flattening hierarchy from {source_root} to {output_dir}...")
    flatten_hierarchy(source_root, output_dir)
    print(f"Done. Flattened files in {output_dir}")


if __name__ == "__main__":
    main()
