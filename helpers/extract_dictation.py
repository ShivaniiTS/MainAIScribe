#!/usr/bin/env python3
"""
Extract dictation files from OneDrive zip into data/dictation.

From the zip:
1. Navigate to "Transcription audio & files" (or "Transcription audio and files")
2. Match each audio file (.mp3) with its Word file (.doc/.docx) by identifier A-XXXXX
3. Create folder data/dictation/{id}/ with:
   - dictation.mp3 (renamed from original mp3)
   - soap_final.doc or soap_final.docx (renamed from original Word file)
   - soap_final.md (generated from Word file)

Example:
  Audio: D 0219-158 ... A-225981 @0219-1100.mp3
  Word:  U 0219-158 ... A-225981 @0219-1100 CHARLENE BRANDON....doc
  -> data/dictation/225981/dictation.mp3, soap_final.doc, soap_final.md
"""

import argparse
import re
import shutil
import subprocess
import zipfile
from pathlib import Path


# Folder names to look for (zip may use & or "and")
TRANSCRIPTION_FOLDER_NAMES = [
    "Transcription audio & files",
    "Transcription audio and files",
]

# Regex to extract A-XXXXX identifier
ID_PATTERN = re.compile(r"A-(\d+)")


def find_transcription_root(zip_path: Path, extract_dir: Path) -> Path | None:
    """Extract zip and return path to Transcription audio folder, or None."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Get all names and find the transcription folder
        names = zf.namelist()
        for folder_name in TRANSCRIPTION_FOLDER_NAMES:
            prefix = folder_name + "/"
            if any(n.startswith(prefix) for n in names):
                # Extract only that folder
                for name in names:
                    if name.startswith(prefix) and not name.endswith("/"):
                        zf.extract(name, extract_dir)
                return extract_dir / folder_name
    return None


def extract_id_from_filename(filename: str) -> str | None:
    """Extract identifier (e.g. 225981) from filename containing A-225981."""
    m = ID_PATTERN.search(filename)
    return m.group(1) if m else None


def convert_doc_to_md(doc_path: Path, md_path: Path) -> bool:
    """Convert .doc/.docx to markdown. Uses antiword for .doc, markitdown for .docx."""
    if doc_path.suffix.lower() == ".doc":
        # Use antiword for legacy .doc format
        try:
            result = subprocess.run(
                ["antiword", str(doc_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                md_path.write_text(result.stdout, encoding="utf-8")
                return True
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    else:
        # .docx: use markitdown
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(str(doc_path))
            md_path.write_text(result.text_content, encoding="utf-8")
            return True
        except Exception:
            return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract dictation files from zip into data/dictation"
    )
    parser.add_argument(
        "--zip",
        default=None,
        help="Path to zip file (default: tmp/OneDrive_2026-03-08.zip)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output base directory (default: data/dictation)",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Temporary extraction directory (default: tmp/dictation_extract)",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    zip_path = Path(args.zip) if args.zip else workspace / "tmp" / "OneDrive_2026-03-08.zip"
    output_base = Path(args.output) if args.output else workspace / "data" / "dictation"
    extract_dir = Path(args.extract_dir) if args.extract_dir else workspace / "tmp" / "dictation_extract"

    if not zip_path.exists():
        raise SystemExit(f"Zip file not found: {zip_path}")

    print(f"Extracting from {zip_path}...")
    transcription_root = find_transcription_root(zip_path, extract_dir)
    if not transcription_root or not transcription_root.exists():
        raise SystemExit(
            f"Could not find '{TRANSCRIPTION_FOLDER_NAMES[0]}' (or variant) in zip"
        )

    # Collect files by identifier
    audio_by_id: dict[str, Path] = {}
    word_by_id: dict[str, Path] = {}

    for f in transcription_root.iterdir():
        if not f.is_file():
            continue
        id_val = extract_id_from_filename(f.name)
        if not id_val:
            continue
        lower = f.suffix.lower()
        if lower == ".mp3":
            audio_by_id[id_val] = f
        elif lower in (".doc", ".docx"):
            word_by_id[id_val] = f

    # Process pairs
    output_base.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped_no_word = 0
    skipped_no_audio = 0

    for id_val in sorted(set(audio_by_id) | set(word_by_id)):
        audio_path = audio_by_id.get(id_val)
        word_path = word_by_id.get(id_val)

        if not audio_path:
            skipped_no_audio += 1
            continue
        if not word_path:
            skipped_no_word += 1
            continue

        folder = output_base / id_val
        folder.mkdir(parents=True, exist_ok=True)

        # Copy and rename
        shutil.copy2(audio_path, folder / "dictation.mp3")
        dest_word = folder / f"soap_final{word_path.suffix}"
        shutil.copy2(word_path, dest_word)

        # Generate soap_final.md
        md_path = folder / "soap_final.md"
        if convert_doc_to_md(dest_word, md_path):
            converted += 1
            print(f"  {id_val}: dictation.mp3, soap_final{word_path.suffix}, soap_final.md")
        else:
            print(f"  {id_val}: dictation.mp3, soap_final{word_path.suffix} (md conversion failed)")

    print(f"\nDone. Created {converted} folders in {output_base}")
    if skipped_no_word:
        print(f"  Skipped {skipped_no_word} (no matching Word file)")
    if skipped_no_audio:
        print(f"  Skipped {skipped_no_audio} (no matching audio)")


if __name__ == "__main__":
    main()
