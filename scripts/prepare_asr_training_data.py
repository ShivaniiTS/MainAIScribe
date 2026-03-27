"""
scripts/prepare_asr_training_data.py

Prepares provider-specific ASR training data for Whisper LoRA fine-tuning.

Strategy:
  1. For each audio sample, run WhisperX forced alignment to produce word-level
     timestamps anchored to the gold note text.
  2. Segment the aligned audio into 10–30 second chunks.
  3. Apply data augmentation (speed perturbation, additive noise).
  4. Save a HuggingFace `datasets`-compatible DatasetDict for use with
     `finetune_whisper_lora.py`.

The gold note (soap_final.md / soap_initial.md) contains the authoritative
clinical text for each encounter.  WhisperX forced alignment pins each word
in the gold note to a timestamp in the audio, giving us accurate audio/text
pairs even when the base Whisper transcript contains medical term errors.

Output:
    data/asr_training/{provider_id}/
        train/   — ~90% of chunks (Arrow format)
        eval/    — ~10% of chunks (held-out for WER evaluation)
        manifest.json — metadata: chunk count, total hours, samples used

Usage:
    python scripts/prepare_asr_training_data.py --provider dr_faraz_rahman
    python scripts/prepare_asr_training_data.py --provider dr_faraz_rahman --augment
    python scripts/prepare_asr_training_data.py --provider dr_faraz_rahman --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.paths import DATA_DIR
OUTPUT_TRAINING_DIR = ROOT / "data" / "asr_training"

# Filtering thresholds
MIN_CHUNK_DURATION_S = 3.0
MAX_CHUNK_DURATION_S = 30.0
MIN_ALIGNMENT_CONFIDENCE = 0.6   # skip word segments below this confidence


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_plain_text(markdown_path: Path) -> str:
    """
    Extract clean clinical narrative from a SOAP note for use as an ASR
    reference / forced-alignment target.

    Strips all note-specific artifacts so the result approximates what the
    provider actually *said* rather than the structured document format:

    Removed (demographics / metadata):
      - Key–value header lines in both formats:
          FIRST NAME:  Shatia          (dictation format)
          **FIRST NAME:** Shatia       (ambient/markdown format)
      - Standalone ALL-CAPS annotation lines:
          INTERNAL USE ONLY, DICTATED BUT NOT READ, etc.
      - Visit-type banner lines:
          FOLLOW-UP EVALUATION, ASSUME CARE EVALUATION, INITIAL EVALUATION, etc.
      - Document footer lines:
          Provider signature, Transcriptionist/DD/Transcription Date lines
          [DICTATED BUT NOT READ TO EXPEDITE REPORT.  SIGNED REPORT ON FILE.]

    Removed (structure / formatting):
      - Markdown section headers:  # Chief Complaint,  ## History, etc.
      - Inline SOAP section labels:  SUBJECTIVE:, OBJECTIVE:, ASSESSMENT:, PLAN:
      - Bold/italic markers, table pipes, HTML tags

    Kept:
      - Narrative prose — the sentences the provider dictated or spoke
    """
    text = markdown_path.read_text(encoding="utf-8", errors="replace")

    # ── Inline SOAP section labels ────────────────────────────────────────────
    # Strip FIRST so that "SUBJECTIVE:  The patient..." becomes "  The patient..."
    # before the ALL-CAPS demographic regex runs (otherwise it would eat the
    # whole line because SUBJECTIVE is all-caps).
    text = re.sub(
        r"(?i)\b(subjective|objective|assessment|plan|history of present illness"
        r"|past medical history|past surgical history|social history|family history"
        r"|review of systems|physical exam(?:ination)?|diagnostic(?:s| exam)?|chief complaint"
        r"|medications?|allergies?|hpi|ros|vitals?)\s*:",
        "",
        text,
    )

    # ── Demographics / metadata ───────────────────────────────────────────────
    # Dictation format ALL-CAPS:  FIRST NAME:  value, PROVIDER LAST:  value, etc.
    text = re.sub(
        r"^[ \t]*[A-Z][A-Z /()]+:[ \t]+[^\n]*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Dictation format mixed-case with 2+ spaces after colon:
    #   Date of Birth:  08/31/1993
    #   Record Number:  1.224889.0
    #   Place of Exam:  Columbia
    # Two-space gap is the consistent indicator in these dictation templates.
    text = re.sub(
        r"^[ \t]*[A-Za-z][A-Za-z0-9 /()]{2,30}:[ \t]{2,}[^\n]*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Ambient/markdown format:  **Field Name:** value
    text = re.sub(r"\*\*[^*]+:\*\*[^\n]*", "", text)

    # ── Document annotations (all-caps standalone lines) ─────────────────────
    # e.g. "INTERNAL USE ONLY", "DICTATED BUT NOT READ TO EXPEDITE REPORT..."
    text = re.sub(r"^\s*[A-Z][A-Z .,'()\[\]-]{8,}\s*$", "", text, flags=re.MULTILINE)

    # ── Visit-type banners ────────────────────────────────────────────────────
    # Only match short standalone banner lines (≤8 words), not clinical sentences
    # that happen to start with "initial" or "follow-up".
    text = re.sub(
        r"(?i)^[ \t]*((?:initial|follow.?up|assume care|discharge|re-?evaluation"
        r"|evaluation|consultation|progress note)(?:[ \t]+\S+){0,6})\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # ── Footer: provider signature block ─────────────────────────────────────
    # Match short lines (≤60 chars) that end with a credential — these are
    # standalone signature lines like "Faraz Rahman, D.O." not mid-sentence
    # occurrences like "Chinery, M.D.'s internal medicine service..."
    text = re.sub(
        r"^[ \t]*[^\n]{0,60}(D\.O\.|M\.D\.|D\.C\.|PA-?C?|N\.P\.)\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(?i)^[ \t]*(transcriptionist|dd:|transcription date)[^\n]*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Bracketed document control lines
    text = re.sub(r"\[[^\]]{10,}\]", "", text)

    # ── Markdown formatting ───────────────────────────────────────────────────
    # Section headers: # Title
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    # Bold/italic
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # Table separators and pipes
    text = re.sub(r"^\|[-:| ]+\|?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|", " ", text)
    # HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # ── Collapse whitespace ───────────────────────────────────────────────────
    text = re.sub(r"[ \t]+", " ", text)
    # Drop short residual lines (likely stray labels or single words)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln.split()) >= 4]
    text = " ".join(lines)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Audio chunking helpers
# ─────────────────────────────────────────────────────────────────────────────

def chunk_segments(
    segments: list[dict],
    min_dur: float = MIN_CHUNK_DURATION_S,
    max_dur: float = MAX_CHUNK_DURATION_S,
) -> list[dict]:
    """
    Group WhisperX word-aligned segments into audio chunks of 10–30 seconds.
    Each chunk contains: start_s, end_s, text.
    """
    chunks: list[dict] = []
    buf_words: list[str] = []
    buf_start: Optional[float] = None
    buf_end: float = 0.0

    for seg in segments:
        for word in seg.get("words", []):
            w_start = word.get("start", 0.0)
            w_end = word.get("end", 0.0)
            w_text = word.get("word", "").strip()
            w_conf = word.get("score", 1.0)

            if w_conf < MIN_ALIGNMENT_CONFIDENCE:
                continue  # skip low-confidence alignments
            if not w_text:
                continue

            if buf_start is None:
                buf_start = w_start

            buf_words.append(w_text)
            buf_end = w_end

            current_dur = buf_end - buf_start
            # Flush if over max duration or if we hit a sentence boundary
            if current_dur >= max_dur or (
                current_dur >= min_dur and w_text.endswith((".", "?", "!"))
            ):
                chunks.append({
                    "start_s": buf_start,
                    "end_s": buf_end,
                    "duration_s": round(buf_end - buf_start, 3),
                    "text": " ".join(buf_words).strip(),
                })
                buf_words = []
                buf_start = None

    # Flush remaining
    if buf_words and buf_start is not None:
        dur = buf_end - buf_start
        if dur >= min_dur:
            chunks.append({
                "start_s": buf_start,
                "end_s": buf_end,
                "duration_s": round(dur, 3),
                "text": " ".join(buf_words).strip(),
            })

    return chunks


def extract_audio_segment(
    audio_path: Path,
    start_s: float,
    end_s: float,
    output_path: Path,
) -> bool:
    """Cut a segment from audio_path using ffmpeg. Returns True on success."""
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-to", str(end_s),
        "-i", str(audio_path),
        "-ar", "16000",   # 16kHz mono for Whisper
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def augment_audio(audio_array, sample_rate: int = 16000):
    """
    Apply speed perturbation (±10%) and additive Gaussian noise (SNR 15–25 dB).

    Requires: audiomentations (pip install audiomentations)
    Falls back silently if not installed.
    """
    try:
        import numpy as np
        from audiomentations import Compose, AddGaussianSNR, TimeStretch

        augment = Compose([
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            AddGaussianSNR(min_snr_db=15.0, max_snr_db=25.0, p=0.5),
        ])
        return augment(samples=audio_array, sample_rate=sample_rate)
    except ImportError:
        return audio_array  # skip augmentation if not available


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def discover_samples(
    provider_id: str,
    modes: list[str] | None = None,
) -> list[dict]:
    """
    Find all audio+gold-note sample pairs available for this provider.
    Returns list of {audio_path, gold_path, sample_id, mode}.

    Args:
        modes: Which recording modes to include.  Defaults to ["dictation"] only.
               Pass ["dictation", "ambient"] to include ambient samples — but note
               that ambient samples require diarization before forced alignment can
               produce reliable (audio, gold-text) training pairs.  Until a
               diarization-aware pipeline is built, ambient samples are excluded
               to avoid training on mixed physician+patient audio against a
               physician-only reference.
    """
    if modes is None:
        modes = ["dictation"]

    mode_config = {
        "dictation": {
            "audio_names": ["dictation.mp3"],
            "gold_names": ["final_soap_note.md", "soap_final.md"],
            "data_subdir": DATA_DIR / "dictation",
        },
        "ambient": {
            "audio_names": ["conversation_audio.mp3", "note_audio.mp3"],
            "gold_names": ["final_soap_note.md", "soap_initial.md", "soap_final.md"],
            "data_subdir": DATA_DIR / "conversation",
        },
    }

    pairs = []
    for mode in modes:
        cfg = mode_config.get(mode)
        if cfg is None:
            log.warning("Unknown mode '%s' — skipping", mode)
            continue
        data_subdir = cfg["data_subdir"]
        if not data_subdir.exists():
            log.warning("Data directory not found: %s — skipping mode '%s'", data_subdir, mode)
            continue

        # Walk nested: data_subdir/<physician>/<encounter>/
        for physician_dir in sorted(data_subdir.iterdir()):
            if not physician_dir.is_dir():
                continue
            for sample_dir in sorted(physician_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue

                audio = None
                for audio_name in cfg["audio_names"]:
                    candidate = sample_dir / audio_name
                    if candidate.exists():
                        audio = candidate
                        break
                if audio is None:
                    continue

                gold = None
                for name in cfg["gold_names"]:
                    if (sample_dir / name).exists():
                        gold = sample_dir / name
                        break
                if gold is None:
                    continue

                pairs.append({
                    "sample_id": sample_dir.name,
                    "audio_path": audio,
                    "gold_path": gold,
                    "mode": mode,
                })

    log.info(
        "Discovered %d sample pairs for provider %s (modes: %s)",
        len(pairs), provider_id, modes,
    )
    return pairs


def process_sample(
    sample: dict,
    out_dir: Path,
    augment: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """
    Run forced alignment on one sample and return a list of chunk records.
    Each record: {audio_file, text, duration_s, sample_id, mode}.
    """
    sample_id = sample["sample_id"]
    audio_path = sample["audio_path"]
    gold_path = sample["gold_path"]

    log.info("Processing sample %s (%s)", sample_id, sample["mode"])

    # 1. Extract clean clinical narrative from gold note.
    #    This is the TRAINING TARGET — what we want the model to produce.
    #    For dictation, this closely matches what the physician actually said
    #    (minus structural artifacts stripped by extract_plain_text).
    plain_text = extract_plain_text(gold_path)
    if len(plain_text) < 50:
        log.warning("  Gold note too short (%d chars) — skipping", len(plain_text))
        return []

    log.info("  Gold text: %d words", len(plain_text.split()))

    if dry_run:
        # Estimate ~1 chunk per 20 words of gold text
        estimated_chunks = max(1, len(plain_text.split()) // 20)
        log.info("  DRY RUN — estimated %d chunks", estimated_chunks)
        return [{"dry_run": True, "sample_id": sample_id, "chunk_count": estimated_chunks}]

    import whisperx  # lazy import — requires GPU env

    # 2. Load audio
    audio = whisperx.load_audio(str(audio_path))
    audio_duration_s = len(audio) / 16000

    # 3. Forced alignment of GOLD TEXT against audio (wav2vec2).
    #
    #    We pass the gold text as a single pseudo-segment spanning the whole
    #    audio.  WhisperX's wav2vec2 aligner finds where each word in the gold
    #    text occurs in the audio, giving us accurate word-level timestamps.
    #
    #    This is the critical correctness fix vs using Whisper's own transcript
    #    as labels.  Self-labeling with Whisper output would teach the model to
    #    reproduce its own errors rather than converge toward the gold text.
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("  Running wav2vec2 forced alignment of gold text…")
    align_model, align_metadata = whisperx.load_align_model(
        language_code="en", device=device
    )

    # Feed gold text as a single segment covering full audio duration
    gold_segments = [{"start": 0.0, "end": audio_duration_s, "text": plain_text}]
    try:
        aligned = whisperx.align(
            gold_segments, align_model, align_metadata, audio, device,
            return_char_alignments=False,
        )
    except Exception as exc:
        log.warning("  Forced alignment failed (%s) — skipping %s", exc, sample_id)
        del align_model
        return []
    del align_model

    if not aligned.get("segments"):
        log.warning("  No aligned segments produced — skipping %s", sample_id)
        return []

    # 4. Chunk into 10–30s segments, preserving gold text as the label
    chunks = chunk_segments(aligned["segments"])
    log.info("  Got %d chunks from sample %s", len(chunks), sample_id)

    # 5. Extract audio segments + (optionally) augment
    sample_out = out_dir / sample_id
    sample_out.mkdir(parents=True, exist_ok=True)

    records = []
    for i, chunk in enumerate(chunks):
        wav_path = sample_out / f"chunk_{i:04d}.wav"
        ok = extract_audio_segment(audio_path, chunk["start_s"], chunk["end_s"], wav_path)
        if not ok:
            continue

        records.append({
            "audio": str(wav_path),
            "sentence": chunk["text"],
            "duration_s": chunk["duration_s"],
            "sample_id": sample_id,
            "mode": sample["mode"],
        })

        # Augmented copy
        if augment:
            try:
                import numpy as np
                import soundfile as sf
                raw, sr = sf.read(str(wav_path))
                augmented = augment_audio(raw.astype(np.float32), sr)
                aug_path = sample_out / f"chunk_{i:04d}_aug.wav"
                sf.write(str(aug_path), augmented, sr, subtype="PCM_16")
                records.append({
                    "audio": str(aug_path),
                    "sentence": chunk["text"],
                    "duration_s": chunk["duration_s"],
                    "sample_id": sample_id,
                    "mode": sample["mode"],
                    "augmented": True,
                })
            except Exception as e:
                log.warning("  Augmentation failed for chunk %d: %s", i, e)

    return records


def build_dataset(
    provider_id: str,
    augment: bool = False,
    dry_run: bool = False,
    modes: list[str] | None = None,
) -> None:
    """
    Full pipeline: discover → align → chunk → split → save HuggingFace dataset.
    """
    out_dir = OUTPUT_TRAINING_DIR / provider_id
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(provider_id, modes=modes)
    if not samples:
        log.error("No samples found. Check data/dictation/ and data/conversations/")
        return

    all_records: list[dict] = []
    for sample in samples:
        records = process_sample(sample, out_dir / "chunks", augment=augment, dry_run=dry_run)
        all_records.extend(records)

    if dry_run:
        total_chunks = sum(r.get("chunk_count", 0) for r in all_records)
        log.info("DRY RUN — %d samples, ~%d chunks would be generated", len(samples), total_chunks)
        return

    if not all_records:
        log.error("No records produced — check audio paths and WhisperX installation")
        return

    log.info("Total records: %d", len(all_records))

    # 90/10 train/eval split
    from datasets import Dataset, DatasetDict
    import random

    random.shuffle(all_records)
    split_idx = max(1, int(len(all_records) * 0.9))
    train_records = all_records[:split_idx]
    eval_records = all_records[split_idx:]

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)
    ds = DatasetDict({"train": train_ds, "eval": eval_ds})

    save_path = out_dir / "dataset"
    ds.save_to_disk(str(save_path))
    log.info("Dataset saved to %s", save_path)

    # Manifest
    total_hours = sum(r.get("duration_s", 0) for r in all_records) / 3600
    manifest = {
        "provider_id": provider_id,
        "modes": modes or ["dictation"],
        "samples_used": len(samples),
        "total_chunks": len(all_records),
        "train_chunks": len(train_records),
        "eval_chunks": len(eval_records),
        "total_hours": round(total_hours, 3),
        "augmented": augment,
        "label_source": "gold_note",  # forced-aligned gold text, not Whisper output
        "dataset_path": str(save_path),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to %s", manifest_path)
    log.info(
        "Summary: %d chunks (%.2f hours) — train=%d, eval=%d",
        len(all_records), total_hours, len(train_records), len(eval_records),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare physician-specific ASR training data for Whisper LoRA fine-tuning"
    )
    parser.add_argument(
        "--provider", required=True,
        help="Provider ID (must match a config/providers/{id}.yaml file)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply speed perturbation + noise augmentation (doubles dataset size)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be produced without running alignment or writing files",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["dictation"],
        choices=["dictation", "ambient"],
        help=(
            "Recording modes to include in training data (default: dictation only). "
            "Ambient mode requires diarization-aware preprocessing and is not yet "
            "supported — physician and patient speech must be separated before "
            "forced alignment against the gold note can produce reliable labels."
        ),
    )
    args = parser.parse_args()

    build_dataset(
        provider_id=args.provider,
        augment=args.augment,
        dry_run=args.dry_run,
        modes=args.modes,
    )


if __name__ == "__main__":
    main()
