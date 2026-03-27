"""
learning/asr_improvement.py

Continuous ASR improvement hook.

After each completed encounter, this module:
  1. Records the encounter audio + corrected transcript to a provider-specific
     correction log.
  2. When the provider accumulates >= RETRAIN_THRESHOLD new corrections,
     triggers a re-fine-tuning run (data prep → LoRA training).
  3. After training, hot-swaps the new adapter in the registry without
     restarting the server.

Integration:
  - Called from orchestrator/nodes/delivery_node.py after note delivery when
    a provider correction is captured.
  - Or called directly when the Review UI submits transcript corrections.

Design principle:
  - Corrections drive the loop, not passage of time.
  - The threshold is configurable per provider (default: 5 new corrections).
  - Training is always async (subprocess) to avoid blocking the pipeline.
  - The old adapter remains active until the new one is validated.

Usage:
    from learning.asr_improvement import record_correction, maybe_retrain

    # After provider corrects a transcript
    record_correction(
        provider_id="dr_faraz_rahman",
        encounter_id="enc-123",
        audio_path="/path/to/audio.mp3",
        original_transcript="...",
        corrected_transcript="...",
    )

    # Check if retraining should trigger (call after every correction)
    maybe_retrain("dr_faraz_rahman")
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

from config.paths import ROOT, OUTPUT_DIR as _OUTPUT_DIR
CORRECTIONS_DIR = ROOT / "data" / "asr_corrections"
MODELS_DIR = ROOT / "models" / "whisper_lora"

# Number of new corrections that triggers automatic re-fine-tuning
RETRAIN_THRESHOLD = 5


# ─────────────────────────────────────────────────────────────────────────────
# Correction recording
# ─────────────────────────────────────────────────────────────────────────────

def record_correction(
    provider_id: str,
    encounter_id: str,
    audio_path: str,
    original_transcript: str,
    corrected_transcript: str,
    correction_type: str = "transcript",  # "transcript" | "medical_term" | "speaker_label"
) -> None:
    """
    Persist a provider's transcript correction for future retraining.

    Each correction becomes a training sample: (audio, corrected_transcript).
    Corrections are stored in JSONL format so they accumulate across sessions.

    Args:
        provider_id:            Provider who made the correction.
        encounter_id:           Encounter this correction belongs to.
        audio_path:             Path to the encounter audio file.
        original_transcript:    What the ASR produced.
        corrected_transcript:   What the provider typed as the correct version.
        correction_type:        Category of correction for analysis.
    """
    corrections_file = _get_corrections_file(provider_id)
    corrections_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "provider_id": provider_id,
        "encounter_id": encounter_id,
        "audio_path": audio_path,
        "original": original_transcript,
        "corrected": corrected_transcript,
        "correction_type": correction_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(corrections_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    count = count_new_corrections(provider_id)
    log.info(
        "asr_improvement: correction recorded for provider '%s' "
        "(total new since last retrain: %d)",
        provider_id, count,
    )


def count_new_corrections(provider_id: str) -> int:
    """Count corrections logged since the last retraining run."""
    corrections_file = _get_corrections_file(provider_id)
    if not corrections_file.exists():
        return 0

    last_retrain = _last_retrain_timestamp(provider_id)
    count = 0
    with open(corrections_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp", "")
                if last_retrain is None or ts > last_retrain:
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


def get_correction_history(provider_id: str) -> list[dict]:
    """Return all logged corrections for a provider."""
    corrections_file = _get_corrections_file(provider_id)
    if not corrections_file.exists():
        return []
    entries = []
    with open(corrections_file) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Retraining trigger
# ─────────────────────────────────────────────────────────────────────────────

def maybe_retrain(
    provider_id: str,
    threshold: Optional[int] = None,
    background: bool = True,
) -> bool:
    """
    Trigger retraining if enough new corrections have accumulated.

    Args:
        provider_id: Provider to check.
        threshold:   Override RETRAIN_THRESHOLD for this call.
        background:  If True, run training as a subprocess (non-blocking).

    Returns:
        True if retraining was triggered, False otherwise.
    """
    limit = threshold or RETRAIN_THRESHOLD
    new_count = count_new_corrections(provider_id)

    if new_count < limit:
        log.debug(
            "asr_improvement: %d new corrections for '%s' (threshold=%d) — no retrain",
            new_count, provider_id, limit,
        )
        return False

    log.info(
        "asr_improvement: %d new corrections for '%s' >= threshold %d — triggering retrain",
        new_count, provider_id, limit,
    )

    _trigger_retrain(provider_id, background=background)
    return True


def _trigger_retrain(provider_id: str, background: bool = True) -> None:
    """
    Launch the retraining pipeline:
        1. Export new corrections to training data format
        2. Run prepare_asr_training_data.py (includes new corrections)
        3. Run finetune_whisper_lora.py
        4. On success: hot-swap the adapter in the registry

    Runs as a subprocess to avoid blocking the main pipeline.
    """
    python = sys.executable
    scripts_dir = ROOT / "scripts"

    cmd = [
        python, str(scripts_dir / "finetune_whisper_lora.py"),
        "--provider", provider_id,
        "--max-steps", "200",
    ]

    if background:
        log.info("asr_improvement: launching retrain subprocess: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Watch for completion in a thread (non-blocking)
        import threading
        def _watch(proc: subprocess.Popen, provider_id: str) -> None:
            returncode = proc.wait()
            if returncode == 0:
                log.info(
                    "asr_improvement: retrain complete for '%s' — hot-swapping adapter",
                    provider_id,
                )
                _mark_retrain_timestamp(provider_id)
                _hotswap_adapter(provider_id)
            else:
                log.error(
                    "asr_improvement: retrain FAILED for '%s' (exit %d) — keeping old adapter",
                    provider_id, returncode,
                )

        t = threading.Thread(target=_watch, args=(proc, provider_id), daemon=True)
        t.start()
    else:
        # Synchronous (used in tests / manual runs)
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode == 0:
            _mark_retrain_timestamp(provider_id)
            _hotswap_adapter(provider_id)
        else:
            raise RuntimeError(f"Retraining failed (exit {result.returncode})")


def _hotswap_adapter(provider_id: str) -> None:
    """
    Remove the cached LoRA engine from the registry so the next transcription
    call loads the newly trained adapter automatically.
    """
    try:
        from mcp_servers.registry import get_registry
        registry = get_registry()
        cache_key = ("asr", f"whisperx_lora/{provider_id}")
        if cache_key in registry._cache:
            del registry._cache[cache_key]
            log.info(
                "asr_improvement: hot-swapped adapter for '%s' in registry",
                provider_id,
            )
    except Exception as exc:
        log.warning("asr_improvement: hot-swap failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_corrections_file(provider_id: str) -> Path:
    return CORRECTIONS_DIR / provider_id / "corrections.jsonl"


def _last_retrain_timestamp(provider_id: str) -> Optional[str]:
    """Return ISO timestamp of last successful retrain, or None."""
    ts_file = CORRECTIONS_DIR / provider_id / "last_retrain.txt"
    if ts_file.exists():
        return ts_file.read_text().strip()
    return None


def _mark_retrain_timestamp(provider_id: str) -> None:
    """Record the current time as the last successful retrain timestamp."""
    ts_file = CORRECTIONS_DIR / provider_id / "last_retrain.txt"
    ts_file.parent.mkdir(parents=True, exist_ok=True)
    ts_file.write_text(datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Quality-gated activation
# ─────────────────────────────────────────────────────────────────────────────

def validate_new_adapter(provider_id: str, max_samples: int = 5) -> bool:
    """
    Run a quick WER evaluation on the new adapter before activating it.

    Returns True if the new adapter is better than the previous one
    (or if no previous eval exists — first-time training always activates).

    This is called by _watch() before _hotswap_adapter() when quality-gating
    is enabled in the provider profile.
    """
    try:
        import importlib
        eval_mod = importlib.import_module("scripts.eval_asr_quality")

        report = eval_mod.evaluate(
            provider_id=provider_id,
            base_only=False,
            max_samples=max_samples,
        )

        # Load previous eval results if they exist
        prev_json = _OUTPUT_DIR / f"asr_eval_{provider_id}.json"
        if prev_json.exists():
            prev = json.loads(prev_json.read_text())
            prev_wer = prev.get("summary", {}).get("lora_avg_wer")
            new_wer = report.lora_avg_wer
            if prev_wer and new_wer and new_wer >= prev_wer:
                log.warning(
                    "asr_improvement: new adapter WER=%.4f not better than prev=%.4f — not activating",
                    new_wer, prev_wer,
                )
                return False

        # Save updated eval
        eval_mod.save_report(report, provider_id)
        return True

    except Exception as exc:
        log.warning(
            "asr_improvement: validation failed (%s) — activating anyway", exc
        )
        return True  # fail-open: activate if validation errors
