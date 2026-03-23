"""
CAPTURE NODE — Audio capture and buffering.

Real implementation (Session 3):
  PRE:  detect_audio_source → normalize_sample_rate → suppress_noise → normalize_gain
        → detect_voice_activity → score_audio_quality
  CORE: stream_and_buffer (chunked audio capture)
  POST: tag_chunk_metadata → detect_commands → merge_addendums → manage_offline_buffer
        → archive_audio

Session 1: Stub — passes audio_file_path through to the transcribe node.
"""

from __future__ import annotations

import logging
import time

from orchestrator.state import AudioSegment, EncounterState, EncounterStatus, RecordingMode

logger = logging.getLogger(__name__)


def capture_node(state: EncounterState) -> dict:
    """Accept and buffer audio (stub)."""
    logger.info("capture_node: start", extra={"encounter_id": state.encounter_id})
    t0 = time.monotonic_ns()

    audio_segments = state.audio_segments

    # --- stub: if a file path was provided directly, wrap it in one segment ---
    if state.audio_file_path and not audio_segments:
        audio_segments = [
            AudioSegment(
                encounter_id=state.encounter_id,
                sequence_number=0,
                start_ms=0,
                end_ms=0,              # Real node measures actual duration
                mode=state.recording_mode,
                storage_path=state.audio_file_path,
            )
        ]

    elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
    logger.info("capture_node: done", extra={"encounter_id": state.encounter_id, "segments": len(audio_segments)})

    return {
        "status": EncounterStatus.CAPTURING,
        "audio_segments": audio_segments,
        "metrics": state.metrics.model_copy(
            update={"nodes_completed": state.metrics.nodes_completed + ["capture"]}
        ),
    }
