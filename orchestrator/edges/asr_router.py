"""
ASR Router — conditional edge after the TRANSCRIBE node.

Decides what happens next based on transcript quality:

    ┌─────────────┐
    │  TRANSCRIBE │
    └──────┬──────┘
           │
    ┌──────▼─────────────────────────────────────────────┐
    │ asr_router(state)                                   │
    │  • "note"       — normal path (always for now)      │
    │  • "error"      — future: hard failure, no audio    │
    └─────────────────────────────────────────────────────┘

Session 4: Single-engine — always routes to "note".
Future sessions: route to a fallback ASR engine when confidence < threshold,
or to an error node when ASR is unavailable and no fallback exists.
"""

from __future__ import annotations

import logging

from orchestrator.state import EncounterState

logger = logging.getLogger(__name__)

# Minimum ASR confidence to proceed to note generation without a warning.
# Below this, a warning is logged but we still proceed (graceful degradation).
_CONFIDENCE_WARN_THRESHOLD = 0.40

# When a fallback ASR engine is configured (future), retry below this threshold.
_CONFIDENCE_RETRY_THRESHOLD = 0.25


def asr_router(state: EncounterState) -> str:
    """
    Conditional edge function: returns the name of the next node.

    Args:
        state: Current EncounterState (post-transcribe).

    Returns:
        Node name string — "note" (always in Session 4).
    """
    confidence = state.metrics.asr_confidence or 0.0
    engine_used = state.asr_engine_used or "unknown"

    # Hard error path: no transcript at all
    if not state.transcript or not state.transcript.full_text:
        logger.error(
            "asr_router: no transcript — routing to note (will use fallback stub)",
            extra={"encounter_id": state.encounter_id},
        )
        return "note"

    # Fallback stub path: ASR engine failed, transcript contains error marker
    if "ASR UNAVAILABLE" in state.transcript.full_text:
        logger.warning(
            "asr_router: ASR fallback stub detected (engine=%s) — routing to note",
            engine_used,
            extra={"encounter_id": state.encounter_id},
        )
        return "note"

    # Low-confidence path: log but still proceed (no retry engine in Session 4)
    if confidence < _CONFIDENCE_WARN_THRESHOLD:
        logger.warning(
            "asr_router: low ASR confidence %.2f (engine=%s) — routing to note anyway",
            confidence,
            engine_used,
            extra={"encounter_id": state.encounter_id},
        )
        return "note"

    logger.debug(
        "asr_router: confidence=%.2f engine=%s → note",
        confidence,
        engine_used,
        extra={"encounter_id": state.encounter_id},
    )
    return "note"
