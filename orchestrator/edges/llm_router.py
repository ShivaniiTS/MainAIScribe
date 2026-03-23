"""
LLM Router — conditional edge after the NOTE node.

Decides what happens next based on note generation quality:

    ┌──────┐
    │ NOTE │
    └──┬───┘
       │
    ┌──▼─────────────────────────────────────────────────┐
    │ llm_router(state)                                   │
    │  • "review"     — normal path (always for now)      │
    │  • "error"      — future: LLM totally failed        │
    └─────────────────────────────────────────────────────┘

Session 4: Single-engine — always routes to "review".
Future sessions:
  - Route to a more capable model when confidence < threshold.
  - Route to error node when LLM fallback stub was used (and manual entry needed).
  - Route to retry node after provider feedback on review.
"""

from __future__ import annotations

import logging

from orchestrator.state import EncounterState

logger = logging.getLogger(__name__)

# Below this note confidence, log a warning (but still route to review).
_CONFIDENCE_WARN_THRESHOLD = 0.50

# LLM fallback stub marker in section content.
_FALLBACK_MARKER = "[LLM UNAVAILABLE]"


def llm_router(state: EncounterState) -> str:
    """
    Conditional edge function: returns the name of the next node.

    Args:
        state: Current EncounterState (post-note-generation).

    Returns:
        Node name string — "review" (always in Session 4).
    """
    note = state.generated_note
    confidence = state.metrics.note_confidence or 0.0
    engine_used = state.llm_engine_used or "unknown"

    # Hard failure: no note generated at all
    if not note:
        logger.error(
            "llm_router: no generated_note — routing to review (will use empty note)",
            extra={"encounter_id": state.encounter_id},
        )
        return "review"

    # Fallback stub: LLM was unavailable
    is_fallback = any(_FALLBACK_MARKER in s.content for s in note.sections)
    if is_fallback:
        logger.warning(
            "llm_router: LLM fallback stub detected (engine=%s) — routing to review",
            engine_used,
            extra={"encounter_id": state.encounter_id},
        )
        return "review"

    # Low-confidence: log warning, still proceed (no retry engine in Session 4)
    if confidence < _CONFIDENCE_WARN_THRESHOLD:
        logger.warning(
            "llm_router: low note confidence %.2f (engine=%s) — routing to review anyway",
            confidence,
            engine_used,
            extra={"encounter_id": state.encounter_id},
        )
        return "review"

    logger.debug(
        "llm_router: confidence=%.2f engine=%s → review",
        confidence,
        engine_used,
        extra={"encounter_id": state.encounter_id},
    )
    return "review"
