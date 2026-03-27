"""
REVIEW NODE — Human-in-the-loop review and correction capture.

Real implementation (Session 6):
  PRE:  render_for_editor → highlight_low_confidence → prepare_diff → preload_audio_clips
  CORE: human_review (HITL — waits for provider approval via web UI or API)
  POST: capture_corrections → classify_corrections
        → generate_training_pairs → update_style_model → refine_template → log_quality

Session 1: Stub — auto-approves and passes note through unchanged.
"""

from __future__ import annotations

import logging

from orchestrator.state import EncounterState, EncounterStatus

logger = logging.getLogger(__name__)


def review_node(state: EncounterState) -> dict:
    """HITL review — auto-approve in stub mode."""
    logger.info("review_node: start (auto-approve stub)", extra={"encounter_id": state.encounter_id})

    # In real implementation: wait for provider to edit/approve note via web UI.
    # The FastAPI backend's review endpoint will push corrections back into state.
    final_note = state.generated_note

    logger.info("review_node: done", extra={"encounter_id": state.encounter_id})

    return {
        "status": EncounterStatus.REVIEWING,
        "final_note": final_note,
        "review_approved": True,
        "metrics": state.metrics.model_copy(
            update={"nodes_completed": state.metrics.nodes_completed + ["review"]}
        ),
    }
