"""
DELIVERY NODE — EHR push, finalization, and audit.

Real implementation (Session 5 / Session 8):
  PRE:  map_ehr_fields → convert_format → select_delivery_method → tag_phi_audit
  CORE: push_to_ehr (via MCP EHR tool server or browser extension)
  POST: confirm_delivery → write_audit_log → apply_retention_policy
        → finalize_encounter → emit_quality_metrics

Session 1: Stub — records delivery as clipboard copy (no actual push).
"""

from __future__ import annotations

import logging
import time

from orchestrator.state import DeliveryMethod, EncounterState, EncounterStatus

logger = logging.getLogger(__name__)


def delivery_node(state: EncounterState) -> dict:
    """Deliver final note to EHR / clipboard (stub)."""
    logger.info("delivery_node: start", extra={"encounter_id": state.encounter_id})
    t0 = time.monotonic_ns()

    note_text = state.final_note.to_text() if state.final_note else "[no note]"
    method = state.delivery_method

    # --- stub: pretend we delivered to clipboard ---
    delivery_result = {
        "method": method.value,
        "success": True,
        "note_chars": len(note_text),
        "stub": True,
        "message": f"[STUB] Note ready for {method.value} delivery ({len(note_text)} chars).",
    }

    elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
    logger.info(
        "delivery_node: done",
        extra={"encounter_id": state.encounter_id, "method": method.value},
    )

    pipeline_end_ms = int(time.time() * 1000)
    updated_metrics = state.metrics.model_copy(
        update={
            "pipeline_end_ms": pipeline_end_ms,
            "delivery_success": True,
            "nodes_completed": state.metrics.nodes_completed + ["delivery"],
        }
    )

    return {
        "status": EncounterStatus.DELIVERED,
        "delivery_result": delivery_result,
        "metrics": updated_metrics,
    }
