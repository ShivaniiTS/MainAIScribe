"""
CONTEXT NODE — Pre-encounter context assembly.

Loads patient demographics, encounter info, and clinical context
via the EHR adapter from the engine registry.

For stubbed mode: reads from patient_demographics.json + encounter_details.json
via StubEHRServer.

For production: will read from FHIR, HL7v2, or browser extension adapters.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from orchestrator.state import (
    ContextPacket,
    EncounterContext,
    EncounterState,
    EncounterStatus,
    FacilityContext,
    PatientDemographics,
    ProviderContext,
)

logger = logging.getLogger(__name__)


def _find_context_files(state: EncounterState) -> Optional[Path]:
    """Locate patient_demographics.json (or legacy patient_context.yaml) near the audio file.

    Returns the parent directory if context files are found, None otherwise.
    """
    if state.audio_file_path:
        audio_dir = Path(state.audio_file_path).parent
        # New format: JSON files
        if (audio_dir / "patient_demographics.json").exists():
            return audio_dir
        if (audio_dir / "encounter_details.json").exists():
            return audio_dir
        # Legacy fallback: YAML
        if (audio_dir / "patient_context.yaml").exists():
            return audio_dir
    return None


def context_node(state: EncounterState) -> dict:
    """Load pre-encounter patient context from EHR adapter."""
    logger.info("context_node: start", extra={"encounter_id": state.encounter_id})
    t0 = time.monotonic_ns()

    context_packet = ContextPacket(source="manual")

    try:
        ctx_dir = _find_context_files(state)
        if ctx_dir:
            from mcp_servers.registry import get_registry
            ehr = get_registry().get_ehr()

            # Set the context path for StubEHRServer
            if hasattr(ehr, "set_context_dir"):
                ehr.set_context_dir(ctx_dir)
            elif hasattr(ehr, "set_context_path"):
                # Legacy: single YAML file
                yaml_path = ctx_dir / "patient_context.yaml"
                if yaml_path.exists():
                    ehr.set_context_path(yaml_path)

            # Load patient demographics (sync wrapper for async)
            import asyncio
            from mcp_servers.ehr.base import PatientIdentifier
            patient_ehr = asyncio.get_event_loop().run_until_complete(
                ehr.get_patient(PatientIdentifier(mrn=state.patient_id))
            )
            patient = PatientDemographics(
                id=patient_ehr.id,
                name=f"{patient_ehr.given_name or ''} {patient_ehr.family_name or ''}".strip() or None,
                dob=patient_ehr.dob,
                sex=patient_ehr.sex,
                mrn=patient_ehr.mrn,
            )

            # Load encounter context (stub-specific accessors)
            encounter_data = {}
            provider_data = {}
            facility_data = {}
            if hasattr(ehr, "get_encounter_context"):
                encounter_data = ehr.get_encounter_context()
            if hasattr(ehr, "get_provider_context"):
                provider_data = ehr.get_provider_context()
            if hasattr(ehr, "get_facility_context"):
                facility_data = ehr.get_facility_context()

            encounter_ctx = EncounterContext(
                date_of_service=encounter_data.get("date_of_service"),
                visit_type=encounter_data.get("visit_type"),
                date_of_injury=encounter_data.get("date_of_injury"),
                case_number=encounter_data.get("case_number"),
            ) if encounter_data else None

            provider_ctx = ProviderContext(
                name=provider_data.get("name"),
                credentials=provider_data.get("credentials"),
                specialty=provider_data.get("specialty"),
            ) if provider_data else None

            facility_ctx = FacilityContext(
                name=facility_data.get("name"),
                location=facility_data.get("location"),
            ) if facility_data else None

            context_packet = ContextPacket(
                patient=patient,
                encounter=encounter_ctx,
                provider_context=provider_ctx,
                facility=facility_ctx,
                loaded_at=datetime.now(timezone.utc),
                source="stub",
            )
            logger.info(
                "context_node: loaded patient context",
                extra={
                    "encounter_id": state.encounter_id,
                    "patient_name": patient.name,
                    "source": str(ctx_dir),
                },
            )
        else:
            logger.info(
                "context_node: no context files found, using empty context",
                extra={"encounter_id": state.encounter_id},
            )

    except Exception as exc:
        logger.warning(
            "context_node: failed to load context, continuing with empty",
            extra={"encounter_id": state.encounter_id, "error": str(exc)},
        )

    elapsed_ms = (time.monotonic_ns() - t0) // 1_000_000
    logger.info("context_node: done", extra={"encounter_id": state.encounter_id, "elapsed_ms": elapsed_ms})

    return {
        "status": EncounterStatus.CONTEXT_LOADING,
        "context_packet": context_packet,
        "metrics": state.metrics.model_copy(
            update={
                "context_load_ms": elapsed_ms,
                "nodes_completed": state.metrics.nodes_completed + ["context"],
            }
        ),
    }
