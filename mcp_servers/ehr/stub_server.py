"""
Stub EHR Server — reads patient context from local JSON/YAML files.

Implements the full EHRAdapter interface but reads from
patient_demographics.json + encounter_details.json (or legacy patient_context.yaml)
instead of a live EHR system.  When real EHR integration is built,
only this file is replaced — the pipeline code is untouched.

Usage:
    server = StubEHRServer.from_config({"type": "stub"})
    patient = await server.get_patient(PatientIdentifier(mrn="1.226680.0"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from mcp_servers.ehr.base import (
    EHRAdapter,
    EHRAllergy,
    EHRLabResult,
    EHRMedication,
    EHRNote,
    EHRPatient,
    EHRProblem,
    NavigationResult,
    PatientIdentifier,
    PushResult,
)

log = logging.getLogger(__name__)


class StubEHRServer(EHRAdapter):
    """
    Reads patient context from local JSON files (new format) or YAML (legacy).

    The context_dir can be set per-encounter via set_context_dir()
    before calling any read methods.
    """

    def __init__(self, data_dir: str = "ai-scribe-data") -> None:
        self._data_dir = Path(data_dir)
        self._context_dir: Optional[Path] = None
        self._context_path: Optional[Path] = None  # legacy YAML path
        self._cache: dict[str, dict] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "StubEHRServer":
        return cls(data_dir=cfg.get("data_dir", "ai-scribe-data"))

    def set_context_dir(self, dir_path: str | Path) -> None:
        """Set the encounter directory containing JSON context files."""
        self._context_dir = Path(dir_path)
        self._context_path = None
        self._cache.clear()

    def set_context_path(self, path: str | Path) -> None:
        """Legacy: set a single patient_context.yaml path."""
        self._context_path = Path(path)
        self._context_dir = None
        self._cache.clear()

    def _load_context(self) -> dict:
        """Load and cache context from JSON files or legacy YAML."""
        cache_key = str(self._context_dir or self._context_path or "none")
        if cache_key in self._cache:
            return self._cache[cache_key]

        ctx: dict = {}

        if self._context_dir:
            # New format: patient_demographics.json + encounter_details.json
            demo_path = self._context_dir / "patient_demographics.json"
            enc_path = self._context_dir / "encounter_details.json"

            demo = {}
            enc = {}
            if demo_path.exists():
                demo = json.loads(demo_path.read_text())
                log.debug(f"stub_ehr: loaded demographics from {demo_path}")
            if enc_path.exists():
                enc = json.loads(enc_path.read_text())
                log.debug(f"stub_ehr: loaded encounter details from {enc_path}")

            provider = enc.get("provider", {})
            ctx = {
                "patient": {
                    "name": f"{demo.get('first_name', '')} {demo.get('last_name', '')}".strip() or None,
                    "date_of_birth": demo.get("date_of_birth"),
                    "sex": None,
                    "mrn": demo.get("record_number"),
                },
                "encounter": {
                    "date_of_service": enc.get("date_of_exam"),
                    "visit_type": enc.get("visit_type"),
                    "date_of_injury": enc.get("date_of_accident") or None,
                    "case_number": enc.get("case_number") or None,
                    "encounter_id": enc.get("encounter_id"),
                    "mode": enc.get("mode"),
                    "location": enc.get("location"),
                },
                "provider": {
                    "name": provider.get("full_name"),
                    "credentials": None,
                    "specialty": None,
                },
                "facility": {
                    "name": None,
                    "location": enc.get("location"),
                },
            }

        elif self._context_path and self._context_path.exists():
            # Legacy: single patient_context.yaml
            with open(self._context_path) as f:
                ctx = yaml.safe_load(f) or {}
            log.debug(f"stub_ehr: loaded context from {self._context_path}")
        else:
            log.warning("stub_ehr: no context dir or path set")

        self._cache[cache_key] = ctx
        return ctx

    # ── READ ─────────────────────────────────────────────────────────────

    async def get_patient(self, identifier: PatientIdentifier) -> EHRPatient:
        ctx = self._load_context()
        patient = ctx.get("patient", {})
        name = patient.get("name", "")
        parts = name.split(maxsplit=1) if name else ["", ""]
        given = parts[0] if len(parts) > 0 else ""
        family = parts[1] if len(parts) > 1 else ""
        return EHRPatient(
            id=patient.get("mrn", identifier.mrn or "unknown"),
            given_name=given or None,
            family_name=family or None,
            dob=patient.get("date_of_birth"),
            sex=patient.get("sex"),
            mrn=patient.get("mrn"),
        )

    async def get_problem_list(self, patient_id: str) -> list[EHRProblem]:
        return []

    async def get_medications(self, patient_id: str) -> list[EHRMedication]:
        return []

    async def get_allergies(self, patient_id: str) -> list[EHRAllergy]:
        return []

    async def get_recent_labs(self, patient_id: str, days: int = 90) -> list[EHRLabResult]:
        return []

    async def get_last_visit_note(self, patient_id: str) -> Optional[EHRNote]:
        return None

    # ── Context accessors (stub-specific) ────────────────────────────────

    def get_encounter_context(self) -> dict:
        """Return the encounter block from context data."""
        return self._load_context().get("encounter", {})

    def get_provider_context(self) -> dict:
        """Return the provider block from context data."""
        return self._load_context().get("provider", {})

    def get_facility_context(self) -> dict:
        """Return the facility block from context data."""
        return self._load_context().get("facility", {})

    # ── WRITE ────────────────────────────────────────────────────────────

    async def push_note(
        self,
        patient_id: str,
        note: EHRNote,
        encounter_id: Optional[str] = None,
    ) -> PushResult:
        log.info(f"stub_ehr: push_note called (stub — no-op) for patient {patient_id}")
        return PushResult(success=True, method="stub")

    # ── NAVIGATE ─────────────────────────────────────────────────────────

    async def navigate(self, command: str) -> NavigationResult:
        return NavigationResult(
            success=False,
            action=command,
            error="Navigation not supported by stub adapter.",
        )

    # ── Health ───────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "stub"
