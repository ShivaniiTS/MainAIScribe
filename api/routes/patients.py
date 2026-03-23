"""
api/routes/patients.py — Patient search via stub EHR roster.

Searches config/ehr_stub/patient_roster.json for matching patients.
Future: replaced by live EHR integration (FHIR R4, HL7) — same interface.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/patients", tags=["patients"])

from config.paths import CONFIG_DIR
ROSTER_PATH = CONFIG_DIR / "ehr_stub" / "patient_roster.json"

_roster_cache: list[dict] | None = None


class PatientResult(BaseModel):
    id: str
    first_name: str
    last_name: str
    date_of_birth: str
    sex: str
    mrn: str
    practice_id: str


def _load_roster() -> list[dict]:
    global _roster_cache
    if _roster_cache is not None:
        return _roster_cache
    if ROSTER_PATH.exists():
        data = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
        _roster_cache = data.get("patients", [])
    else:
        _roster_cache = []
    return _roster_cache


@router.get("/search", response_model=list[PatientResult])
def search_patients(
    q: str = Query("", description="Search by name, MRN, or DOB"),
    provider_id: Optional[str] = Query(None, description="Filter by provider's practice"),
):
    """Search patient roster. Returns up to 10 matching patients."""
    roster = _load_roster()
    query = q.strip().lower()

    if not query:
        # Return first 10 if no query
        results = roster[:10]
    else:
        results = []
        for p in roster:
            full_name = f"{p['first_name']} {p['last_name']}".lower()
            mrn = str(p.get("mrn", ""))
            dob = p.get("date_of_birth", "")
            if query in full_name or query in mrn or query in dob:
                results.append(p)
            if len(results) >= 10:
                break

    return [PatientResult(**p) for p in results]
