"""
EHR MCP Server Base Interface.

All EHR adapters (FHIR, HL7v2, browser extension, manual) implement EHRAdapter.
The Context node (READ) and Delivery node (WRITE) call only this interface.

Implementations:
    fhir_server.py          — FHIR R4 (Epic, Cerner, Athena)
    hl7v2_server.py         — HL7v2 (legacy systems)
    extension_server.py     — Browser extension DOM scraping/injection
    manual_server.py        — Manual entry fallback (DEFAULT)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PatientIdentifier:
    mrn: Optional[str] = None
    fhir_id: Optional[str] = None
    npi: Optional[str] = None    # Ordering provider NPI


@dataclass
class EHRPatient:
    id: str
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    dob: Optional[str] = None       # ISO 8601
    sex: Optional[str] = None
    mrn: Optional[str] = None


@dataclass
class EHRProblem:
    code: str
    system: str = "ICD-10"
    description: str = ""
    status: str = "active"
    onset_date: Optional[str] = None


@dataclass
class EHRMedication:
    name: str
    code: Optional[str] = None     # RxNorm
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    status: str = "active"


@dataclass
class EHRAllergy:
    substance: str
    reaction: Optional[str] = None
    severity: Optional[str] = None   # mild | moderate | severe
    status: str = "active"


@dataclass
class EHRLabResult:
    name: str
    value: str
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    date: Optional[str] = None
    flag: Optional[str] = None     # H | L | HH | LL


@dataclass
class EHRNote:
    text: str
    note_type: str = ""
    date: Optional[str] = None
    author: Optional[str] = None


@dataclass
class PushResult:
    success: bool
    method: str
    note_id: Optional[str] = None   # EHR-assigned note ID
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class NavigationResult:
    success: bool
    action: str
    screen: Optional[str] = None
    error: Optional[str] = None


class EHRAdapter(ABC):
    """
    Abstract base class for all EHR MCP servers.

    READ methods are called by the Context node (pre-encounter context assembly).
    WRITE methods are called by the Delivery node (post-encounter note push).
    NAVIGATE methods are called by the voice command handler.
    """

    # --- READ ---------------------------------------------------------------

    @abstractmethod
    async def get_patient(self, identifier: PatientIdentifier) -> EHRPatient:
        """Fetch patient demographics."""
        ...

    @abstractmethod
    async def get_problem_list(self, patient_id: str) -> list[EHRProblem]:
        """Fetch active problem list."""
        ...

    @abstractmethod
    async def get_medications(self, patient_id: str) -> list[EHRMedication]:
        """Fetch active medication list."""
        ...

    @abstractmethod
    async def get_allergies(self, patient_id: str) -> list[EHRAllergy]:
        """Fetch allergy list."""
        ...

    @abstractmethod
    async def get_recent_labs(
        self,
        patient_id: str,
        days: int = 90,
    ) -> list[EHRLabResult]:
        """Fetch recent lab results (default: past 90 days)."""
        ...

    @abstractmethod
    async def get_last_visit_note(self, patient_id: str) -> Optional[EHRNote]:
        """Fetch the most recent clinical note for context."""
        ...

    # --- WRITE --------------------------------------------------------------

    @abstractmethod
    async def push_note(
        self,
        patient_id: str,
        note: EHRNote,
        encounter_id: Optional[str] = None,
    ) -> PushResult:
        """
        Push the finalized clinical note into the EHR.

        Args:
            patient_id:   EHR patient identifier.
            note:         The clinical note to push.
            encounter_id: Encounter/visit identifier (if known).
        """
        ...

    # --- NAVIGATE -----------------------------------------------------------

    async def navigate(self, command: str) -> NavigationResult:
        """
        Execute a voice navigation command in the EHR.

        Not all adapters support navigation (e.g., manual entry does not).
        Default: return not-supported.
        """
        return NavigationResult(
            success=False,
            action=command,
            error="Navigation not supported by this adapter.",
        )

    # --- Lifecycle ----------------------------------------------------------

    async def health_check(self) -> bool:
        """Verify the EHR adapter is reachable."""
        return True

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Server", "").replace("Adapter", "").lower()
