"""
EncounterState — the single state object that flows through the LangGraph pipeline.

Every node receives this state and returns a dict of fields to update.
LangGraph merges partial updates back into the state automatically.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EncounterStatus(str, Enum):
    CREATED = "CREATED"
    CONTEXT_LOADING = "CONTEXT_LOADING"
    READY = "READY"
    CAPTURING = "CAPTURING"
    PROCESSING = "PROCESSING"
    REVIEWING = "REVIEWING"
    DELIVERED = "DELIVERED"
    ERROR = "ERROR"


class RecordingMode(str, Enum):
    AMBIENT = "ambient"       # Multi-speaker encounter
    DICTATION = "dictation"   # Single-speaker physician dictation


class DeliveryMethod(str, Enum):
    EXTENSION = "extension"
    FHIR = "fhir"
    CLIPBOARD = "clipboard"
    MANUAL = "manual"


class NoteType(str, Enum):
    SOAP = "SOAP"
    HP = "H&P"
    PROGRESS = "PROGRESS"
    DISCHARGE = "DISCHARGE"


class CorrectionType(str, Enum):
    ASR_ERROR = "ASR_ERROR"
    STYLE = "STYLE"
    CONTENT = "CONTENT"
    CODING = "CODING"
    TEMPLATE = "TEMPLATE"


# ---------------------------------------------------------------------------
# Sub-models: Provider
# ---------------------------------------------------------------------------

class ProviderProfile(BaseModel):
    id: str
    name: str
    specialty: str
    credentials: Optional[str] = None   # "MD", "DO", "NP", "PA", etc.
    npi: Optional[str] = None
    practice_id: Optional[str] = None

    # Note preferences
    note_format: NoteType = NoteType.SOAP
    template_id: str = "soap_default"
    style_directives: list[str] = Field(default_factory=list)
    custom_vocabulary: list[str] = Field(default_factory=list)

    # Engine overrides (None = use defaults from engines.yaml)
    asr_override: Optional[str] = None
    llm_override: Optional[str] = None
    noise_suppression_level: str = "moderate"
    postprocessor_mode: str = "hybrid"

    # Learned quality scores (updated by feedback loop)
    style_model_version: str = "v0"
    correction_count: int = 0
    quality_scores: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sub-models: Context
# ---------------------------------------------------------------------------

class PatientDemographics(BaseModel):
    id: str
    name: Optional[str] = None     # Redacted in transit
    dob: Optional[str] = None      # ISO 8601 date
    sex: Optional[str] = None
    mrn: Optional[str] = None


class EncounterContext(BaseModel):
    """Encounter-level context extracted from EHR or gold notes."""
    date_of_service: Optional[str] = None
    visit_type: Optional[str] = None     # initial_evaluation, follow_up, etc.
    date_of_injury: Optional[str] = None
    case_number: Optional[str] = None


class ProviderContext(BaseModel):
    """Provider info from EHR or gold notes (separate from ProviderProfile)."""
    name: Optional[str] = None
    credentials: Optional[str] = None
    specialty: Optional[str] = None


class FacilityContext(BaseModel):
    """Facility info from EHR or gold notes."""
    name: Optional[str] = None
    location: Optional[str] = None


class Problem(BaseModel):
    code: str
    description: str
    status: str = "active"


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    status: str = "active"


class Allergy(BaseModel):
    substance: str
    reaction: Optional[str] = None
    severity: Optional[str] = None


class LabResult(BaseModel):
    name: str
    value: str
    unit: Optional[str] = None
    date: Optional[str] = None
    flag: Optional[str] = None


class ContextPacket(BaseModel):
    patient: Optional[PatientDemographics] = None
    encounter: Optional[EncounterContext] = None
    provider_context: Optional[ProviderContext] = None
    facility: Optional[FacilityContext] = None
    problem_list: list[Problem] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    recent_labs: list[LabResult] = Field(default_factory=list)
    last_visit_note_summary: Optional[str] = None
    loaded_at: Optional[datetime] = None
    source: str = "manual"   # manual | fhir | extension | stub


# ---------------------------------------------------------------------------
# Sub-models: Audio capture
# ---------------------------------------------------------------------------

class AudioSegment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    encounter_id: str
    sequence_number: int
    start_ms: int
    end_ms: int
    mode: RecordingMode = RecordingMode.AMBIENT
    storage_path: Optional[str] = None
    sample_rate: int = 16000
    snr_estimate: Optional[float] = None


class ModeEvent(BaseModel):
    timestamp_ms: int
    from_mode: RecordingMode
    to_mode: RecordingMode
    triggered_by: str = "manual"   # manual | voice_command


class VoiceCommand(BaseModel):
    timestamp_ms: int
    raw_text: str
    parsed_command: Optional[str] = None
    executed: bool = False


class Addendum(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp_ms: int
    text: str
    author: str = "provider"


# ---------------------------------------------------------------------------
# Sub-models: Transcription
# ---------------------------------------------------------------------------

class WordToken(BaseModel):
    text: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0


class TranscriptSegment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    speaker: Optional[str] = None      # "SPEAKER_00", "SPEAKER_01", …
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    words: list[WordToken] = Field(default_factory=list)
    mode: RecordingMode = RecordingMode.AMBIENT
    source: str = "asr"   # asr | typed_addendum


class UnifiedTranscript(BaseModel):
    segments: list[TranscriptSegment] = Field(default_factory=list)
    engine_used: str = ""
    diarization_engine: str = ""
    audio_duration_ms: int = 0
    full_text: str = ""      # Flat concatenation for prompt assembly


# ---------------------------------------------------------------------------
# Sub-models: Note generation
# ---------------------------------------------------------------------------

class CitationAnchor(BaseModel):
    transcript_segment_id: str
    audio_start_ms: int
    audio_end_ms: int
    confidence: float = 1.0


class NoteSection(BaseModel):
    type: str       # subjective | objective | assessment | plan | …
    content: str
    citations: list[CitationAnchor] = Field(default_factory=list)


class NoteMetadata(BaseModel):
    generated_at: Optional[datetime] = None
    llm_used: str = ""
    template_used: str = ""
    confidence_score: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ClinicalNote(BaseModel):
    note_type: NoteType = NoteType.SOAP
    sections: list[NoteSection] = Field(default_factory=list)
    metadata: NoteMetadata = Field(default_factory=NoteMetadata)

    def to_text(self) -> str:
        """Flat text rendering for display / clipboard."""
        parts = []
        for section in self.sections:
            parts.append(f"## {section.type.upper().replace('_', ' ')}\n{section.content}")
        return "\n\n".join(parts)


class ICD10Code(BaseModel):
    code: str
    description: str
    confidence: float = 0.0


class CPTCode(BaseModel):
    code: str
    description: str
    confidence: float = 0.0


class CodingSuggestion(BaseModel):
    em_level: Optional[str] = None        # E&M level (99213, 99214, …)
    em_rationale: Optional[str] = None
    icd10_codes: list[ICD10Code] = Field(default_factory=list)
    cpt_codes: list[CPTCode] = Field(default_factory=list)
    hcc_codes: list[str] = Field(default_factory=list)
    generated_at: Optional[datetime] = None


class PatientSummary(BaseModel):
    text: str
    reading_level: str = "5th_grade"
    generated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Sub-models: Review
# ---------------------------------------------------------------------------

class Correction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    section: str
    original: str
    corrected: str
    correction_type: CorrectionType
    timestamp: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Sub-models: Quality metrics
# ---------------------------------------------------------------------------

class EncounterMetrics(BaseModel):
    # Timing
    pipeline_start_ms: Optional[int] = None
    pipeline_end_ms: Optional[int] = None
    context_load_ms: Optional[int] = None
    asr_duration_ms: Optional[int] = None
    postprocessor_ms: Optional[int] = None
    note_gen_ms: Optional[int] = None

    # ASR quality
    asr_confidence: float = 0.0
    postprocessor_corrections: int = 0
    stutter_pairs_removed: int = 0

    # Note quality
    note_confidence: float = 0.0
    hallucination_flags: int = 0

    # Delivery
    delivery_success: bool = False

    # Per-node status tracking (set by each stub/real node)
    nodes_completed: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level EncounterState
# ---------------------------------------------------------------------------

class EncounterState(BaseModel):
    """
    The single state object that flows through the LangGraph encounter pipeline.

    LangGraph passes this to each node; nodes return a dict of fields to update.
    """

    # Identity
    encounter_id: str = Field(default_factory=lambda: str(uuid4()))
    provider_id: str
    patient_id: str
    status: EncounterStatus = EncounterStatus.CREATED

    # Provider profile (loaded at start, governs the whole pipeline)
    provider_profile: ProviderProfile

    # ── CONTEXT NODE output ───────────────────────────────────────────────
    context_packet: Optional[ContextPacket] = None

    # ── CAPTURE NODE output ───────────────────────────────────────────────
    audio_segments: list[AudioSegment] = Field(default_factory=list)
    audio_file_path: Optional[str] = None   # Convenience: path to a single uploaded file
    note_audio_file_path: Optional[str] = None  # Physician dictation audio (conversation mode)
    mode_events: list[ModeEvent] = Field(default_factory=list)
    voice_commands: list[VoiceCommand] = Field(default_factory=list)
    typed_addendums: list[Addendum] = Field(default_factory=list)
    recording_mode: RecordingMode = RecordingMode.DICTATION

    # ── TRANSCRIBE NODE output ────────────────────────────────────────────
    transcript: Optional[UnifiedTranscript] = None
    asr_engine_used: str = ""
    diarization_engine_used: str = ""
    postprocessor_version: str = ""
    postprocessor_metrics: dict[str, Any] = Field(default_factory=dict)

    # ── NOTE NODE output ──────────────────────────────────────────────────
    generated_note: Optional[ClinicalNote] = None
    coding_suggestions: list[CodingSuggestion] = Field(default_factory=list)
    patient_summary: Optional[PatientSummary] = None
    llm_engine_used: str = ""
    template_used: str = ""

    # ── REVIEW NODE output ────────────────────────────────────────────────
    corrections: list[Correction] = Field(default_factory=list)
    final_note: Optional[ClinicalNote] = None
    review_approved: bool = False

    # ── DELIVERY NODE output ──────────────────────────────────────────────
    delivery_method: DeliveryMethod = DeliveryMethod.CLIPBOARD
    delivery_result: Optional[dict[str, Any]] = None

    # ── Quality metrics (accumulated across all nodes) ────────────────────
    metrics: EncounterMetrics = Field(default_factory=EncounterMetrics)

    # ── Error tracking ────────────────────────────────────────────────────
    errors: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
