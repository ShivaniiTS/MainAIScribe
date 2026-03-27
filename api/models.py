"""
api/models.py — Pydantic response models for the API.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class SampleSummary(BaseModel):
    sample_id: str
    mode: str  # "dictation" | "ambient"
    physician: str = ""
    versions: list[str]
    latest_version: Optional[str]
    has_gold: bool
    quality: Optional[dict]
    is_test: bool = False


class SampleDetail(BaseModel):
    sample_id: str
    mode: str
    physician: str = ""
    versions: list[str]
    latest_version: Optional[str]
    has_gold: bool
    quality: Optional[dict]
    patient_context: Optional[dict]
    is_test: bool = False


class NoteContent(BaseModel):
    sample_id: str
    version: str
    content: str  # Markdown


class ComparisonContent(BaseModel):
    sample_id: str
    version: str
    content: str  # Markdown


class GoldNoteContent(BaseModel):
    sample_id: str
    content: str  # Markdown


class QualityScore(BaseModel):
    sample_id: str
    mode: str
    version: str
    overall: Optional[float]
    accuracy: Optional[float]
    completeness: Optional[float]
    no_hallucination: Optional[float]
    structure: Optional[float]
    language: Optional[float]
    overlap: Optional[str]


class AggregateQuality(BaseModel):
    version: str
    sample_count: int
    average: float
    min: float
    max: float
    dimensions: dict


class QualityTrend(BaseModel):
    trend: list[AggregateQuality]


class ProviderSummary(BaseModel):
    id: str
    name: Optional[str]
    credentials: Optional[str]
    specialty: Optional[str]
    latest_score: Optional[float]
    quality_scores: dict


class EncounterCreateRequest(BaseModel):
    provider_id: str
    patient_id: str
    visit_type: str = "follow_up"
    mode: str = "dictation"  # "dictation" | "ambient"


class EncounterResponse(BaseModel):
    encounter_id: str
    status: str  # "pending" | "processing" | "complete" | "error"
    provider_id: str
    patient_id: str
    visit_type: str
    mode: str
    message: Optional[str] = None
