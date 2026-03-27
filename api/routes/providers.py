"""
api/routes/providers.py — CRUD for provider profiles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import data_loader as dl
from api.models import ProviderSummary

router = APIRouter(prefix="/providers", tags=["providers"])

PROVIDER_DIR = Path("config/providers")


class ProviderCreate(BaseModel):
    id: str
    name: str
    credentials: str = ""
    specialty: str = ""
    practice_id: str = ""
    note_format: str = "SOAP"
    noise_suppression_level: str = "moderate"
    postprocessor_mode: str = "hybrid"
    style_directives: list[str] = []
    custom_vocabulary: list[str] = []
    template_routing: dict[str, str] = {}


class ProviderUpdate(BaseModel):
    name: Optional[str] = None
    credentials: Optional[str] = None
    specialty: Optional[str] = None
    practice_id: Optional[str] = None
    note_format: Optional[str] = None
    noise_suppression_level: Optional[str] = None
    postprocessor_mode: Optional[str] = None
    style_directives: Optional[list[str]] = None
    custom_vocabulary: Optional[list[str]] = None
    template_routing: Optional[dict[str, str]] = None


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


@router.get("", response_model=list[ProviderSummary])
def list_providers():
    providers = dl.list_providers()
    return [ProviderSummary(**p) for p in providers]


@router.get("/{provider_id}")
def get_provider(provider_id: str):
    provider = dl.get_provider(provider_id)
    if not provider:
        all_providers = dl.list_providers()
        match = next((p for p in all_providers if p["id"] == provider_id), None)
        if match:
            return match
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")
    return provider


@router.post("", response_model=ProviderSummary, status_code=201)
def create_provider(body: ProviderCreate):
    pid = body.id.lower().strip().replace(" ", "_")
    path = PROVIDER_DIR / f"{pid}.yaml"
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Provider '{pid}' already exists")

    # Validate specialty has a dictionary
    spec_dir = Path("config/dictionaries")
    if body.specialty:
        spec_path = spec_dir / f"{body.specialty}.txt"
        if not spec_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Specialty '{body.specialty}' has no dictionary. Create the specialty first.",
            )

    # Validate template routing references valid templates
    template_dir = Path("config/templates")
    for visit_type, tpl_id in body.template_routing.items():
        tpl_path = template_dir / f"{tpl_id}.yaml"
        if not tpl_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Template '{tpl_id}' (for visit_type '{visit_type}') does not exist.",
            )

    data = {
        "id": pid,
        "name": body.name,
        "credentials": body.credentials,
        "specialty": body.specialty,
        "practice_id": body.practice_id,
        "note_format": body.note_format,
        "noise_suppression_level": body.noise_suppression_level,
        "postprocessor_mode": body.postprocessor_mode,
        "npi": "",
        "asr_override": None,
        "llm_override": None,
        "correction_count": 0,
        "style_model_version": "v0",
        "style_directives": body.style_directives,
        "custom_vocabulary": body.custom_vocabulary,
        "template_routing": body.template_routing,
        "quality_scores": {},
        "quality_history": [],
    }
    PROVIDER_DIR.mkdir(parents=True, exist_ok=True)
    _save_yaml(path, data)

    return ProviderSummary(
        id=pid,
        name=body.name,
        credentials=body.credentials,
        specialty=body.specialty,
        latest_score=None,
        quality_scores={},
    )


@router.put("/{provider_id}", response_model=ProviderSummary)
def update_provider(provider_id: str, body: ProviderUpdate):
    path = PROVIDER_DIR / f"{provider_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")

    data = _load_yaml(path)

    if body.name is not None:
        data["name"] = body.name
    if body.credentials is not None:
        data["credentials"] = body.credentials
    if body.specialty is not None:
        spec_dir = Path("config/dictionaries")
        spec_path = spec_dir / f"{body.specialty}.txt"
        if not spec_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Specialty '{body.specialty}' has no dictionary.",
            )
        data["specialty"] = body.specialty
    if body.practice_id is not None:
        data["practice_id"] = body.practice_id
    if body.note_format is not None:
        data["note_format"] = body.note_format
    if body.noise_suppression_level is not None:
        data["noise_suppression_level"] = body.noise_suppression_level
    if body.postprocessor_mode is not None:
        data["postprocessor_mode"] = body.postprocessor_mode
    if body.style_directives is not None:
        data["style_directives"] = body.style_directives
    if body.custom_vocabulary is not None:
        data["custom_vocabulary"] = body.custom_vocabulary
    if body.template_routing is not None:
        template_dir = Path("config/templates")
        for visit_type, tpl_id in body.template_routing.items():
            tpl_path = template_dir / f"{tpl_id}.yaml"
            if not tpl_path.exists():
                raise HTTPException(
                    status_code=422,
                    detail=f"Template '{tpl_id}' (for '{visit_type}') does not exist.",
                )
        data["template_routing"] = body.template_routing

    _save_yaml(path, data)

    scores = data.get("quality_scores", {})
    latest = max(scores.values()) if scores else None

    return ProviderSummary(
        id=provider_id,
        name=data.get("name"),
        credentials=data.get("credentials"),
        specialty=data.get("specialty"),
        latest_score=latest,
        quality_scores=scores,
    )


@router.get("/{provider_id}/quality-trend")
def get_quality_trend(provider_id: str):
    provider = dl.get_provider(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")

    history = provider.get("quality_history", [])
    scores = provider.get("quality_scores", {})

    trend = []
    for v in sorted(scores.keys(), key=lambda x: x.lstrip("v").zfill(3)):
        entry = next((h for h in history if h.get("version") == v), None)
        trend.append({
            "version": v,
            "score": scores[v],
            "date": entry.get("date") if entry else None,
            "samples": entry.get("samples") if entry else None,
        })

    return {"provider_id": provider_id, "trend": trend}
