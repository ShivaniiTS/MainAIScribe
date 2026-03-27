"""
api/routes/templates.py — CRUD for note templates.

Templates are YAML files in config/templates/{template_id}.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/templates", tags=["templates"])

TEMPLATE_DIR = Path("config/templates")
PROVIDER_DIR = Path("config/providers")


class TemplateSectionInput(BaseModel):
    id: str
    label: str
    required: bool = True
    prompt_hint: str = ""


class TemplateFormattingInput(BaseModel):
    voice: str = "active"
    tense: str = "past"
    person: str = "third"
    abbreviations: str = "spell_out"
    measurements: str = "include_units"


class TemplateSummary(BaseModel):
    id: str
    name: str
    specialty: str
    visit_type: str
    section_count: int
    providers: list[str]  # provider IDs that route to this template


class TemplateDetail(BaseModel):
    id: str
    name: str
    specialty: str
    visit_type: str
    header_fields: list[str]
    sections: list[dict]
    formatting: dict
    providers: list[str]


class TemplateCreate(BaseModel):
    id: str
    name: str
    specialty: str
    visit_type: str
    header_fields: list[str] = []
    sections: list[TemplateSectionInput] = []
    formatting: Optional[TemplateFormattingInput] = None


class TemplateUpdate(BaseModel):
    name: Optional[str] = None
    specialty: Optional[str] = None
    visit_type: Optional[str] = None
    header_fields: Optional[list[str]] = None
    sections: Optional[list[TemplateSectionInput]] = None
    formatting: Optional[TemplateFormattingInput] = None


def _load_template(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_template(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _provider_template_map() -> dict[str, list[str]]:
    """Build mapping: template_id -> [provider_ids] from provider YAML files."""
    mapping: dict[str, list[str]] = {}
    if PROVIDER_DIR.exists():
        for pf in PROVIDER_DIR.glob("*.yaml"):
            data = _load_template(pf)
            pid = data.get("id", pf.stem)
            routing = data.get("template_routing", {})
            for tpl_id in set(routing.values()):
                mapping.setdefault(tpl_id, []).append(pid)
    return mapping


def _list_templates() -> list[dict]:
    provider_map = _provider_template_map()
    templates = []
    if TEMPLATE_DIR.exists():
        for f in sorted(TEMPLATE_DIR.glob("*.yaml")):
            t = _load_template(f)
            tid = f.stem
            templates.append({
                "id": tid,
                "name": t.get("name", tid),
                "specialty": t.get("specialty", ""),
                "visit_type": t.get("visit_type", ""),
                "section_count": len(t.get("sections", [])),
                "providers": provider_map.get(tid, []),
            })
    return templates


@router.get("", response_model=list[TemplateSummary])
def list_templates():
    return [TemplateSummary(**t) for t in _list_templates()]


@router.get("/{template_id}", response_model=TemplateDetail)
def get_template(template_id: str):
    path = TEMPLATE_DIR / f"{template_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
    t = _load_template(path)
    provider_map = _provider_template_map()
    return TemplateDetail(
        id=template_id,
        name=t.get("name", template_id),
        specialty=t.get("specialty", ""),
        visit_type=t.get("visit_type", ""),
        header_fields=t.get("header_fields", []),
        sections=t.get("sections", []),
        formatting=t.get("formatting", {}),
        providers=provider_map.get(template_id, []),
    )


@router.post("", response_model=TemplateSummary, status_code=201)
def create_template(body: TemplateCreate):
    tid = body.id.lower().strip().replace(" ", "_")
    path = TEMPLATE_DIR / f"{tid}.yaml"
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Template '{tid}' already exists")

    # Validate specialty has a dictionary
    from api.routes.specialties import DICT_DIR as SPEC_DIR
    spec_path = SPEC_DIR / f"{body.specialty}.txt"
    if body.specialty and not spec_path.exists():
        raise HTTPException(
            status_code=422,
            detail=f"Specialty '{body.specialty}' has no dictionary file. Create the specialty first.",
        )

    data: dict[str, Any] = {
        "name": body.name,
        "specialty": body.specialty,
        "visit_type": body.visit_type,
        "header_fields": body.header_fields,
        "sections": [s.model_dump() for s in body.sections],
        "formatting": (body.formatting.model_dump() if body.formatting else {
            "voice": "active",
            "tense": "past",
            "person": "third",
            "abbreviations": "spell_out",
            "measurements": "include_units",
        }),
    }
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    _save_template(path, data)

    return TemplateSummary(
        id=tid,
        name=body.name,
        specialty=body.specialty,
        visit_type=body.visit_type,
        section_count=len(body.sections),
        providers=[],
    )


@router.put("/{template_id}", response_model=TemplateDetail)
def update_template(template_id: str, body: TemplateUpdate):
    path = TEMPLATE_DIR / f"{template_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    t = _load_template(path)

    if body.name is not None:
        t["name"] = body.name
    if body.specialty is not None:
        t["specialty"] = body.specialty
    if body.visit_type is not None:
        t["visit_type"] = body.visit_type
    if body.header_fields is not None:
        t["header_fields"] = body.header_fields
    if body.sections is not None:
        t["sections"] = [s.model_dump() for s in body.sections]
    if body.formatting is not None:
        t["formatting"] = body.formatting.model_dump()

    _save_template(path, t)
    provider_map = _provider_template_map()

    return TemplateDetail(
        id=template_id,
        name=t.get("name", template_id),
        specialty=t.get("specialty", ""),
        visit_type=t.get("visit_type", ""),
        header_fields=t.get("header_fields", []),
        sections=t.get("sections", []),
        formatting=t.get("formatting", {}),
        providers=provider_map.get(template_id, []),
    )


@router.delete("/{template_id}", status_code=204)
def delete_template(template_id: str):
    path = TEMPLATE_DIR / f"{template_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    # Check if any provider routes to this template
    provider_map = _provider_template_map()
    users = provider_map.get(template_id, [])
    if users:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete: template is used by providers: {', '.join(users)}",
        )
    path.unlink()
