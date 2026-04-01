"""
api/routes/specialties.py — CRUD for specialties and their keyword dictionaries.

Specialties are defined by their dictionary files in config/dictionaries/{specialty}.txt.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/specialties", tags=["specialties"])

DICT_DIR = Path("config/dictionaries")


class SpecialtySummary(BaseModel):
    id: str
    name: str  # display name (capitalized)
    term_count: int
    has_dictionary: bool


class SpecialtyDetail(BaseModel):
    id: str
    name: str
    term_count: int
    terms: list[str]


class SpecialtyCreate(BaseModel):
    id: str  # e.g. "dermatology"
    terms: list[str] = []


class SpecialtyUpdate(BaseModel):
    terms: list[str]


def _load_terms(path: Path) -> list[str]:
    """Read a dictionary file, stripping comments and blanks."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _write_terms(path: Path, terms: list[str], specialty_id: str) -> None:
    header = f"# {specialty_id.replace('_', ' ').title()} specialty vocabulary\n"
    content = header + "\n".join(terms) + "\n"
    path.write_text(content, encoding="utf-8")


def _discover_specialties() -> list[dict]:
    """Discover all specialties from dictionary files, plus built-in 'general'."""
    specs = []
    if DICT_DIR.exists():
        for f in sorted(DICT_DIR.glob("*.txt")):
            if f.name == "base_medical.txt":
                continue
            sid = f.stem
            terms = _load_terms(f)
            specs.append({
                "id": sid,
                "name": sid.replace("_", " ").title(),
                "term_count": len(terms),
                "has_dictionary": True,
            })
    # Always include 'general' — used by soap_default template, no dictionary file needed
    if not any(s["id"] == "general" for s in specs):
        specs.insert(0, {"id": "general", "name": "General", "term_count": 0, "has_dictionary": False})
    return specs


@router.get("", response_model=list[SpecialtySummary])
def list_specialties():
    return [SpecialtySummary(**s) for s in _discover_specialties()]


@router.get("/{specialty_id}", response_model=SpecialtyDetail)
def get_specialty(specialty_id: str):
    path = DICT_DIR / f"{specialty_id}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Specialty '{specialty_id}' not found")
    terms = _load_terms(path)
    return SpecialtyDetail(
        id=specialty_id,
        name=specialty_id.replace("_", " ").title(),
        term_count=len(terms),
        terms=terms,
    )


@router.post("", response_model=SpecialtySummary, status_code=201)
def create_specialty(body: SpecialtyCreate):
    sid = body.id.lower().strip().replace(" ", "_")
    path = DICT_DIR / f"{sid}.txt"
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Specialty '{sid}' already exists")
    DICT_DIR.mkdir(parents=True, exist_ok=True)
    _write_terms(path, body.terms, sid)
    return SpecialtySummary(
        id=sid,
        name=sid.replace("_", " ").title(),
        term_count=len(body.terms),
        has_dictionary=True,
    )


@router.get("/audit/consistency")
def audit_consistency():
    """Check all specialties, templates, and providers for data consistency issues."""
    issues: list[dict] = []

    # Collect known specialties (those with dictionary files) + "general" as built-in fallback
    known_specialties = {f.stem for f in DICT_DIR.glob("*.txt") if f.name != "base_medical.txt"} if DICT_DIR.exists() else set()
    known_specialties.add("general")  # soap_default uses "general" — no dictionary needed

    # Check templates
    from api.routes.templates import TEMPLATE_DIR as template_dir, PROVIDER_DIR as provider_dir
    if template_dir.exists():
        for tf in template_dir.glob("*.yaml"):
            with open(tf) as f:
                import yaml
                t = yaml.safe_load(f) or {}
            spec = t.get("specialty", "")
            if spec and spec not in known_specialties:
                issues.append({
                    "type": "template",
                    "id": tf.stem,
                    "severity": "error",
                    "message": f"Template '{tf.stem}' references specialty '{spec}' which has no dictionary file",
                })

    # Check providers
    if provider_dir.exists():
        for pf in provider_dir.glob("*.yaml"):
            with open(pf) as f:
                import yaml
                p = yaml.safe_load(f) or {}
            pid = p.get("id", pf.stem)
            spec = p.get("specialty", "")
            if spec and spec not in known_specialties:
                issues.append({
                    "type": "provider",
                    "id": pid,
                    "severity": "error",
                    "message": f"Provider '{pid}' has specialty '{spec}' with no dictionary file",
                })
            routing = p.get("template_routing", {})
            for vt, tpl_id in routing.items():
                tpl_path = template_dir / f"{tpl_id}.yaml"
                if not tpl_path.exists():
                    issues.append({
                        "type": "provider",
                        "id": pid,
                        "severity": "error",
                        "message": f"Provider '{pid}' routes visit_type '{vt}' to template '{tpl_id}' which does not exist",
                    })
                else:
                    with open(tpl_path) as f:
                        tpl = yaml.safe_load(f) or {}
                    tpl_spec = tpl.get("specialty", "")
                    if spec and tpl_spec and spec != tpl_spec:
                        issues.append({
                            "type": "provider",
                            "id": pid,
                            "severity": "warning",
                            "message": f"Provider '{pid}' ({spec}) routes to template '{tpl_id}' ({tpl_spec}) — specialty mismatch",
                        })

    return {
        "total_issues": len(issues),
        "errors": len([i for i in issues if i["severity"] == "error"]),
        "warnings": len([i for i in issues if i["severity"] == "warning"]),
        "issues": issues,
    }


@router.put("/{specialty_id}/dictionary", response_model=SpecialtyDetail)
def update_dictionary(specialty_id: str, body: SpecialtyUpdate):
    path = DICT_DIR / f"{specialty_id}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Specialty '{specialty_id}' not found")
    _write_terms(path, body.terms, specialty_id)
    return SpecialtyDetail(
        id=specialty_id,
        name=specialty_id.replace("_", " ").title(),
        term_count=len(body.terms),
        terms=body.terms,
    )
