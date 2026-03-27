#!/usr/bin/env python3
"""
Extract patient demographics from gold-standard notes → patient_context.yaml.

Parses the structured header fields (FIRST NAME, LAST NAME, DOB, etc.) present
in all gold notes.  Also infers sex and visit type from note body text.

Works with both formats:
  - Dictation gold notes (plain text headers):  FIRST NAME:  RILEY
  - Conversation gold notes (markdown bold):    **FIRST NAME:** RILEY

Usage:
    python scripts/extract_patient_context.py                  # all samples
    python scripts/extract_patient_context.py --data-dir data/dictation
    python scripts/extract_patient_context.py --sample data/dictation/226680
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import DATA_DIR as _DATA_ROOT
_MODES = ("conversation", "dictation")

# ─── Header field patterns ──────────────────────────────────────────────────
# Handles both plain text ("FIRST NAME:  VALUE") and markdown ("**FIRST NAME:** VALUE")
_HEADER_RE = re.compile(
    r"^\s*\*{0,2}\s*"                    # optional markdown bold
    r"(?P<key>[A-Za-z][A-Za-z /]+?)"     # field name (mixed case OK)
    r"\s*\*{0,2}\s*:\s*"                 # colon separator
    r"(?P<value>.+?)\s*$",               # value
    re.MULTILINE,
)

# ─── Body text patterns ────────────────────────────────────────────────────
_SEX_RE = re.compile(
    r"\b(?:(?:he|she)\s+is\s+(?:a|an)\s+.*?\b(male|female))|"
    r"\b(\d+[- ]year[- ]old\s+(male|female))",
    re.IGNORECASE,
)

_VISIT_TYPE_RE = re.compile(
    r"(?:ASSUME\s+CARE\s+EVALUATION|INITIAL\s+EVALUATION|"
    r"RE-?EVALUATION|FOLLOW[- ]?UP\s+(?:EVALUATION|VISIT|EXAM)|"
    r"PROGRESS\s+NOTE)",
    re.IGNORECASE,
)

_VISIT_TYPE_MAP = {
    "assume care evaluation": "assume_care_evaluation",
    "initial evaluation": "initial_evaluation",
    "re-evaluation": "re_evaluation",
    "reevaluation": "re_evaluation",
    "follow-up evaluation": "follow_up",
    "follow-up visit": "follow_up",
    "follow-up exam": "follow_up",
    "follow up evaluation": "follow_up",
    "followup evaluation": "follow_up",
    "progress note": "progress_note",
}


def _extract_headers(text: str) -> dict[str, str]:
    """Extract key-value pairs from the structured header block."""
    headers: dict[str, str] = {}
    for m in _HEADER_RE.finditer(text[:2000]):  # headers are always at the top
        key = m.group("key").strip().replace("*", "").strip().upper()
        value = m.group("value").strip().replace("**", "").strip()
        if value and value not in ("", "N/A"):
            headers[key] = value
    return headers


def _normalize_date(raw: str) -> str:
    """Normalize date strings to YYYY-MM-DD."""
    raw = raw.strip()
    # Already ISO: 2025-12-18 or 2025-12-18 00:00:00
    if re.match(r"\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    # MM/DD/YYYY
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return raw


def _infer_sex(text: str) -> str:
    """Infer patient sex from note body text."""
    m = _SEX_RE.search(text)
    if m:
        for g in m.groups():
            if g and g.lower() in ("male", "female"):
                return g.capitalize()
    # Pronoun counting fallback
    he_count = len(re.findall(r"\bhe\b", text, re.IGNORECASE))
    she_count = len(re.findall(r"\bshe\b", text, re.IGNORECASE))
    if he_count > she_count * 2:
        return "Male"
    if she_count > he_count * 2:
        return "Female"
    return ""


def _infer_visit_type(text: str) -> str:
    """Infer visit type from note body text."""
    m = _VISIT_TYPE_RE.search(text)
    if m:
        matched = m.group(0).lower().strip()
        return _VISIT_TYPE_MAP.get(matched, matched.replace(" ", "_"))
    return "initial_evaluation"


def extract_context(gold_note_path: Path) -> dict:
    """Extract structured patient context from a gold-standard note."""
    text = gold_note_path.read_text(errors="replace")
    headers = _extract_headers(text)

    first_name = headers.get("FIRST NAME", "")
    last_name = headers.get("LAST NAME", "")
    patient_name = f"{first_name} {last_name}".strip() or None

    dob_raw = headers.get("DATE OF BIRTH", headers.get("DOB", ""))
    dob = _normalize_date(dob_raw) if dob_raw else None

    date_of_exam_raw = headers.get("DATE OF EXAM", "")
    date_of_exam = _normalize_date(date_of_exam_raw) if date_of_exam_raw else None

    date_of_accident_raw = headers.get("D/ACCIDENT", headers.get("DATE OF ACCIDENT", ""))
    date_of_accident = _normalize_date(date_of_accident_raw) if date_of_accident_raw else None

    record_number = headers.get("RECORD NUMBER", "")
    case_number = headers.get("CASE NUMBER", "")
    place_of_exam = headers.get("PLACE OF EXAM", "")

    provider_first = headers.get("PROVIDER FIRST", "")
    provider_last = headers.get("PROVIDER LAST", "")
    provider_name = f"Dr. {provider_first.title()} {provider_last.title()}".strip()
    if provider_name == "Dr.":
        provider_name = ""

    supervising = headers.get("SUPERVISING PHYSICIAN", "")

    sex = _infer_sex(text)
    visit_type = _infer_visit_type(text)

    context = {
        "patient": {
            "name": patient_name,
            "date_of_birth": dob,
            "sex": sex or None,
            "mrn": record_number or None,
        },
        "encounter": {
            "date_of_service": date_of_exam,
            "visit_type": visit_type,
            "date_of_injury": date_of_accident,
            "case_number": case_number or None,
        },
        "provider": {
            "name": provider_name or None,
            "credentials": "MD",
            "specialty": "Orthopedic",
        },
        "facility": {
            "name": "Excelsia Injury Care",
            "location": place_of_exam or None,
        },
    }

    if supervising and supervising.strip("* "):
        context["supervising_physician"] = {"name": supervising.strip("* ")}

    return context


def process_sample(sample_dir: Path) -> Path | None:
    """Extract context from a sample directory and write patient_context.yaml."""
    # Find gold note (new format first, then legacy)
    gold_path = sample_dir / "final_soap_note.md"
    if not gold_path.exists():
        for gold_name in ("soap_final.md", "soap_initial.md"):
            gold_path = sample_dir / gold_name
            if gold_path.exists():
                break
        else:
            return None

    context = extract_context(gold_path)
    out_path = sample_dir / "patient_context.yaml"
    with open(out_path, "w") as f:
        yaml.dump(context, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract patient context from gold notes")
    parser.add_argument("--data-dir", default=None,
                        help="Data root (default: ai-scribe-data)")
    parser.add_argument("--sample", type=str, default=None,
                        help="Single sample directory to process")
    args = parser.parse_args()

    if args.sample:
        sample_dir = Path(args.sample)
        result = process_sample(sample_dir)
        if result:
            print(f"Wrote: {result}")
        else:
            print(f"No gold note found in {sample_dir}")
        return

    data_root = Path(args.data_dir) if args.data_dir else _DATA_ROOT
    total = 0
    for mode in _MODES:
        mode_dir = data_root / mode
        if not mode_dir.exists():
            continue
        for physician_dir in sorted(mode_dir.iterdir()):
            if not physician_dir.is_dir():
                continue
            for encounter_dir in sorted(physician_dir.iterdir()):
                if not encounter_dir.is_dir():
                    continue
                result = process_sample(encounter_dir)
                if result:
                    total += 1
                    print(f"  {result}")

    print(f"\nExtracted context for {total} samples")


if __name__ == "__main__":
    main()
