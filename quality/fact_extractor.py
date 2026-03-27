"""
Fact extractor — pulls structured clinical facts from note text using the LLM.

Extracts: medications (name+dose+freq), diagnoses, exam findings, plan items.
Used to compute precision/recall between generated and gold notes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def _repair_json(raw: str) -> dict:
    """Attempt to parse JSON, applying incremental repairs on failure.

    Handles common LLM output issues: unterminated strings, trailing commas,
    missing closing brackets, and truncated output.
    """
    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    text = raw

    # Strategy 2: strip trailing incomplete entries and fix structure
    # Remove trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix unterminated strings: find last unterminated quote and close it
    # Count quotes — if odd number, append a closing quote
    if text.count('"') % 2 == 1:
        text = text.rstrip()
        # If we're mid-string, close it
        if not text.endswith('"'):
            text += '"'

    # Ensure arrays are closed — count [ vs ]
    open_brackets = text.count("[") - text.count("]")
    open_braces = text.count("{") - text.count("}")
    text = text.rstrip().rstrip(",")
    text += "]" * max(0, open_brackets)
    text += "}" * max(0, open_braces)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 3: truncate to last valid array entry
    # Find the last complete "..." entry and cut there
    last_good = text.rfind('",')
    if last_good > 0:
        truncated = text[: last_good + 1]
        open_brackets = truncated.count("[") - truncated.count("]")
        open_braces = truncated.count("{") - truncated.count("}")
        truncated += "]" * max(0, open_brackets)
        truncated += "}" * max(0, open_braces)
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    # Strategy 4: regex extraction — pull out individual arrays
    result: dict[str, list[str]] = {}
    for key in ("medications", "diagnoses", "exam_findings", "plan_items"):
        pattern = rf'"{key}"\s*:\s*\[(.*?)\]'
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            items = re.findall(r'"([^"]*)"', m.group(1))
            result[key] = items
    if result:
        return result

    # All strategies failed — return empty
    logger.warning("fact_extractor: all JSON repair strategies failed")
    return {}


@dataclass
class ExtractedFacts:
    medications: list[str] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    exam_findings: list[str] = field(default_factory=list)
    plan_items: list[str] = field(default_factory=list)


@dataclass
class FactCheckResult:
    medications: tuple[int, int]    # (found, total)
    diagnoses: tuple[int, int]
    exam_findings: tuple[int, int]
    plan_items: tuple[int, int]
    missed_medications: list[str] = field(default_factory=list)
    missed_diagnoses: list[str] = field(default_factory=list)
    missed_findings: list[str] = field(default_factory=list)
    missed_plan_items: list[str] = field(default_factory=list)

    def precision(self, category: str) -> float:
        found, total = getattr(self, category)
        return round(found / total, 2) if total > 0 else 1.0

    def summary(self) -> dict[str, str]:
        cats = ["medications", "diagnoses", "exam_findings", "plan_items"]
        return {c: f"{getattr(self, c)[0]}/{getattr(self, c)[1]}" for c in cats}


def _extract_facts_with_llm(note_text: str, engine: Any) -> ExtractedFacts:
    """Ask the LLM to extract structured facts from a clinical note."""
    from mcp_servers.llm.base import LLMConfig, LLMMessage

    system = (
        "You are a medical information extractor. Extract structured facts from the "
        "clinical note provided. Return ONLY valid JSON, no commentary."
    )
    prompt = f"""Extract clinical facts from this note.

NOTE:
{note_text[:3000]}

Return valid JSON only:
{{
  "medications": ["medication name dose frequency", ...],
  "diagnoses": ["diagnosis description", ...],
  "exam_findings": ["specific finding", ...],
  "plan_items": ["plan item", ...]
}}

Be specific. For medications include name, dose, frequency when mentioned.
For diagnoses include laterality and etiology if present.
For exam findings include specific measurements (degrees, grades) if mentioned."""

    cfg = LLMConfig(model=None, temperature=0.0, max_tokens=600)
    try:
        response = engine.generate_sync(
            system_prompt=system,
            messages=[LLMMessage(role="user", content=prompt)],
            config=cfg,
            task="command_parse",
        )
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = _repair_json(raw)
        return ExtractedFacts(
            medications=[str(m).lower() for m in data.get("medications", [])],
            diagnoses=[str(d).lower() for d in data.get("diagnoses", [])],
            exam_findings=[str(f).lower() for f in data.get("exam_findings", [])],
            plan_items=[str(p).lower() for p in data.get("plan_items", [])],
        )
    except Exception as exc:
        logger.warning("fact_extractor: extraction failed — %s", exc)
        return ExtractedFacts()


def _fuzzy_match(gen_items: list[str], gold_items: list[str], threshold: int = 3) -> tuple[int, list[str]]:
    """
    Count how many gold items appear (approximately) in generated items.

    Uses substring matching: a gold item is 'found' if any key word from it
    appears in any generated item.
    """
    found = 0
    missed = []
    for gold_item in gold_items:
        # Extract significant words (len >= threshold)
        gold_words = {w for w in gold_item.split() if len(w) >= threshold}
        gen_text = " ".join(gen_items)
        if gold_words and any(w in gen_text for w in gold_words):
            found += 1
        else:
            missed.append(gold_item)
    return found, missed


def compare_facts(generated: ExtractedFacts, gold: ExtractedFacts) -> FactCheckResult:
    """Compare extracted facts between generated and gold notes."""
    med_found, med_missed = _fuzzy_match(generated.medications, gold.medications)
    dx_found, dx_missed = _fuzzy_match(generated.diagnoses, gold.diagnoses)
    exam_found, exam_missed = _fuzzy_match(generated.exam_findings, gold.exam_findings)
    plan_found, plan_missed = _fuzzy_match(generated.plan_items, gold.plan_items)

    return FactCheckResult(
        medications=(med_found, len(gold.medications)),
        diagnoses=(dx_found, len(gold.diagnoses)),
        exam_findings=(exam_found, len(gold.exam_findings)),
        plan_items=(plan_found, len(gold.plan_items)),
        missed_medications=med_missed,
        missed_diagnoses=dx_missed,
        missed_findings=exam_missed,
        missed_plan_items=plan_missed,
    )


def extract_and_compare(generated_note: str, gold_note: str, engine: Any) -> FactCheckResult:
    """Full pipeline: extract facts from both notes, compare, return result."""
    gen_facts = _extract_facts_with_llm(generated_note, engine)
    gold_facts = _extract_facts_with_llm(gold_note, engine)
    return compare_facts(gen_facts, gold_facts)
