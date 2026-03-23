"""
Quality evaluator — LLM-as-judge scoring for generated clinical notes.

Scores each generated note against its gold standard on 6 dimensions (1-5 each),
plus fact extraction precision/recall.

Usage:
    evaluator = QualityEvaluator(engine)
    result = evaluator.evaluate(
        sample_id="224889",
        generated_note="...",
        gold_note="...",
        transcript="...",
    )
    print(result.overall_score)   # 0.0 – 5.0
    print(result.to_markdown())
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from quality.dimensions import DIMENSIONS, weighted_score
from quality.fact_extractor import FactCheckResult, extract_and_compare

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    sample_id: str
    version: str
    overall_score: float                      # 0–5 weighted
    dimensions: dict[str, float]              # dimension_id → 1–5 score
    fact_check: Optional[FactCheckResult]
    keyword_overlap: float                    # simple fast metric (0–1)
    sections_present: list[str]
    sections_missing: list[str]
    llm_rationale: str                        # LLM judge's reasoning
    elapsed_s: float = 0.0
    has_gold: bool = True

    def to_dict(self) -> dict:
        fc = self.fact_check
        return {
            "sample_id": self.sample_id,
            "version": self.version,
            "overall_score": self.overall_score,
            **{f"dim_{k}": v for k, v in self.dimensions.items()},
            "keyword_overlap": round(self.keyword_overlap, 3),
            "sections_present": len(self.sections_present),
            "sections_missing": len(self.sections_missing),
            "meds_found": f"{fc.medications[0]}/{fc.medications[1]}" if fc else "—",
            "dx_found": f"{fc.diagnoses[0]}/{fc.diagnoses[1]}" if fc else "—",
            "exam_found": f"{fc.exam_findings[0]}/{fc.exam_findings[1]}" if fc else "—",
            "plan_found": f"{fc.plan_items[0]}/{fc.plan_items[1]}" if fc else "—",
            "elapsed_s": round(self.elapsed_s, 1),
        }


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
    "for", "she", "her", "he", "his", "with", "has", "have", "be", "that",
    "this", "from", "as", "at", "by", "on", "not", "no", "but", "it", "its",
    "will", "may", "can", "we", "our", "patient", "presents", "today", "due",
}


def _keywords(text: str, min_len: int = 4) -> set[str]:
    return {
        w.lower()
        for w in re.findall(r"[a-z]+", text.lower())
        if len(w) >= min_len and w not in _STOPWORDS
    }


def _keyword_overlap(generated: str, reference: str, top_n: int = 50) -> float:
    ref_kw = _keywords(reference)
    gen_kw = _keywords(generated)
    if not ref_kw:
        return 0.0
    important = sorted(ref_kw, key=len, reverse=True)[:top_n]
    matched = sum(1 for w in important if w in gen_kw)
    return matched / len(important)


def _extract_sections(text: str) -> list[str]:
    """Find section headers in a clinical note."""
    pattern = re.compile(
        r"^(?:\*{1,2}|#{1,3}\s*)?"
        r"(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN|ASSESSMENT AND PLAN|"
        r"CHIEF COMPLAINT|HISTORY OF PRESENT ILLNESS|HPI|"
        r"PAST MEDICAL HISTORY|PMH|PAST SURGICAL HISTORY|PSH|"
        r"CURRENT MEDICATIONS|MEDICATIONS|ALLERGIES|"
        r"FAMILY HISTORY|SOCIAL HISTORY|REVIEW OF SYSTEMS|ROS|"
        r"PHYSICAL EXAMINATION|PHYSICAL EXAM|EXAMINATION|"
        r"IMAGING|IMAGING / DIAGNOSTICS|INTERVAL HISTORY)"
        r"[\s:*#]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    return [m.group(1).upper().strip() for m in pattern.finditer(text)]


class QualityEvaluator:
    """
    LLM-as-judge quality evaluator for clinical notes.

    Args:
        engine:         LLM engine (must implement generate_sync)
        run_fact_check: Whether to run fact extraction (2 extra LLM calls per sample)
    """

    def __init__(self, engine: Any, run_fact_check: bool = True) -> None:
        self.engine = engine
        self.run_fact_check = run_fact_check

    def evaluate(
        self,
        sample_id: str,
        generated_note: str,
        gold_note: str,
        transcript: str = "",
        version: str = "v1",
    ) -> QualityResult:
        t0 = time.time()

        # Fast metrics
        kw_overlap = _keyword_overlap(generated_note, gold_note) if gold_note else 0.0
        gen_sections = _extract_sections(generated_note)
        gold_sections = _extract_sections(gold_note)
        sections_missing = [s for s in gold_sections if s not in gen_sections]

        # LLM judge
        dim_scores, rationale = self._llm_judge(generated_note, gold_note, transcript)
        overall = weighted_score(dim_scores)

        # Fact check
        fact_result = None
        if self.run_fact_check and gold_note:
            try:
                fact_result = extract_and_compare(generated_note, gold_note, self.engine)
            except Exception as exc:
                logger.warning("quality: fact extraction failed for %s — %s", sample_id, exc)

        return QualityResult(
            sample_id=sample_id,
            version=version,
            overall_score=overall,
            dimensions=dim_scores,
            fact_check=fact_result,
            keyword_overlap=kw_overlap,
            sections_present=gen_sections,
            sections_missing=sections_missing,
            llm_rationale=rationale,
            elapsed_s=time.time() - t0,
            has_gold=bool(gold_note),
        )

    def _llm_judge(
        self,
        generated: str,
        gold: str,
        transcript: str,
    ) -> tuple[dict[str, float], str]:
        """Ask the LLM to score the generated note on all dimensions."""
        from mcp_servers.llm.base import LLMConfig, LLMMessage

        rubric_lines = "\n".join(
            f"- {d.label} ({d.id}): {d.rubric}" for d in DIMENSIONS
        )

        gold_block = f"\nGOLD STANDARD NOTE:\n{gold[:2000]}\n" if gold else ""
        transcript_block = f"\nTRANSCRIPT (source):\n{transcript[:1500]}\n" if transcript else ""

        prompt = f"""You are evaluating a generated clinical note against a gold standard.

Score the GENERATED NOTE on each dimension from 1 to 5.

SCORING RUBRIC:
{rubric_lines}

GENERATED NOTE:
{generated[:2000]}
{gold_block}{transcript_block}
Return ONLY valid JSON:
{{
  "medical_accuracy": <1-5>,
  "completeness": <1-5>,
  "no_hallucination": <1-5>,
  "structure": <1-5>,
  "clinical_language": <1-5>,
  "readability": <1-5>,
  "rationale": "<2-3 sentences explaining the main strengths and weaknesses>"
}}"""

        cfg = LLMConfig(model=None, temperature=0.0, max_tokens=400)
        try:
            response = self.engine.generate_sync(
                system_prompt="You are a clinical note quality evaluator. Return only valid JSON.",
                messages=[LLMMessage(role="user", content=prompt)],
                config=cfg,
                task="command_parse",
            )
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
            scores = {
                d.id: float(data.get(d.id, 3.0))
                for d in DIMENSIONS
            }
            rationale = data.get("rationale", "")
            return scores, rationale
        except Exception as exc:
            logger.warning("quality: LLM judge failed — %s", exc)
            return {d.id: 3.0 for d in DIMENSIONS}, f"Evaluation failed: {exc}"
