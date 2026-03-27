"""
Quality evaluation framework — LLM-as-judge + fact extraction + section coverage.

Usage:
    from quality.evaluator import QualityEvaluator
    evaluator = QualityEvaluator(engine)
    result = evaluator.evaluate(generated_note, gold_note, transcript)
    # result.overall_score, result.dimensions, result.fact_check, result.missing_sections
"""

from quality.evaluator import QualityEvaluator, QualityResult
from quality.report import write_quality_report

__all__ = ["QualityEvaluator", "QualityResult", "write_quality_report"]
