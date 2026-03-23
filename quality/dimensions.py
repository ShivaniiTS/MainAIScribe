"""
Scoring rubric definitions for quality evaluation.

Each dimension is scored 1–5 by the LLM judge.
Weights sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dimension:
    id: str
    label: str
    weight: float
    description: str
    rubric: str  # 1-5 scale description for the LLM judge


DIMENSIONS: list[Dimension] = [
    Dimension(
        id="medical_accuracy",
        label="Medical Accuracy",
        weight=0.30,
        description="Are all medical facts, findings, and values correct?",
        rubric=(
            "1=Many errors or contradictions. "
            "2=Several inaccuracies. "
            "3=Minor inaccuracies. "
            "4=Mostly accurate, trivial gaps. "
            "5=Fully accurate, matches transcript."
        ),
    ),
    Dimension(
        id="completeness",
        label="Completeness",
        weight=0.25,
        description="Are all clinically significant findings from the transcript captured?",
        rubric=(
            "1=Most content missing. "
            "2=Major sections missing. "
            "3=Some important content missing. "
            "4=Minor omissions only. "
            "5=All relevant content captured."
        ),
    ),
    Dimension(
        id="no_hallucination",
        label="No Hallucination",
        weight=0.20,
        description="Does the note avoid adding information not in the transcript?",
        rubric=(
            "1=Significant invented content. "
            "2=Several invented details. "
            "3=Minor unsupported inferences. "
            "4=Essentially grounded, trivial additions. "
            "5=Perfectly grounded in transcript."
        ),
    ),
    Dimension(
        id="structure",
        label="Structure Compliance",
        weight=0.10,
        description="Does the note follow the expected section structure?",
        rubric=(
            "1=Wrong structure. "
            "2=Major structural issues. "
            "3=Some sections missing or misplaced. "
            "4=Mostly correct structure. "
            "5=Perfect section structure."
        ),
    ),
    Dimension(
        id="clinical_language",
        label="Clinical Language",
        weight=0.10,
        description="Is the language appropriately clinical and professional?",
        rubric=(
            "1=Very informal or lay terms. "
            "2=Often inappropriate language. "
            "3=Mostly clinical with some lapses. "
            "4=Good clinical language. "
            "5=Excellent clinical terminology throughout."
        ),
    ),
    Dimension(
        id="readability",
        label="Readability",
        weight=0.05,
        description="Is the note well-organized, concise, and easy to read?",
        rubric=(
            "1=Very hard to read. "
            "2=Difficult to parse. "
            "3=Readable with effort. "
            "4=Clear and well-organized. "
            "5=Exceptionally clear and concise."
        ),
    ),
]

DIMENSION_MAP: dict[str, Dimension] = {d.id: d for d in DIMENSIONS}


def weighted_score(scores: dict[str, float]) -> float:
    """Compute overall weighted score from dimension scores."""
    total = 0.0
    weight_sum = 0.0
    for dim in DIMENSIONS:
        if dim.id in scores:
            total += scores[dim.id] * dim.weight
            weight_sum += dim.weight
    return round(total / weight_sum, 2) if weight_sum > 0 else 0.0
