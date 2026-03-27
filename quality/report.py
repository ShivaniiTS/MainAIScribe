"""
Quality report writer — renders QualityResult as a Markdown file.

Output format (per CLAUDE.md spec):
    # Quality Report — sample_001
    **Pipeline Version:** v2 | **Overall Score:** 4.2 / 5.0

    ## Dimension Scores
    | Dimension | Score | Weight |
    ...

    ## Fact Check
    | Category | Found | Total | Missed |
    ...

    ## Section Coverage
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from quality.dimensions import DIMENSIONS
from quality.evaluator import QualityResult


def write_quality_report(result: QualityResult, path: Path) -> None:
    """Render a QualityResult as a Markdown quality report file."""
    lines: list[str] = [
        f"# Quality Report — {result.sample_id}",
        "",
        f"**Pipeline Version:** {result.version}  ",
        f"**Overall Score:** {result.overall_score:.1f} / 5.0  ",
        f"**Keyword Overlap:** {result.keyword_overlap:.0%}  ",
        f"**Evaluation time:** {result.elapsed_s:.1f}s  ",
        "",
        "---",
        "",
        "## Dimension Scores",
        "",
        "| Dimension | Score | Weight | Assessment |",
        "|-----------|-------|--------|------------|",
    ]

    for dim in DIMENSIONS:
        score = result.dimensions.get(dim.id, 0.0)
        stars = "★" * int(round(score)) + "☆" * (5 - int(round(score)))
        assessment = "✓ Good" if score >= 4.0 else ("⚠ Fair" if score >= 3.0 else "✗ Poor")
        lines.append(
            f"| {dim.label} | {score:.1f}/5 {stars} | {dim.weight:.0%} | {assessment} |"
        )

    if result.llm_rationale:
        lines += [
            "",
            "**Evaluator notes:**",
            f"> {result.llm_rationale}",
        ]

    # Fact check
    if result.fact_check:
        fc = result.fact_check
        lines += [
            "",
            "---",
            "",
            "## Fact Check",
            "",
            "| Category | Found | Total | Missed |",
            "|----------|-------|-------|--------|",
        ]
        cats = [
            ("Medications", "medications", fc.missed_medications),
            ("Diagnoses", "diagnoses", fc.missed_diagnoses),
            ("Exam findings", "exam_findings", fc.missed_findings),
            ("Plan items", "plan_items", fc.missed_plan_items),
        ]
        for label, attr, missed in cats:
            found, total = getattr(fc, attr)
            missed_str = ", ".join(missed[:5]) if missed else "—"
            lines.append(f"| {label} | {found} | {total} | {missed_str} |")

    # Section coverage
    lines += [
        "",
        "---",
        "",
        "## Section Coverage",
        "",
        f"**Present:** {', '.join(result.sections_present) or '—'}  ",
        f"**Missing (vs gold):** {', '.join(result.sections_missing) or '—'}  ",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_aggregate_report(
    results: list[QualityResult],
    version: str,
    out_path: Path,
) -> None:
    """Write aggregate quality report across all samples."""
    if not results:
        return

    scored = [r for r in results if r.has_gold]
    avg_overall = sum(r.overall_score for r in scored) / len(scored) if scored else 0

    lines: list[str] = [
        f"# Aggregate Quality Report — Pipeline {version}",
        "",
        f"**Samples evaluated:** {len(results)} ({len(scored)} with gold standard)  ",
        f"**Average overall score:** {avg_overall:.2f} / 5.0  ",
        f"**Average keyword overlap:** {sum(r.keyword_overlap for r in scored)/len(scored):.0%}  " if scored else "",
        "",
        "---",
        "",
        "## Dimension Averages",
        "",
        "| Dimension | Avg Score | Min | Max |",
        "|-----------|-----------|-----|-----|",
    ]

    for dim in DIMENSIONS:
        scores = [r.dimensions.get(dim.id, 0) for r in scored]
        if scores:
            avg = sum(scores) / len(scores)
            lines.append(
                f"| {dim.label} | {avg:.2f} | {min(scores):.1f} | {max(scores):.1f} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Per-Sample Scores",
        "",
        "| Sample | Overall | Accuracy | Complete | No Halluc | Structure | Language | Overlap | Status |",
        "|--------|---------|----------|----------|-----------|-----------|----------|---------|--------|",
    ]

    for r in results:
        d = r.dimensions
        status = "✓" if r.has_gold else "no gold"
        lines.append(
            f"| {r.sample_id} "
            f"| {r.overall_score:.2f} "
            f"| {d.get('medical_accuracy', 0):.1f} "
            f"| {d.get('completeness', 0):.1f} "
            f"| {d.get('no_hallucination', 0):.1f} "
            f"| {d.get('structure', 0):.1f} "
            f"| {d.get('clinical_language', 0):.1f} "
            f"| {r.keyword_overlap:.0%} "
            f"| {status} |"
        )

    # Worst performers
    if scored:
        worst = sorted(scored, key=lambda r: r.overall_score)[:5]
        lines += [
            "",
            "---",
            "",
            "## Lowest Scoring Samples",
            "",
        ]
        for r in worst:
            weak_dims = sorted(r.dimensions.items(), key=lambda x: x[1])[:2]
            weak_str = ", ".join(f"{k}={v:.1f}" for k, v in weak_dims)
            lines.append(f"- **{r.sample_id}**: {r.overall_score:.2f}/5 — weakest: {weak_str}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Quality report  : {out_path}")
