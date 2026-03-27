"""
Quality regression test — asserts that pipeline v2 quality scores meet or exceed baseline.

Reads quality_baseline.yaml (thresholds) and the latest quality_report_v{N}.md
and fails if any threshold is violated.

Usage:
    pytest tests/quality/test_regression.py
    pytest tests/quality/test_regression.py -v --version v2
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from config.paths import ROOT, CONFIG_DIR, OUTPUT_DIR
BASELINE_PATH = CONFIG_DIR / "quality_baseline.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_baseline() -> dict:
    return yaml.safe_load(BASELINE_PATH.read_text())


def _parse_quality_report(version: str) -> dict:
    """
    Parse output/quality_report_{version}.md and return a dict with:
        overall_avg, keyword_overlap_avg,
        dimension_averages: {dim: score},
        samples: [{id, overall, ...}, ...]
    """
    report_path = OUTPUT_DIR / f"quality_report_{version}.md"
    if not report_path.exists():
        pytest.skip(f"Quality report not found: {report_path}")

    text = report_path.read_text()

    # Overall score
    m = re.search(r"\*\*Average overall score:\*\*\s*([\d.]+)", text)
    overall_avg = float(m.group(1)) if m else 0.0

    # Keyword overlap
    m = re.search(r"\*\*Average keyword overlap:\*\*\s*([\d]+)%", text)
    keyword_overlap_avg = int(m.group(1)) / 100.0 if m else 0.0

    # Dimension averages table
    # | Medical Accuracy | 4.00 | ... |
    dim_map = {
        "Medical Accuracy": "medical_accuracy_avg",
        "Completeness": "completeness_avg",
        "No Hallucination": "no_hallucination_avg",
        "Structure Compliance": "structure_compliance_avg",
        "Clinical Language": "clinical_language_avg",
        "Readability": "readability_avg",
    }
    dimension_averages: dict[str, float] = {}
    for label, key in dim_map.items():
        pattern = rf"\|\s*{re.escape(label)}\s*\|\s*([\d.]+)"
        m = re.search(pattern, text)
        if m:
            dimension_averages[key] = float(m.group(1))

    # Per-sample scores
    # | 224889 | 4.55 | 4.0 | 5.0 | 5.0 | 5.0 | 4.0 | 32% | ✓ |
    samples = []
    sample_pattern = re.compile(
        r"\|\s*(\S+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    )
    in_table = False
    for line in text.splitlines():
        if "| Sample |" in line:
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            m = sample_pattern.match(line)
            if m:
                samples.append({
                    "id": m.group(1),
                    "overall": float(m.group(2)),
                    "accuracy": float(m.group(3)),
                    "completeness": float(m.group(4)),
                    "no_hallucination": float(m.group(5)),
                    "structure": float(m.group(6)),
                    "language": float(m.group(7)),
                })
        elif in_table and not line.startswith("|"):
            in_table = False

    return {
        "overall_avg": overall_avg,
        "keyword_overlap_avg": keyword_overlap_avg,
        "dimension_averages": dimension_averages,
        "samples": samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE_PATH.exists(), f"Baseline not found: {BASELINE_PATH}"
    return _load_baseline()


@pytest.fixture(scope="module")
def report(request) -> dict:
    version = getattr(request, "param", None) or "v2"
    return _parse_quality_report(version)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_baseline_file_exists():
    """Quality baseline YAML must exist."""
    assert BASELINE_PATH.exists(), f"Missing: {BASELINE_PATH}"


def test_overall_avg(baseline, report):
    """Average overall quality score must meet baseline threshold."""
    threshold = baseline["thresholds"]["overall_avg"]
    actual = report["overall_avg"]
    assert actual >= threshold, (
        f"Overall avg {actual:.2f} < baseline {threshold:.2f}"
    )


def test_min_sample_score(baseline, report):
    """No individual sample may fall below the minimum threshold."""
    threshold = baseline["thresholds"]["min_sample_score"]
    failing = [
        f"{s['id']}: {s['overall']:.2f}"
        for s in report["samples"]
        if s["overall"] < threshold
    ]
    assert not failing, (
        f"Samples below min threshold ({threshold}):\n" + "\n".join(failing)
    )


@pytest.mark.parametrize("dim,key", [
    ("Medical Accuracy",      "medical_accuracy_avg"),
    ("Completeness",          "completeness_avg"),
    ("No Hallucination",      "no_hallucination_avg"),
    ("Structure Compliance",  "structure_compliance_avg"),
    ("Clinical Language",     "clinical_language_avg"),
    ("Readability",           "readability_avg"),
])
def test_dimension_avg(baseline, report, dim, key):
    """Each dimension average must meet its baseline threshold."""
    threshold = baseline["thresholds"].get(key)
    if threshold is None:
        pytest.skip(f"No threshold defined for {key}")
    actual = report["dimension_averages"].get(key)
    if actual is None:
        pytest.skip(f"Dimension {dim} not found in report")
    assert actual >= threshold, (
        f"{dim}: {actual:.2f} < baseline {threshold:.2f}"
    )


def test_no_zero_score_samples(report):
    """No sample should have an overall score of 0 (pipeline failure)."""
    zeros = [s["id"] for s in report["samples"] if s["overall"] == 0.0]
    assert not zeros, f"Zero-score samples (pipeline failures): {zeros}"


def test_report_has_samples(report):
    """Quality report must contain at least 10 evaluated samples."""
    assert len(report["samples"]) >= 10, (
        f"Only {len(report['samples'])} samples in report — expected >= 10"
    )
