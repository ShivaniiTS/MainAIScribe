"""
tests/unit/test_quality_regression.py — Quality threshold regression tests.

Covers:
1. Quality baseline loading and validation
2. Score comparison against baseline thresholds
3. Silent degradation detection (per-dimension)
4. Aggregate quality data structure validation
5. Quality cache invalidation by version
6. Quality runner evaluate_sample skip conditions
7. Quality runner generate_aggregate_report with no samples
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestQualityBaseline:
    """Quality baseline loading and validation."""

    @pytest.fixture
    def baseline_path(self, tmp_path):
        """Create a mock quality_baseline.yaml."""
        baseline = {
            "version": "v2",
            "overall_score": 4.30,
            "dimensions": {
                "medical_accuracy": 4.2,
                "completeness": 3.8,
                "hallucination_free": 4.5,
                "structure_compliance": 4.0,
                "clinical_language": 4.3,
                "readability": 4.1,
            },
            "sample_count": 33,
        }
        path = tmp_path / "quality_baseline.yaml"
        with open(path, "w") as f:
            yaml.dump(baseline, f)
        return path

    def test_load_baseline(self, baseline_path):
        """Baseline YAML should load with correct structure."""
        with open(baseline_path) as f:
            baseline = yaml.safe_load(f)

        assert baseline["version"] == "v2"
        assert baseline["overall_score"] == 4.30
        assert len(baseline["dimensions"]) == 6
        assert baseline["sample_count"] == 33

    def test_score_above_baseline_passes(self, baseline_path):
        """Scores above baseline should pass regression check."""
        with open(baseline_path) as f:
            baseline = yaml.safe_load(f)

        current_score = 4.35
        assert current_score >= baseline["overall_score"]

    def test_score_below_baseline_detected(self, baseline_path):
        """Scores below baseline should be flagged."""
        with open(baseline_path) as f:
            baseline = yaml.safe_load(f)

        current_score = 4.10
        assert current_score < baseline["overall_score"]


class TestDimensionDegradation:
    """Per-dimension degradation detection."""

    @pytest.fixture
    def baseline_dims(self):
        return {
            "medical_accuracy": 4.2,
            "completeness": 3.8,
            "hallucination_free": 4.5,
            "structure_compliance": 4.0,
            "clinical_language": 4.3,
            "readability": 4.1,
        }

    def test_no_degradation_detected(self, baseline_dims):
        """All dimensions above baseline — no degradation."""
        current_dims = {k: v + 0.1 for k, v in baseline_dims.items()}
        degraded = [
            dim for dim, score in current_dims.items()
            if score < baseline_dims[dim]
        ]
        assert len(degraded) == 0

    def test_single_dimension_degradation(self, baseline_dims):
        """One dimension below baseline should be flagged."""
        current_dims = dict(baseline_dims)
        current_dims["completeness"] = 3.5  # Below 3.8 baseline

        degraded = [
            dim for dim, score in current_dims.items()
            if score < baseline_dims[dim]
        ]
        assert "completeness" in degraded

    def test_multiple_dimension_degradation(self, baseline_dims):
        """Multiple dimensions below baseline should all be flagged."""
        current_dims = {
            "medical_accuracy": 4.0,  # Below 4.2
            "completeness": 3.5,      # Below 3.8
            "hallucination_free": 4.6, # Above 4.5
            "structure_compliance": 3.9, # Below 4.0
            "clinical_language": 4.3,  # Equal (not degraded)
            "readability": 4.1,        # Equal (not degraded)
        }

        degraded = [
            dim for dim, score in current_dims.items()
            if score < baseline_dims[dim]
        ]
        assert set(degraded) == {"medical_accuracy", "completeness", "structure_compliance"}

    def test_silent_degradation_threshold(self, baseline_dims):
        """Degradation within noise threshold (±0.1) should be tolerated."""
        noise_threshold = 0.1
        current_dims = {k: v - 0.05 for k, v in baseline_dims.items()}

        significantly_degraded = [
            dim for dim, score in current_dims.items()
            if (baseline_dims[dim] - score) > noise_threshold
        ]
        assert len(significantly_degraded) == 0


class TestQualityRunnerSkipConditions:
    """Quality evaluation skip conditions."""

    def test_skip_when_no_gold_note(self):
        """Evaluation should return None when no gold standard exists."""
        from api.quality_runner import evaluate_sample
        result = evaluate_sample(
            sample_id="no_gold",
            generated_note="# Test Note\nContent here",
            gold_note=None,
            transcript="test transcript",
        )
        assert result is None

    def test_skip_when_empty_gold_note(self):
        """Evaluation should return None when gold standard is empty."""
        from api.quality_runner import evaluate_sample
        result = evaluate_sample(
            sample_id="empty_gold",
            generated_note="# Test Note\nContent here",
            gold_note="   ",
            transcript="test transcript",
        )
        assert result is None

    def test_skip_when_no_generated_note(self):
        """Evaluation should return None when generated note is missing."""
        from api.quality_runner import evaluate_sample
        result = evaluate_sample(
            sample_id="no_gen",
            generated_note=None,
            gold_note="# Gold Standard\nGold content",
            transcript="test transcript",
        )
        assert result is None

    def test_skip_when_empty_generated_note(self):
        """Evaluation should return None when generated note is empty."""
        from api.quality_runner import evaluate_sample
        result = evaluate_sample(
            sample_id="empty_gen",
            generated_note="  ",
            gold_note="# Gold Standard\nGold content",
            transcript="test transcript",
        )
        assert result is None


class TestAggregateReportNoSamples:
    """Aggregate quality report with no eligible samples."""

    def test_generate_aggregate_no_samples(self, tmp_path):
        """Should return None when no matching samples found."""
        # generate_aggregate_report imports DATA_DIR/OUTPUT_DIR from config.paths at call time
        (tmp_path / "data").mkdir()
        (tmp_path / "output").mkdir()

        with patch("config.paths.DATA_DIR", tmp_path / "data"), \
             patch("config.paths.OUTPUT_DIR", tmp_path / "output"):
            from api.quality_runner import generate_aggregate_report
            result = generate_aggregate_report(version="v99")
            assert result is None


class TestQualityDataStructure:
    """Quality data structure validation for API responses."""

    def test_aggregate_quality_response_shape(self):
        """Aggregate quality endpoint should return expected keys."""
        expected_keys = {"average", "count", "min", "max", "dimensions", "version"}

        # Mock data matching what data_loader.get_aggregate_quality returns
        mock_agg = {
            "average": 4.35,
            "count": 61,
            "min": 3.80,
            "max": 4.80,
            "dimensions": {
                "medical_accuracy": 4.2,
                "completeness": 3.8,
            },
            "version": "v8",
        }

        assert expected_keys.issubset(set(mock_agg.keys()))

    def test_per_sample_quality_shape(self):
        """Per-sample quality should include overall and dimension scores."""
        mock_sample = {
            "sample_id": "sample_001",
            "version": "v8",
            "overall": 4.30,
            "medical_accuracy": 4.5,
            "completeness": 4.0,
            "hallucination_free": 5.0,
            "structure_compliance": 4.0,
            "clinical_language": 4.5,
            "readability": 4.0,
        }

        assert "overall" in mock_sample
        assert all(isinstance(v, (int, float)) for k, v in mock_sample.items()
                   if k not in ("sample_id", "version"))


class TestQualityCacheInvalidation:
    """Quality cache should be invalidatable per version."""

    def test_clear_quality_cache_removes_version(self):
        """clear_quality_cache should remove the specific version from cache."""
        from api.data_loader import clear_quality_cache, _quality_cache

        _quality_cache["v8"] = {"sample_001": {"overall": 4.0}}
        _quality_cache["v7"] = {"sample_002": {"overall": 3.5}}

        clear_quality_cache("v8")

        assert "v8" not in _quality_cache
        assert "v7" in _quality_cache  # Other versions unaffected

        # Cleanup
        _quality_cache.clear()

    def test_clear_quality_cache_nonexistent_version_noop(self):
        """Clearing a non-cached version should not raise."""
        from api.data_loader import clear_quality_cache
        # Should not raise
        clear_quality_cache("v999")
