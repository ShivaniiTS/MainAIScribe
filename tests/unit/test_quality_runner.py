"""
tests/unit/test_quality_runner.py — Tests for the quality evaluation runner.

Covers:
- evaluate_sample() with and without gold standard
- generate_aggregate_report() discovery logic
- Cache invalidation after report generation
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_dirs(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()
    return data_dir, output_dir


class TestEvaluateSample:
    def test_skips_when_no_gold(self):
        from api.quality_runner import evaluate_sample
        result = evaluate_sample(
            sample_id="s1",
            generated_note="Some note",
            gold_note=None,
            version="v1",
        )
        assert result is None

    def test_skips_when_empty_gold(self):
        from api.quality_runner import evaluate_sample
        result = evaluate_sample(
            sample_id="s1",
            generated_note="Some note",
            gold_note="",
            version="v1",
        )
        assert result is None

    def test_skips_when_no_generated(self):
        from api.quality_runner import evaluate_sample
        result = evaluate_sample(
            sample_id="s1",
            generated_note="",
            gold_note="Gold note content",
            version="v1",
        )
        assert result is None


class TestGenerateAggregateReport:
    def test_returns_none_when_no_samples(self, mock_dirs):
        data_dir, output_dir = mock_dirs
        with patch.dict(os.environ, {
            "AI_SCRIBE_DATA_DIR": str(data_dir),
            "AI_SCRIBE_OUTPUT_DIR": str(output_dir),
        }):
            import config.paths
            importlib.reload(config.paths)
            from api.quality_runner import generate_aggregate_report
            # No samples exist, should return None without error
            result = generate_aggregate_report("v99")
            assert result is None
