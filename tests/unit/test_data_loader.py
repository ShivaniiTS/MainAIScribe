"""
tests/unit/test_data_loader.py — Tests for data_loader module.

Covers:
- Version discovery (dynamic, no hardcoding)
- resolve_version("latest") resolution
- Quality report parsing with edge cases
- Sample listing with empty directories
- Quality cache invalidation
- Graceful handling of missing/malformed data
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_dirs(tmp_path):
    """Create mock data and output directories."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()
    return data_dir, output_dir


@pytest.fixture
def dl(mock_dirs):
    """Import data_loader with mocked paths."""
    data_dir, output_dir = mock_dirs
    with patch.dict(os.environ, {
        "AI_SCRIBE_DATA_DIR": str(data_dir),
        "AI_SCRIBE_OUTPUT_DIR": str(output_dir),
    }):
        # Must reload config.paths first so DATA_DIR/OUTPUT_DIR pick up env vars
        import config.paths as paths_mod
        importlib.reload(paths_mod)
        import api.data_loader as mod
        importlib.reload(mod)
        # Ensure module uses our temp dirs
        assert mod.OUTPUT_DIR == output_dir
        assert mod.DATA_DIR == data_dir
        # Clear caches
        mod._quality_cache.clear()
        yield mod


def _create_sample(output_dir: Path, data_dir: Path, mode: str, physician: str,
                   sample_id: str, versions: list[str], has_gold: bool = False):
    """Helper to create a sample with generated notes and optional gold."""
    out_enc = output_dir / mode / physician / sample_id
    out_enc.mkdir(parents=True, exist_ok=True)
    for v in versions:
        (out_enc / f"generated_note_{v}.md").write_text(f"Note {v}")
        (out_enc / f"audio_transcript_{v}.txt").write_text(f"Transcript {v}")

    data_enc = data_dir / mode / physician / sample_id
    data_enc.mkdir(parents=True, exist_ok=True)
    if has_gold:
        (data_enc / "final_soap_note.md").write_text("Gold standard note")

    # Create demographics
    (data_enc / "patient_demographics.json").write_text(json.dumps({
        "first_name": "Jane",
        "last_name": "Doe",
        "date_of_birth": "1985-01-15",
    }))
    (data_enc / "encounter_details.json").write_text(json.dumps({
        "encounter_id": "enc-001",
        "mode": "dictation" if mode == "dictation" else "ambient",
        "provider": {"full_name": "Dr. Smith"},
    }))


class TestVersionDiscovery:
    def test_no_output_returns_empty(self, dl):
        assert dl.get_versions() == []

    def test_discovers_versions_from_files(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "sample_001", ["v1", "v3", "v5"])
        versions = dl.get_versions()
        assert versions == ["v5", "v3", "v1"]

    def test_get_latest_version_with_no_data(self, dl):
        assert dl.get_latest_version() == "v1"

    def test_get_latest_version_returns_highest(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v2", "v9"])
        _create_sample(output_dir, data_dir, "conversation", "dr_b", "s2", ["v3"])
        assert dl.get_latest_version() == "v9"


class TestResolveVersion:
    def test_latest_resolves_to_actual(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_x", "s1", ["v7"])
        assert dl.resolve_version("latest") == "v7"

    def test_explicit_version_passes_through(self, dl):
        assert dl.resolve_version("v5") == "v5"

    def test_latest_with_no_data_returns_v1(self, dl):
        assert dl.resolve_version("latest") == "v1"


class TestListSamples:
    def test_empty_dirs_return_empty(self, dl):
        assert dl.list_samples() == []

    def test_discovers_samples_from_output(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "encounter_001", ["v9"])
        samples = dl.list_samples()
        assert len(samples) == 1
        assert samples[0]["sample_id"] == "encounter_001"
        assert samples[0]["mode"] == "dictation"
        assert samples[0]["physician"] == "dr_smith"
        assert samples[0]["versions"] == ["v9"]
        assert samples[0]["latest_version"] == "v9"

    def test_discovers_samples_from_data_without_output(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        # Create only data directory (no output yet)
        enc = data_dir / "conversation" / "dr_jones" / "patient_123"
        enc.mkdir(parents=True)
        (enc / "conversation_audio.mp3").write_bytes(b"audio")
        samples = dl.list_samples()
        assert len(samples) == 1
        assert samples[0]["versions"] == []
        assert samples[0]["latest_version"] is None
        assert samples[0]["mode"] == "ambient"

    def test_no_duplicate_samples(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "encounter_001", ["v9"], has_gold=True)
        samples = dl.list_samples()
        assert len(samples) == 1


class TestGetGeneratedNote:
    def test_returns_note_content(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v5"])
        content = dl.get_generated_note("s1", "v5")
        assert content == "Note v5"

    def test_returns_none_for_missing_version(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v5"])
        assert dl.get_generated_note("s1", "v99") is None

    def test_returns_none_for_missing_sample(self, dl):
        assert dl.get_generated_note("nonexistent", "v1") is None

    def test_latest_resolves_dynamically(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v3", "v5"])
        content = dl.get_generated_note("s1", "latest")
        assert content == "Note v5"


class TestGetTranscript:
    def test_returns_transcript(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v5"])
        content = dl.get_transcript("s1", "v5")
        assert content == "Transcript v5"

    def test_latest_resolves(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v3", "v7"])
        content = dl.get_transcript("s1", "latest")
        assert content == "Transcript v7"


class TestGetGoldNote:
    def test_returns_gold_when_exists(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v1"], has_gold=True)
        content = dl.get_gold_note("s1")
        assert content == "Gold standard note"

    def test_returns_none_when_no_gold(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v1"], has_gold=False)
        assert dl.get_gold_note("s1") is None


class TestQualityReportParsing:
    def _write_quality_report(self, output_dir: Path, version: str, samples: list[dict]):
        """Write a mock quality_report_{version}.md."""
        lines = [
            f"# Aggregate Quality Report — Pipeline {version}",
            "",
            "## Per-Sample Scores",
            "",
            "| Sample | Overall | Accuracy | Complete | No Halluc | Structure | Language | Overlap | Status |",
            "|--------|---------|----------|----------|-----------|-----------|----------|---------|--------|",
        ]
        for s in samples:
            lines.append(
                f"| {s['id']} | {s.get('overall', '—')} | {s.get('accuracy', '—')} "
                f"| {s.get('completeness', '—')} | {s.get('no_hallucination', '—')} "
                f"| {s.get('structure', '—')} | {s.get('language', '—')} "
                f"| {s.get('overlap', '—')} | {s.get('status', '✓')} |"
            )
        (output_dir / f"quality_report_{version}.md").write_text("\n".join(lines))

    def test_parses_valid_report(self, dl, mock_dirs):
        _, output_dir = mock_dirs
        self._write_quality_report(output_dir, "v9", [
            {"id": "s1", "overall": "4.50", "accuracy": "4.5", "completeness": "4.0",
             "no_hallucination": "5.0", "structure": "4.5", "language": "4.0", "overlap": "45%"},
        ])
        agg = dl.get_aggregate_quality("v9")
        assert agg["average"] == 4.5
        assert agg["sample_count"] == 1

    def test_empty_quality_returns_empty_dict(self, dl):
        agg = dl.get_aggregate_quality("v99")
        assert agg == {}

    def test_latest_resolves_for_quality(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        # Create a sample so version discovery finds v9
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v9"])
        self._write_quality_report(output_dir, "v9", [
            {"id": "s1", "overall": "4.20", "accuracy": "4.0", "completeness": "4.0",
             "no_hallucination": "5.0", "structure": "4.0", "language": "4.0", "overlap": "40%"},
        ])
        agg = dl.get_aggregate_quality("latest")
        assert agg["average"] == 4.2

    def test_malformed_lines_skipped(self, dl, mock_dirs):
        _, output_dir = mock_dirs
        content = """# Quality Report
| Sample | Overall | Accuracy |
|--------|---------|----------|
| bad line |
| s1 | 4.0 | 4.0 | 3.5 | 5.0 | 4.0 | 4.0 | 40% | ✓ |
"""
        (output_dir / "quality_report_v1.md").write_text(content)
        scores = dl._parse_quality_report("v1")
        assert "s1" in scores
        assert scores["s1"]["overall"] == 4.0


class TestQualityCacheInvalidation:
    def test_cache_invalidation_by_version(self, dl, mock_dirs):
        _, output_dir = mock_dirs
        # Manually populate cache (simulates a parsed report)
        dl._quality_cache["v1"] = {"s1": {"overall": 4.0}}
        dl.clear_quality_cache("v1")
        assert "v1" not in dl._quality_cache

    def test_cache_invalidation_all(self, dl):
        dl._quality_cache["v1"] = {"test": True}
        dl._quality_cache["v2"] = {"test": True}
        dl.clear_quality_cache()
        assert dl._quality_cache == {}


class TestGetPatientContext:
    def test_returns_context_from_json(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_a", "s1", ["v1"])
        ctx = dl.get_patient_context("s1")
        assert ctx is not None
        assert ctx["patient"]["name"] == "Jane Doe"

    def test_returns_none_for_missing_sample(self, dl):
        assert dl.get_patient_context("nonexistent") is None


class TestQualityByProvider:
    def test_groups_by_physician(self, dl, mock_dirs):
        data_dir, output_dir = mock_dirs
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s1", ["v5"])
        _create_sample(output_dir, data_dir, "dictation", "dr_smith", "s2", ["v5"])
        _create_sample(output_dir, data_dir, "dictation", "dr_jones", "s3", ["v5"])

        # Write quality report
        lines = [
            "# Report",
            "",
            "## Per-Sample Scores",
            "",
            "| Sample | Overall | Accuracy | Complete | No Halluc | Structure | Language | Overlap | Status |",
            "|--------|---------|----------|----------|-----------|-----------|----------|---------|--------|",
            "| s1 | 4.50 | 4.5 | 4.0 | 5.0 | 4.5 | 4.0 | 45% | ✓ |",
            "| s2 | 4.00 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 40% | ✓ |",
            "| s3 | 3.50 | 3.5 | 3.5 | 3.5 | 3.5 | 3.5 | 35% | ✓ |",
        ]
        (output_dir / "quality_report_v5.md").write_text("\n".join(lines))

        result = dl.get_aggregate_quality_by_provider("v5")
        assert len(result) == 2
        smith = next(r for r in result if r["provider_id"] == "dr_smith")
        assert smith["sample_count"] == 2
        assert smith["average"] == 4.25

    def test_empty_when_no_quality_data(self, dl):
        assert dl.get_aggregate_quality_by_provider("v99") == []
