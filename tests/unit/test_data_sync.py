"""
tests/unit/test_data_sync.py — Tests for data sync scripts.

Tests the PHI isolation logic, file discovery, and sync behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestSyncToPipelineDiscovery:
    """Tests for sync_to_pipeline.py sample discovery."""

    def test_discovers_dictation_samples(self, tmp_path):
        """Should discover samples in dictation/ subdirectories."""
        from scripts.sync_to_pipeline import discover_samples, SAFE_FILES

        # Create test structure
        sample_dir = tmp_path / "dictation" / "dr_test" / "patient_001"
        sample_dir.mkdir(parents=True)
        (sample_dir / "dictation.mp3").write_bytes(b"audio data")
        (sample_dir / "encounter_details.json").write_text("{}")
        (sample_dir / "patient_demographics.json").write_text("{}")  # PHI — should be excluded

        samples = discover_samples(tmp_path)
        assert len(samples) == 1
        assert samples[0]["sample_id"] == "patient_001"
        assert samples[0]["mode"] == "dictation"
        assert samples[0]["provider_id"] == "dr_test"

        # Verify PHI files are NOT included
        filenames = {f.name for f in samples[0]["files"]}
        assert "dictation.mp3" in filenames
        assert "encounter_details.json" in filenames
        assert "patient_demographics.json" not in filenames  # PHI excluded

    def test_discovers_conversation_samples(self, tmp_path):
        """Should discover samples in conversation/ subdirectories."""
        from scripts.sync_to_pipeline import discover_samples

        sample_dir = tmp_path / "conversation" / "dr_test" / "patient_002"
        sample_dir.mkdir(parents=True)
        (sample_dir / "conversation_audio.mp3").write_bytes(b"audio")
        (sample_dir / "encounter_details.json").write_text("{}")

        samples = discover_samples(tmp_path)
        assert len(samples) == 1
        assert samples[0]["mode"] == "ambient"

    def test_filter_by_sample_ids(self, tmp_path):
        """Should only return specified sample IDs."""
        from scripts.sync_to_pipeline import discover_samples

        for name in ["sample_001", "sample_002", "sample_003"]:
            d = tmp_path / "dictation" / "dr_test" / name
            d.mkdir(parents=True)
            (d / "dictation.mp3").write_bytes(b"audio")

        samples = discover_samples(tmp_path, sample_ids=["sample_002"])
        assert len(samples) == 1
        assert samples[0]["sample_id"] == "sample_002"

    def test_skips_empty_directories(self, tmp_path):
        """Directories without audio files should be skipped."""
        from scripts.sync_to_pipeline import discover_samples

        d = tmp_path / "dictation" / "dr_test" / "empty_sample"
        d.mkdir(parents=True)
        (d / "patient_demographics.json").write_text("{}")  # Only PHI, no audio

        samples = discover_samples(tmp_path)
        assert len(samples) == 0


class TestPhiIsolation:
    """Tests to ensure PHI files are never sent to the pipeline server."""

    def test_safe_files_list(self):
        from scripts.sync_to_pipeline import SAFE_FILES, PHI_FILES

        # No overlap between safe and PHI
        assert SAFE_FILES.isdisjoint(PHI_FILES)

        # Audio files are safe
        assert "dictation.mp3" in SAFE_FILES
        assert "conversation_audio.mp3" in SAFE_FILES

        # Demographics are PHI
        assert "patient_demographics.json" in PHI_FILES
        assert "patient_context.yaml" in PHI_FILES
        assert "final_soap_note.md" in PHI_FILES

    def test_encounter_details_is_safe(self):
        """encounter_details.json should be safe — it has mode/visit_type but no patient names."""
        from scripts.sync_to_pipeline import SAFE_FILES
        assert "encounter_details.json" in SAFE_FILES


class TestSyncFromPipelineDiscovery:
    """Tests for sync_from_pipeline.py helper functions."""

    def test_discover_local_samples(self, tmp_path):
        from scripts.sync_from_pipeline import discover_local_samples

        # Create test output structure
        (tmp_path / "output" / "dictation" / "dr_test" / "s001").mkdir(parents=True)
        (tmp_path / "output" / "conversation" / "dr_test" / "s002").mkdir(parents=True)
        (tmp_path / "data" / "dictation" / "dr_test" / "s003").mkdir(parents=True)

        samples = discover_local_samples(tmp_path / "output", tmp_path / "data")
        assert "s001" in samples
        assert "s002" in samples
        assert "s003" in samples
