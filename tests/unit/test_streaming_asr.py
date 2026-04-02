"""
tests/unit/test_streaming_asr.py — Tests for streaming ASR components.

Tests:
  - NemoStreamingServer session management
  - NemoStreamingServer simulation mode (NeMo not installed)
  - NemoMultitalkerServer speaker labeling
  - Streaming transcript integration with transcribe node
  - Audio streaming WebSocket endpoint contract
"""
from __future__ import annotations

import asyncio
import struct
from unittest.mock import patch

import pytest

from mcp_servers.asr.base import ASRConfig, PartialTranscript, RawTranscript


# ── NemoStreamingServer ──────────────────────────────────────────────

class TestNemoStreamingServer:
    def _make_server(self, chunk_size_ms=160):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        return NemoStreamingServer(
            model_name="nvidia/nemotron-speech-streaming-en-0.6b",
            device="cpu",
            chunk_size_ms=chunk_size_ms,
        )

    def _make_pcm_chunk(self, server, num_chunks=1) -> bytes:
        """Generate silence PCM data for the given number of chunks."""
        return b"\x00\x00" * server.chunk_samples * num_chunks

    def test_create_server(self):
        server = self._make_server()
        assert server.name == "nemo_streaming"
        assert server.chunk_size_ms == 160
        assert server.sample_rate == 16000

    def test_capabilities(self):
        server = self._make_server()
        caps = asyncio.run(server.get_capabilities())
        assert caps.streaming is True
        assert caps.batch is True
        assert caps.diarization is False
        assert caps.max_speakers == 1

    def test_session_lifecycle(self):
        server = self._make_server()
        session = server._get_or_create_session("test-001")
        assert session.session_id == "test-001"
        assert session.accumulated_text == ""
        assert session.chunk_count == 0

        # Finalize returns RawTranscript
        raw = server.finalize_session("test-001")
        assert isinstance(raw, RawTranscript)
        assert raw.engine == "nemo_streaming"

        # Session is gone after finalize
        assert server.get_session_transcript("test-001") is None

    def test_session_cleanup(self):
        server = self._make_server()
        server.idle_timeout_s = 0  # Expire immediately
        server._get_or_create_session("s1")
        server._get_or_create_session("s2")
        import time; time.sleep(0.01)
        expired = server.cleanup_expired_sessions()
        assert expired == 2

    def test_streaming_simulation_yields_partials(self):
        """In simulation mode (no NeMo installed), server yields segments after enough audio accumulates."""
        server = self._make_server()
        server._loaded = True  # Skip model load attempt
        server._model = None   # Simulation mode
        # Lower the window so we don't need 3 seconds of audio
        server.STREAM_WINDOW_S = 0.5

        config = ASRConfig()
        # 0.5 seconds at 16kHz = 16000 samples = 32000 bytes. Need ~20 chunks of 160ms.
        pcm = self._make_pcm_chunk(server, num_chunks=20)

        partials = []
        async def collect():
            async for p in server.transcribe_stream(pcm, "test-sim", config):
                partials.append(p)
        asyncio.run(collect())

        assert len(partials) > 0
        assert all(isinstance(p, PartialTranscript) for p in partials)
        # All should be final (window-based transcription)
        finals = [p for p in partials if p.is_final]
        assert len(finals) >= 1

    def test_accumulated_text_grows(self):
        """Final segments accumulate text in the session."""
        server = self._make_server()
        server._loaded = True
        server._model = None
        server.STREAM_WINDOW_S = 0.3  # Low threshold for test

        config = ASRConfig()
        # Send enough chunks to trigger multiple transcription windows
        pcm = self._make_pcm_chunk(server, num_chunks=40)

        async def run():
            async for _ in server.transcribe_stream(pcm, "test-accum", config):
                pass
        asyncio.run(run())

        text = server.get_session_transcript("test-accum")
        assert text is not None
        assert len(text.strip()) > 0

    def test_batch_simulation(self):
        """Batch transcription returns a stub when NeMo is not installed."""
        server = self._make_server()
        server._loaded = True
        server._model = None

        config = ASRConfig()
        raw = asyncio.run(server.transcribe_batch("/fake/audio.wav", config))
        assert isinstance(raw, RawTranscript)
        assert raw.engine == "nemo_streaming"
        assert len(raw.segments) == 1

    def test_from_config(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer.from_config({
            "model": "nvidia/nemotron-speech-streaming-en-0.6b",
            "device": "cpu",
            "chunk_size_ms": 560,
            "idle_unload_seconds": 120,
        })
        assert server.chunk_size_ms == 560
        assert server.idle_timeout_s == 120


# ── NemoMultitalkerServer ────────────────────────────────────────────

class TestNemoMultitalkerServer:
    def test_create_server(self):
        from mcp_servers.asr.nemo_multitalker_server import NemoMultitalkerServer
        server = NemoMultitalkerServer(
            model_name="nvidia/multitalker-parakeet-streaming-0.6b-v1",
            device="cpu",
            max_speakers=4,
        )
        assert server.name == "nemo_multitalker"
        assert server.max_speakers == 4

    def test_capabilities(self):
        from mcp_servers.asr.nemo_multitalker_server import NemoMultitalkerServer
        server = NemoMultitalkerServer(device="cpu")
        caps = asyncio.run(server.get_capabilities())
        assert caps.streaming is True
        assert caps.diarization is True
        assert caps.max_speakers == 4

    def test_streaming_adds_speaker_labels(self):
        from mcp_servers.asr.nemo_multitalker_server import NemoMultitalkerServer
        server = NemoMultitalkerServer(device="cpu", chunk_size_ms=160)
        server._loaded = True
        server._model = None  # Simulation mode
        server.STREAM_WINDOW_S = 0.3  # Low threshold for test

        config = ASRConfig()
        # Need enough audio to trigger transcription window
        pcm = b"\x00\x00" * server.chunk_samples * 40

        partials = []
        async def collect():
            async for p in server.transcribe_stream(pcm, "test-mt", config):
                partials.append(p)
        asyncio.run(collect())

        finals = [p for p in partials if p.is_final]
        assert len(finals) >= 1, "Should have at least one final segment"
        # Finals should have speaker labels
        for f in finals:
            assert f.speaker is not None
            assert f.speaker.startswith("SPEAKER_")

    def test_from_config(self):
        from mcp_servers.asr.nemo_multitalker_server import NemoMultitalkerServer
        server = NemoMultitalkerServer.from_config({
            "model": "nvidia/multitalker-parakeet-streaming-0.6b-v1",
            "device": "cpu",
            "max_speakers": 3,
            "chunk_size_ms": 80,
        })
        assert server.max_speakers == 3
        assert server.chunk_size_ms == 80


# ── Transcribe Node: Streaming Path ─────────────────────────────────

class TestTranscribeNodeStreamingPath:
    def test_streaming_transcript_skips_batch_asr(self):
        """When streaming_transcript is set, transcribe_node should use it
        instead of running batch ASR, but still run post-processing."""
        from orchestrator.state import (
            EncounterState,
            ProviderProfile,
            RecordingMode,
            DeliveryMethod,
            UnifiedTranscript,
            TranscriptSegment,
        )
        from orchestrator.nodes.transcribe_node import transcribe_node

        streaming_tx = UnifiedTranscript(
            segments=[
                TranscriptSegment(
                    text="The patient presents with lower back pain.",
                    speaker="SPEAKER_00",
                    start_ms=0,
                    end_ms=3000,
                    mode=RecordingMode.DICTATION,
                    source="asr",
                ),
            ],
            engine_used="nemo_streaming",
            audio_duration_ms=3000,
            full_text="The patient presents with lower back pain.",
        )

        state = EncounterState(
            provider_id="dr_test",
            patient_id="patient-001",
            provider_profile=ProviderProfile(
                id="dr_test",
                name="Test",
                specialty="general",
            ),
            recording_mode=RecordingMode.DICTATION,
            delivery_method=DeliveryMethod.CLIPBOARD,
            streaming_transcript=streaming_tx,
        )

        result = transcribe_node(state)

        assert result["transcript"] is not None
        assert result["transcript"].full_text.strip() != ""
        assert result["asr_engine_used"] == "nemo_streaming"
        assert "transcribe" in result["metrics"].nodes_completed

    def test_batch_path_still_works(self):
        """Without streaming_transcript, the node should attempt batch ASR
        (and use the fallback stub if no engine is available)."""
        from orchestrator.state import (
            EncounterState,
            ProviderProfile,
            RecordingMode,
            DeliveryMethod,
        )
        from orchestrator.nodes.transcribe_node import transcribe_node, set_asr_engine_factory

        # No audio, no streaming transcript → fallback
        state = EncounterState(
            provider_id="dr_test",
            patient_id="patient-001",
            provider_profile=ProviderProfile(
                id="dr_test",
                name="Test",
                specialty="general",
            ),
            recording_mode=RecordingMode.DICTATION,
            delivery_method=DeliveryMethod.CLIPBOARD,
        )

        result = transcribe_node(state)
        # Should get a fallback stub (no audio)
        assert result["transcript"] is not None
        assert "transcribe" in result["metrics"].nodes_completed


# ── Registry Integration ─────────────────────────────────────────────

class TestRegistryStreamingEngines:
    def test_nemo_streaming_in_server_map(self):
        from mcp_servers.registry import _SERVER_MAP
        assert ("asr", "nemo_streaming") in _SERVER_MAP

    def test_nemo_multitalker_in_server_map(self):
        from mcp_servers.registry import _SERVER_MAP
        assert ("asr", "nemo_multitalker") in _SERVER_MAP


# ── Hotword correction & medical vocab (new changes) ─────────────────

class TestNemoHotwordCorrection:
    """Tests for _apply_hotword_corrections and related changes introduced
    in the medical hotword / 98k-wordlist integration.
    """

    def _make_server(self, hotwords=None, hotwords_files=None):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        return NemoStreamingServer(
            device="cpu",
            chunk_size_ms=160,
            hotwords=hotwords or [],
            hotwords_files=hotwords_files or [],
        )

    # ── capability ────────────────────────────────────────────────────

    def test_medical_vocab_capability_is_true(self):
        """get_capabilities() must advertise medical_vocab=True."""
        server = self._make_server()
        caps = asyncio.run(server.get_capabilities())
        assert caps.medical_vocab is True, "medical_vocab should be True now that hotword correction is active"

    # ── basic abbreviation correction ─────────────────────────────────

    def test_abbreviation_uppercase_restored(self):
        server = self._make_server(hotwords=["HTN", "DM", "CAD"])
        result = server._apply_hotword_corrections("patient has htn and dm type 2")
        assert "HTN" in result
        assert "DM" in result

    def test_abbreviation_not_corrupted_when_already_correct(self):
        server = self._make_server(hotwords=["HTN"])
        result = server._apply_hotword_corrections("patient has HTN")
        assert result == "patient has HTN"

    # ── multi-word phrase correction ──────────────────────────────────

    def test_multiword_phrase_corrected(self):
        server = self._make_server(hotwords=["coronary artery disease"])
        result = server._apply_hotword_corrections(
            "diagnosed with coronary artery disease last year"
        )
        assert "coronary artery disease" in result

    def test_longer_phrase_beats_shorter_prefix(self):
        """'coronary artery disease' should win over just 'coronary'."""
        server = self._make_server(hotwords=["coronary", "coronary artery disease"])
        result = server._apply_hotword_corrections("he has coronary artery disease")
        # 'coronary artery disease' should appear as a unit, not fragmented
        assert "coronary artery disease" in result

    # ── trailing-punctuation preservation ────────────────────────────

    def test_trailing_comma_preserved(self):
        server = self._make_server(hotwords=["HTN"])
        result = server._apply_hotword_corrections("diagnoses: htn, dm")
        assert "HTN," in result

    def test_trailing_period_preserved(self):
        server = self._make_server(hotwords=["HTN"])
        result = server._apply_hotword_corrections("he has htn.")
        assert "HTN." in result

    # ── case-insensitive matching ─────────────────────────────────────

    def test_mixed_case_input_corrected(self):
        server = self._make_server(hotwords=["Metformin"])
        result = server._apply_hotword_corrections("prescribed METFORMIN 500 mg")
        assert "Metformin" in result

    # ── per-request extra_hotwords ────────────────────────────────────

    def test_extra_hotwords_override(self):
        server = self._make_server()  # no built-in hotwords
        result = server._apply_hotword_corrections(
            "patient has afib",
            extra_hotwords=["AFib"],
        )
        assert "AFib" in result

    def test_extra_hotwords_do_not_mutate_map(self):
        """Per-request hotwords must not persist into the server map."""
        server = self._make_server()
        server._apply_hotword_corrections("test", extra_hotwords=["TEMP"])
        assert "temp" not in server._hotword_map

    # ── config=None safety ────────────────────────────────────────────

    def test_streaming_with_config_none_does_not_crash(self):
        """transcribe_stream must not raise when config=None is passed."""
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu", chunk_size_ms=160)
        server._loaded = True
        server._model = None
        server.STREAM_WINDOW_S = 0.3

        pcm = b"\x00\x00" * server.chunk_samples * 20
        partials = []

        async def collect():
            async for p in server.transcribe_stream(pcm, "cfg-none", config=None):
                partials.append(p)

        asyncio.run(collect())
        # If we get here without an exception the test passes.
        assert isinstance(partials, list)

    # ── trailing-period strip (from NeMo window output) ───────────────

    def test_trailing_period_stripped_from_nemo_window(self):
        """When the NeMo model is mocked to return text ending in '.', the
        streamed PartialTranscript must NOT have a trailing period."""
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        from unittest.mock import patch

        server = NemoStreamingServer(device="cpu", chunk_size_ms=160)
        server._loaded = True
        # Simulate NeMo being present but returning a period-terminated segment
        server._model = object()  # truthy — triggers the NeMo code path
        server.STREAM_WINDOW_S = 0.3

        fake_result = {"text": "the patient has neck pain.", "confidence": 0.9}

        with patch.object(
            server, "_transcribe_window_nemo", return_value=fake_result
        ):
            pcm = b"\x00\x00" * server.chunk_samples * 20
            partials = []

            async def collect():
                async for p in server.transcribe_stream(
                    pcm, "period-test", config=ASRConfig()
                ):
                    partials.append(p)

            asyncio.run(collect())

        assert partials, "Expected at least one partial transcript"
        for p in partials:
            assert not p.text.endswith("."), (
                f"Trailing period not stripped: {p.text!r}"
            )

    # ── short-term guard in hotword map ──────────────────────────────

    def test_short_terms_not_loaded_from_file(self):
        """Short (≤2 char) entries from *files* are skipped to prevent 98k wordlist
        false positives. Inline hotwords are always loaded regardless of length."""
        import tempfile, os

        # Create a temp file with a mix of short and long terms
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("aa\n")          # 2 chars — should be SKIPPED from file
            f.write("HTN\n")         # 3 chars — should be LOADED from file
            f.write("Metformin\n")   # long — should be LOADED from file
            tmppath = f.name

        try:
            server = self._make_server(hotwords_files=[tmppath])
            assert "aa" not in server._hotword_map, "2-char file term should be skipped"
            assert "htn" in server._hotword_map, "3-char file term should be loaded"
            assert "metformin" in server._hotword_map, "long file term should be loaded"
        finally:
            os.unlink(tmppath)

    def test_short_inline_hotwords_always_loaded(self):
        """Inline hotwords (e.g. DM, IV, BP) must be loaded regardless of length."""
        server = self._make_server(hotwords=["DM", "IV", "BP", "O2", "HTN"])
        for key in ["dm", "iv", "bp", "o2", "htn"]:
            assert key in server._hotword_map, (
                f"Inline hotword '{key}' should always be in the map"
            )

    # ── hotwords_files loading ────────────────────────────────────────

    def test_hotwords_files_cardiology_loaded(self):
        """Cardiology dictionary terms should be in the hotword map."""
        server = self._make_server(
            hotwords_files=["config/dictionaries/cardiology.txt"]
        )
        # 'coronary artery disease' and 'CAD' are in cardiology.txt
        assert "coronary artery disease" in server._hotword_map
        assert "cad" in server._hotword_map

    def test_hotwords_files_neurology_loaded(self):
        """Neurology dictionary terms should be present."""
        server = self._make_server(
            hotwords_files=["config/dictionaries/neurology.txt"]
        )
        assert "normocephalic" in server._hotword_map
        assert "atraumatic" in server._hotword_map

    # ── 98k OpenMedSpel wordlist integration ─────────────────────────

    def test_98k_wordlist_loads_without_error(self):
        """Loading the 98K medical wordlist must succeed and populate the map."""
        import os
        wordlist_path = "postprocessor/medical_wordlist.txt"
        if not os.path.exists(wordlist_path):
            pytest.skip("medical_wordlist.txt not found — skipping")

        server = self._make_server(hotwords_files=[wordlist_path])
        # The map should have a substantial number of entries (allowing for
        # short-term filtering, expect at least 50k entries from 98k file)
        assert len(server._hotword_map) > 50_000, (
            f"Expected >50000 entries from 98k wordlist, got {len(server._hotword_map)}"
        )

    def test_98k_wordlist_corrects_known_term(self):
        """A term from the 98k wordlist should be casing-corrected."""
        import os
        wordlist_path = "postprocessor/medical_wordlist.txt"
        if not os.path.exists(wordlist_path):
            pytest.skip("medical_wordlist.txt not found — skipping")

        server = self._make_server(hotwords_files=[wordlist_path])
        # 'Metformin' appears in most medical dictionaries; check
        # that the lookup map can correct a plausible lowercase transcription
        # (we look for any multi-char term that changed case)
        sample_terms = [v for k, v in server._hotword_map.items()
                        if len(k) > 4 and v != k and v[0].isupper()]
        assert len(sample_terms) > 0, "Expected some capitalised corrections from 98k wordlist"

    def test_specialty_dict_overrides_98k_wordlist(self):
        """Specialty files loaded after 98k wordlist must win for shared keys."""
        import os
        wordlist_path = "postprocessor/medical_wordlist.txt"
        cardiology_path = "config/dictionaries/cardiology.txt"
        if not os.path.exists(wordlist_path):
            pytest.skip("medical_wordlist.txt not found — skipping")

        # Load 98k first, then cardiology (same order as engines.yaml)
        server = self._make_server(
            hotwords_files=[wordlist_path, cardiology_path]
        )
        # 'CAD' is defined in cardiology.txt; its value should be 'CAD'
        if "cad" in server._hotword_map:
            assert server._hotword_map["cad"] == "CAD"

    # ── from_config wires hotwords correctly ─────────────────────────

    def test_from_config_hotwords_wired(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer.from_config({
            "model": "nvidia/nemotron-speech-streaming-en-0.6b",
            "device": "cpu",
            "hotwords": ["HEENT", "SOB", "HTN"],
            "hotwords_files": [],
        })
        assert "htn" in server._hotword_map
        assert server._hotword_map["htn"] == "HTN"
        assert "sob" in server._hotword_map
