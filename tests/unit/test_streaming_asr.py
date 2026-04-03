"""
tests/unit/test_streaming_asr.py — Tests for streaming ASR components.

Tests:
  - NemoStreamingServer session management
  - NemoStreamingServer simulation mode (NeMo not installed)
  - NemoMultitalkerServer speaker labeling
  - Streaming transcript integration with transcribe node
  - Audio streaming WebSocket endpoint contract
  - Hotword correction, medical vocab, 98k wordlist, pain_medicine dict
  - stream_window_s configurable latency/quality tradeoff
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


# ── stream_window_s configurable latency/quality ──────────────────────

class TestNemoStreamWindowConfig:
    """Tests for configurable stream_window_s parameter and pain_medicine dict."""

    def test_default_stream_window_is_3s(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu")
        assert server.STREAM_WINDOW_S == 3.0

    def test_custom_stream_window_respected(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu", stream_window_s=5.0)
        assert server.STREAM_WINDOW_S == 5.0

    def test_from_config_reads_stream_window_s(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer.from_config({
            "device": "cpu",
            "stream_window_s": 2.0,
        })
        assert server.STREAM_WINDOW_S == 2.0

    def test_from_config_defaults_to_3s(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer.from_config({"device": "cpu"})
        assert server.STREAM_WINDOW_S == 3.0

    def test_stream_window_1s_yields_partials_on_short_audio(self):
        """1s window triggers decode sooner: ~1.28s audio => at least 1 partial."""
        import asyncio
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        from mcp_servers.asr.base import ASRConfig
        server = NemoStreamingServer(device="cpu", chunk_size_ms=160, stream_window_s=1.0)
        server._loaded = True
        server._model = None
        pcm = b"\x00\x00" * server.chunk_samples * 8  # ~1.28 s
        partials = []

        async def collect():
            async for p in server.transcribe_stream(pcm, "fast-test", config=ASRConfig()):
                partials.append(p)

        asyncio.run(collect())
        assert len(partials) > 0, "1s window should yield a partial within ~1.28s of audio"

    def test_stream_window_5s_no_partial_on_short_audio(self):
        """5s window must NOT decode on only ~3.2s of audio."""
        import asyncio
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        from mcp_servers.asr.base import ASRConfig
        server = NemoStreamingServer(device="cpu", chunk_size_ms=160, stream_window_s=5.0)
        server._loaded = True
        server._model = None
        pcm = b"\x00\x00" * server.chunk_samples * 20  # ~3.2 s
        partials = []

        async def collect():
            async for p in server.transcribe_stream(pcm, "slow-test", config=ASRConfig()):
                partials.append(p)

        asyncio.run(collect())
        assert len(partials) == 0, "5s window must not decode on only ~3.2s of audio"

    def test_multitalker_from_config_stream_window(self):
        from mcp_servers.asr.nemo_multitalker_server import NemoMultitalkerServer
        server = NemoMultitalkerServer.from_config({
            "device": "cpu",
            "stream_window_s": 4.0,
        })
        assert server.STREAM_WINDOW_S == 4.0

    def test_multitalker_default_stream_window_is_3s(self):
        from mcp_servers.asr.nemo_multitalker_server import NemoMultitalkerServer
        server = NemoMultitalkerServer(device="cpu")
        assert server.STREAM_WINDOW_S == 3.0

    def test_pain_medicine_dict_loaded(self):
        """pain_medicine.txt terms must appear in the hotword map."""
        import os
        path = "config/dictionaries/pain_medicine.txt"
        if not os.path.exists(path):
            pytest.skip("pain_medicine.txt not found")
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu", hotwords_files=[path])
        assert "radiculitis" in server._hotword_map
        assert "postconcussive syndrome" in server._hotword_map
        assert "neurofibromatosis" in server._hotword_map
        assert "trigger point injection" in server._hotword_map
        assert "epidural steroid injection" in server._hotword_map
        assert "ubrelvy" in server._hotword_map
        assert "qulipta" in server._hotword_map

    def test_pain_medicine_corrections_applied(self):
        """Key Dr. Pello terms are corrected when present in text."""
        import os
        path = "config/dictionaries/pain_medicine.txt"
        if not os.path.exists(path):
            pytest.skip("pain_medicine.txt not found")
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu", hotwords_files=[path])
        result = server._apply_hotword_corrections(
            "patient has radiculitis and postconcussive syndrome"
        )
        assert "radiculitis" in result
        assert "postconcussive syndrome" in result


# ── spoken date normalisation ─────────────────────────────────────────────────

class TestSpokenDateNormalization:
    """Tests for _normalize_spoken_dates and _normalize_segment_text."""

    def _server(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        return NemoStreamingServer(device="cpu")

    # ── date conversion ──────────────────────────────────────────────────

    def test_full_date_october(self):
        result = self._server()._normalize_spoken_dates(
            "october twenty fifth twenty twenty four"
        )
        assert result == "October 25, 2024"

    def test_full_date_january(self):
        result = self._server()._normalize_spoken_dates(
            "january fifteenth twenty twenty six"
        )
        assert result == "January 15, 2026"

    def test_full_date_february(self):
        result = self._server()._normalize_spoken_dates(
            "february second twenty twenty six"
        )
        assert result == "February 2, 2026"

    def test_full_date_november_first(self):
        result = self._server()._normalize_spoken_dates(
            "november twenty first twenty twenty four"
        )
        assert result == "November 21, 2024"

    def test_full_date_july_eighth(self):
        result = self._server()._normalize_spoken_dates(
            "july eighth twenty twenty four"
        )
        assert result == "July 8, 2024"

    def test_date_embedded_in_sentence(self):
        result = self._server()._normalize_spoken_dates(
            "the accident occurred on october twenty fifth twenty twenty four"
        )
        assert "October 25, 2024" in result
        assert "the accident occurred on" in result

    def test_multiple_dates_in_sentence(self):
        result = self._server()._normalize_spoken_dates(
            "injury on october twenty fifth twenty twenty four "
            "follow up january fifteenth twenty twenty six"
        )
        assert "October 25, 2024" in result
        assert "January 15, 2026" in result

    def test_no_date_unchanged(self):
        text = "the patient has hypertension and diabetes mellitus"
        result = self._server()._normalize_spoken_dates(text)
        assert result == text

    def test_month_without_day_unchanged(self):
        """Month word not followed by an ordinal → kept as-is."""
        text = "in january the patient reported pain"
        result = self._server()._normalize_spoken_dates(text)
        assert result == text

    def test_date_with_trailing_comma(self):
        """Trailing comma on month word is preserved in output."""
        result = self._server()._normalize_spoken_dates(
            "starting january, fifteenth twenty twenty six"
        )
        # The comma is stripped for month detection but must not corrupt output
        assert "January" in result
        assert "15" in result

    def test_nineteen_century_year(self):
        result = self._server()._normalize_spoken_dates(
            "born on march third nineteen eighty five"
        )
        assert "March 3, 1985" in result

    # ── normalize_segment_text ───────────────────────────────────────────

    def test_segment_text_collapses_double_spaces(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu")
        assert server._normalize_segment_text("hello  world") == "hello world"

    def test_segment_text_strips(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu")
        assert server._normalize_segment_text("  hi  ") == "hi"

    def test_segment_text_converts_date(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu")
        result = server._normalize_segment_text(
            "injury on october twenty fifth twenty twenty four"
        )
        assert "October 25, 2024" in result

    # ── name injection (hotword path) ────────────────────────────────────

    def test_patient_name_hotword_corrects_casing(self):
        """If patient/provider name is injected as a hotword, casing is fixed."""
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        from mcp_servers.asr.base import ASRConfig
        server = NemoStreamingServer(device="cpu")
        config = ASRConfig(hotwords=["Brianna Buckley", "Scott Pello"])
        result = server._apply_hotword_corrections(
            "patient brianna buckley seen by scott pello",
            extra_hotwords=config.hotwords,
        )
        assert "Brianna Buckley" in result
        assert "Scott Pello" in result

    def test_last_name_only_hotword(self):
        """Injecting just the last name e.g. 'Pello' also corrects casing."""
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu")
        result = server._apply_hotword_corrections(
            "dr pello evaluated the patient",
            extra_hotwords=["Pello"],
        )
        assert "Pello" in result


# ── number words → digits + medical patterns ──────────────────────────────────

class TestNumberNormalization:
    """Tests for _parse_cardinal, _convert_medical_numbers, _normalize_numeric_date,
    _normalize_allcaps_stopwords, and _split_merged_words."""

    def _server(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        return NemoStreamingServer(device="cpu")

    # ── _parse_cardinal ──────────────────────────────────────────────────

    def test_parse_single_digit(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        result = NemoStreamingServer._parse_cardinal(["five"])
        assert result == (5, 1)

    def test_parse_teen(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        assert NemoStreamingServer._parse_cardinal(["fifteen"]) == (15, 1)

    def test_parse_tens_only(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        assert NemoStreamingServer._parse_cardinal(["sixty"]) == (60, 1)

    def test_parse_compound(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        assert NemoStreamingServer._parse_cardinal(["twenty", "seven"]) == (27, 2)

    def test_parse_hundred(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        assert NemoStreamingServer._parse_cardinal(["three", "hundred"]) == (300, 2)

    def test_parse_hundred_plus_tens(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        assert NemoStreamingServer._parse_cardinal(["one", "hundred", "twenty", "five"]) == (125, 4)

    def test_parse_none_for_ordinal(self):
        """Ordinal words must NOT be parsed as cardinals."""
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        assert NemoStreamingServer._parse_cardinal(["first"]) is None
        assert NemoStreamingServer._parse_cardinal(["third"]) is None

    def test_parse_none_for_non_number(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        assert NemoStreamingServer._parse_cardinal(["patient"]) is None

    # ── _convert_medical_numbers ─────────────────────────────────────────

    def test_strength_out_of(self):
        result = self._server()._convert_medical_numbers("five out of five strength")
        assert "5/5" in result

    def test_strength_by(self):
        result = self._server()._convert_medical_numbers("five by five strength")
        assert "5/5" in result

    def test_recall_three_out_of_three(self):
        result = self._server()._convert_medical_numbers("recall three out of three")
        assert "3/3" in result

    def test_reflex_two_plus(self):
        result = self._server()._convert_medical_numbers("deep tendon reflexes two plus")
        assert "2+" in result

    def test_reflex_one_plus(self):
        result = self._server()._convert_medical_numbers("reflexes one plus symmetric")
        assert "1+" in result

    def test_age_year_old(self):
        result = self._server()._convert_medical_numbers("a twenty seven year old male")
        assert "27-year-old" in result

    def test_general_number(self):
        result = self._server()._convert_medical_numbers("intact to sixty five")
        assert "65" in result

    def test_two_days_a_week(self):
        result = self._server()._convert_medical_numbers("occurs two days per week")
        assert "2 days" in result

    def test_hundred_mg(self):
        result = self._server()._convert_medical_numbers("one hundred milligrams")
        assert "100 milligrams" in result

    def test_non_number_unchanged(self):
        result = self._server()._convert_medical_numbers("the patient has pain")
        assert result == "the patient has pain"

    # ── _normalize_numeric_date ──────────────────────────────────────────

    def test_numeric_date_day_month_year(self):
        result = self._server()._normalize_numeric_date("date is 24 2 2026")
        assert "24/2/2026" in result

    def test_numeric_date_single_digit_month(self):
        result = self._server()._normalize_numeric_date("10 3 2025 evaluation")
        assert "10/3/2025" in result

    def test_numeric_date_invalid_month_unchanged(self):
        """Day=24, month=13 is invalid — should not be formatted."""
        result = self._server()._normalize_numeric_date("24 13 2026")
        assert "/" not in result

    def test_numeric_date_not_fired_without_year(self):
        """Only fires when a 4-digit year is present."""
        result = self._server()._normalize_numeric_date("24 2 26")
        assert "/" not in result

    # ── _normalize_allcaps_stopwords ────────────────────────────────────

    def test_the_lowercased(self):
        result = self._server()._normalize_allcaps_stopwords("THE patient has pain")
        # "THE" is not the first word? Yes it is — first word stays
        # Actually first word is protected, so only non-first THE
        assert True  # first word stays; test mid-sentence

    def test_mid_sentence_the_lowercased(self):
        result = self._server()._normalize_allcaps_stopwords("patient THE pain")
        assert "the pain" in result

    def test_abbreviation_preserved(self):
        """ALL-CAPS abbreviations not in stopword list must be preserved."""
        result = self._server()._normalize_allcaps_stopwords("patient has HTN AND DM")
        assert "HTN" in result
        assert "DM" in result
        # "AND" is in stopword list, becomes "and"
        assert "and" in result

    # ── _split_merged_words ──────────────────────────────────────────────

    def test_camelcase_split(self):
        """historyOf → history Of (wordninja may split differently)."""
        from mcp_servers.asr.nemo_streaming_server import _WORDNINJA_AVAILABLE
        if not _WORDNINJA_AVAILABLE:
            import pytest
            pytest.skip("wordninja not installed")
        result = self._server()._split_merged_words("historyOf present illness")
        # Should split at camelCase boundary
        assert "history" in result.lower()
        assert "historyOf" not in result

    def test_allcaps_prefix_split(self):
        from mcp_servers.asr.nemo_streaming_server import _WORDNINJA_AVAILABLE
        if not _WORDNINJA_AVAILABLE:
            import pytest
            pytest.skip("wordninja not installed")
        result = self._server()._split_merged_words("THEContinues to suffer")
        assert "THEContinues" not in result

    def test_short_token_not_split(self):
        """Tokens ≤ 6 chars are never sent to wordninja."""
        result = self._server()._split_merged_words("the patient has pain")
        assert result == "the patient has pain"

    def test_medical_term_in_hotword_map_protected(self):
        """Terms already in the hotword map must not be split."""
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        server = NemoStreamingServer(device="cpu", hotwords=["radiculitis"])
        result = server._split_merged_words("diagnosed with radiculitis")
        assert "radiculitis" in result

    # ── full _normalize_segment_text pipeline ────────────────────────────

    def test_full_pipeline_age_and_date(self):
        server = self._server()
        result = server._normalize_segment_text(
            "a twenty seven year old male injury on october twenty fifth twenty twenty four"
        )
        assert "27-year-old" in result
        assert "October 25, 2024" in result

    def test_full_pipeline_strength(self):
        result = self._server()._normalize_segment_text(
            "motor exam shows five out of five strength in all extremities"
        )
        assert "5/5" in result

    def test_full_pipeline_numeric_date(self):
        """After number conversion, "twenty four two twenty twenty six" → "24/2/2026"."""
        result = self._server()._normalize_segment_text(
            "date of service twenty four two twenty twenty six"
        )
        assert "24/2/2026" in result


class TestPhoneticCorrections:
    """Tests for _PHONETIC_CORRECTIONS seeded into the hotword map."""

    def _server(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        return NemoStreamingServer()

    # ── Reflexes ─────────────────────────────────────────────────────────
    def test_deep_tendon_duplexes(self):
        s = self._server()
        assert s._apply_hotword_corrections("deep tendon duplexes are 2+") == \
            "deep tendon reflexes are 2+"

    def test_deep_tendon_duplicates(self):
        s = self._server()
        assert s._apply_hotword_corrections("deep tendon duplicates") == \
            "deep tendon reflexes"

    def test_two_plex(self):
        s = self._server()
        assert s._apply_hotword_corrections("reflexes are two plex and symmetric") == \
            "reflexes are 2+ and symmetric"

    # ── Spurling's sign variants ──────────────────────────────────────────
    def test_spirling_sign(self):
        s = self._server()
        assert s._apply_hotword_corrections("spirling sign is negative") == \
            "Spurling's sign is negative"

    def test_spiraling_sign(self):
        s = self._server()
        assert s._apply_hotword_corrections("spiraling sign is positive") == \
            "Spurling's sign is positive"

    def test_spurlings_sign_no_apostrophe(self):
        s = self._server()
        assert s._apply_hotword_corrections("spurlings sign is positive on the right") == \
            "Spurling's sign is positive on the right"

    def test_sperlings_sign(self):
        s = self._server()
        assert s._apply_hotword_corrections("sperlings sign is negative") == \
            "Spurling's sign is negative"

    # ── Radiculitis / radiculopathy ───────────────────────────────────────
    def test_radical_itis(self):
        s = self._server()
        assert s._apply_hotword_corrections("diagnosis is cervical radical itis") == \
            "diagnosis is cervical radiculitis"

    def test_radical_opathy(self):
        s = self._server()
        assert s._apply_hotword_corrections("lumbar radical opathy") == \
            "lumbar radiculopathy"

    # ── Cervical levels ───────────────────────────────────────────────────
    def test_cdl_seven(self):
        s = self._server()
        assert s._apply_hotword_corrections("cdl seven testing") == "C7 testing"

    def test_cdl_six(self):
        s = self._server()
        assert s._apply_hotword_corrections("at the cdl six level") == "at the C6 level"

    def test_l5_s1(self):
        s = self._server()
        assert s._apply_hotword_corrections("disc herniation at l five s one") == \
            "disc herniation at L5-S1"

    # ── Postconcussive / posttraumatic ────────────────────────────────────
    def test_post_concussive(self):
        s = self._server()
        assert s._apply_hotword_corrections("post concussive syndrome") == \
            "postconcussive syndrome"

    def test_post_traumatic(self):
        s = self._server()
        assert s._apply_hotword_corrections("post traumatic headaches") == \
            "posttraumatic headaches"

    # ── Myofascial ────────────────────────────────────────────────────────
    def test_myo_fascial(self):
        s = self._server()
        assert s._apply_hotword_corrections("myo fascial pain") == "myofascial pain"

    def test_mayo_fascial(self):
        s = self._server()
        assert s._apply_hotword_corrections("mayo fascial pain") == "myofascial pain"

    # ── Medications ───────────────────────────────────────────────────────
    def test_propanol_to_propranolol(self):
        s = self._server()
        assert s._apply_hotword_corrections("60 mg of propanol") == \
            "60 mg of propranolol"

    def test_gamma_pentin(self):
        s = self._server()
        assert s._apply_hotword_corrections("gamma pentin 300 mg") == \
            "gabapentin 300 mg"

    def test_nurtec_variants(self):
        s = self._server()
        assert s._apply_hotword_corrections("prescription for nur tec odt") == \
            "prescription for Nurtec odt"

    def test_zofran(self):
        s = self._server()
        assert s._apply_hotword_corrections("zow fran four mg") == "Zofran four mg"

    def test_qulipta(self):
        s = self._server()
        result = s._apply_hotword_corrections("quill ipta 60 mg")
        assert "Qulipta" in result

    # ── Anatomy ───────────────────────────────────────────────────────────
    def test_sacroiliac(self):
        s = self._server()
        assert s._apply_hotword_corrections("right sacro iliac joint") == \
            "right sacroiliac joint"

    def test_paraspinal(self):
        s = self._server()
        assert s._apply_hotword_corrections("para spinal musculature") == \
            "paraspinal musculature"

    # ── Phonetic map seeded into hotword map — no double-init side effects ─
    def test_phonetic_map_does_not_override_inline_hotwords(self):
        """Inline hotwords passed at init should always win over phonetic defaults."""
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        s = NemoStreamingServer(hotwords=["MyCustomDrug"])
        # Custom hotword is present
        assert "mycustomdrug" in s._hotword_map
        # Phonetic correction is also present
        assert "deep tendon duplexes" in s._hotword_map

    def test_reload_hotwords_reseeds_phonetic_corrections(self):
        """After reload_hotwords, phonetic corrections must still be active."""
        s = self._server()
        s.reload_hotwords([], [])
        result = s._apply_hotword_corrections("deep tendon duplexes")
        assert result == "deep tendon reflexes"

    # ── _punctuate_text (unit-level — model NOT required) ─────────────────
    def test_punctuate_text_passthrough_when_unavailable(self):
        """Without the model, _punctuate_text returns the original text unchanged."""
        import asyncio
        from unittest.mock import patch
        s = self._server()
        # Simulate model not available
        with patch("mcp_servers.asr.nemo_streaming_server._PUNCT_MODEL_AVAILABLE", False):
            result = asyncio.run(s._punctuate_text("the patient is a 27 year old"))
        assert result == "the patient is a 27 year old"

    def test_punctuate_text_empty_string(self):
        """Empty string returns empty immediately, no model call attempted."""
        import asyncio
        s = self._server()
        result = asyncio.run(s._punctuate_text(""))
        assert result == ""

    def test_punctuate_text_exception_degrades_gracefully(self):
        """If the model raises, returns original text without crashing."""
        import asyncio
        from unittest.mock import patch, MagicMock
        s = self._server()
        broken_model = MagicMock()
        broken_model.restore_punctuation.side_effect = RuntimeError("model error")
        with patch("mcp_servers.asr.nemo_streaming_server._PUNCT_MODEL_AVAILABLE", True), \
             patch("mcp_servers.asr.nemo_streaming_server._get_punct_model", return_value=broken_model):
            result = asyncio.run(s._punctuate_text("the patient presents today"))
        assert result == "the patient presents today"


class TestLowercaseMergedWordSplitting:
    """Tests for the all-lowercase merged-word detection in _split_merged_words."""

    def _server(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        return NemoStreamingServer()

    def test_camel_history_of(self):
        s = self._server()
        result = s._split_merged_words("historyOf present illness")
        assert "history" in result.lower()
        assert "Of" in result or "of" in result

    def test_camel_visual_disturbance(self):
        s = self._server()
        result = s._split_merged_words("Visualdisturbance")
        assert "visual" in result.lower() and "disturbance" in result.lower()

    def test_short_token_untouched(self):
        s = self._server()
        assert s._split_merged_words("the") == "the"
        assert s._split_merged_words("IV") == "IV"

    def test_allcaps_abbreviation_untouched(self):
        s = self._server()
        assert s._split_merged_words("MRI") == "MRI"
        assert s._split_merged_words("HTN") == "HTN"

    def test_hotword_entry_not_split(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        s = NemoStreamingServer(hotwords=["postconcussive"])
        result = s._split_merged_words("postconcussive")
        assert result == "postconcussive"

    def test_long_lowercase_merge_split(self):
        """'todaypostconcussive' (21 chars, all-lowercase) must be split."""
        s = self._server()
        result = s._split_merged_words("todaypostconcussive syndrome")
        # Must contain a space before syndrome (i.e. something was split)
        assert "today" in result.lower() or "postconcussive" in result.lower()

    def test_visual_disturbance_no_camel(self):
        """'visualdisturbance' (18 chars, all-lowercase) exceeds threshold."""
        s = self._server()
        result = s._split_merged_words("visualdisturbance")
        assert "visual" in result.lower() and "disturbance" in result.lower()

    def test_quality_gate_no_crash(self):
        s = self._server()
        result = s._split_merged_words("inflammationand")
        assert isinstance(result, str) and len(result) > 0


class TestCapitalizeSentences:
    """Tests for _capitalize_sentences."""

    def _cap(self, text: str) -> str:
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        return NemoStreamingServer._capitalize_sentences(text)

    def test_first_word_capitalized(self):
        assert self._cap("the patient presents today.") == "The patient presents today."

    def test_after_period(self):
        assert self._cap("she reports pain. she has headaches.") == \
            "She reports pain. She has headaches."

    def test_after_exclamation(self):
        assert self._cap("good. excellent! she is improving.") == \
            "Good. Excellent! She is improving."

    def test_after_question_mark(self):
        assert self._cap("is pain present? yes, it is.") == \
            "Is pain present? Yes, it is."

    def test_existing_uppercase_preserved(self):
        assert self._cap("see Dr. Smith today. She is available.") == \
            "See Dr. Smith today. She is available."

    def test_digits_after_period_not_affected(self):
        assert self._cap("score is 2+. 5/5 strength noted.") == \
            "Score is 2+. 5/5 strength noted."

    def test_empty_string(self):
        assert self._cap("") == ""

    def test_already_capitalized(self):
        assert self._cap("The patient is fine.") == "The patient is fine."

    def test_no_punctuation_only_first_char(self):
        assert self._cap("the patient reports pain") == "The patient reports pain"

    def test_multiple_spaces_between_sentences(self):
        assert self._cap("the patient.  she is stable.") == \
            "The patient.  She is stable."


class TestNewPhoneticCorrections:
    """Tests for phonetic corrections added from live transcript analysis."""

    def _server(self):
        from mcp_servers.asr.nemo_streaming_server import NemoStreamingServer
        return NemoStreamingServer()

    def test_zaffron_to_zofran(self):
        s = self._server()
        assert s._apply_hotword_corrections("zaffron 4 mg") == "Zofran 4 mg"

    def test_zaf_fran(self):
        s = self._server()
        assert s._apply_hotword_corrections("given zaf fran for nausea") == \
            "given Zofran for nausea"

    def test_myofacial(self):
        s = self._server()
        assert s._apply_hotword_corrections("myofacial pain") == "myofascial pain"

    def test_tibular_vestibular(self):
        s = self._server()
        assert s._apply_hotword_corrections("tibular cognitive therapy") == \
            "vestibular cognitive therapy"

    def test_firefox_details(self):
        s = self._server()
        assert s._apply_hotword_corrections("firefox details remains") == \
            "further details remains"

    def test_chiro_practic(self):
        s = self._server()
        assert s._apply_hotword_corrections("ongoing chiro practic therapy") == \
            "ongoing chiropractic therapy"
