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
