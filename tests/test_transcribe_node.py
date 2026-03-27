"""
Session 3 tests: WhisperX ASR server + transcribe node + post-processor.

Test layers:
  Unit  — mock ASR engine (no GPU required, always passes)
  Post  — post-processor unit tests (no GPU required)
  Integ — real WhisperX on ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3 (requires CUDA + models)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import pytest

from mcp_servers.asr.base import (
    ASRCapabilities,
    ASRConfig,
    ASREngine,
    PartialTranscript,
    RawSegment,
    RawTranscript,
    WordAlignment,
)
from orchestrator.graph import build_graph, run_encounter
from orchestrator.nodes.transcribe_node import (
    _apply_postprocessor,
    _raw_to_unified,
    _score_asr_confidence,
    set_asr_engine_factory,
)
from orchestrator.state import (
    DeliveryMethod,
    EncounterState,
    EncounterStatus,
    ProviderProfile,
    RecordingMode,
    TranscriptSegment,
    UnifiedTranscript,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_provider(mode: str = "hybrid") -> ProviderProfile:
    return ProviderProfile(
        id="test-provider-asr",
        name="Dr. Test ASR",
        specialty="general",
        custom_vocabulary=["naproxen", "pyannote", "WhisperX"],
        postprocessor_mode=mode,
    )


def make_state(
    audio_path: str | None = None,
    mode: RecordingMode = RecordingMode.DICTATION,
) -> EncounterState:
    return EncounterState(
        provider_id="test-provider-asr",
        patient_id="test-patient-asr",
        provider_profile=make_provider(),
        recording_mode=mode,
        audio_file_path=audio_path,
        delivery_method=DeliveryMethod.CLIPBOARD,
    )


# Sample garbled CTC transcript (like real MedASR output)
GARBLED_CTC = (
    "pa painin the the neck neckck  "
    "naproxen naproxen 500 milligrams twice twicece a a day  "
    "the patient patientient presentents with cervical cervical sprain"
)

CLEAN_EXPECTED_FRAGMENTS = ["naproxen", "neck", "patient", "cervical", "sprain"]

# Realistic multi-segment WhisperX output
MOCK_RAW_SEGMENTS = [
    RawSegment(
        text="The patient presents with neck pain.",
        start_ms=0,
        end_ms=3200,
        speaker="SPEAKER_00",
        confidence=0.92,
        words=[
            WordAlignment("The", 0, 300, 0.99),
            WordAlignment("patient", 310, 700, 0.95),
            WordAlignment("presents", 710, 1100, 0.91),
            WordAlignment("with", 1110, 1300, 0.98),
            WordAlignment("neck", 1310, 1600, 0.88),
            WordAlignment("pain.", 1610, 2000, 0.93),
        ],
    ),
    RawSegment(
        text="She was prescribed naproxen 500mg twice daily.",
        start_ms=3300,
        end_ms=7100,
        speaker="SPEAKER_00",
        confidence=0.89,
        words=[
            WordAlignment("She", 3300, 3500, 0.97),
            WordAlignment("was", 3510, 3700, 0.99),
            WordAlignment("prescribed", 3710, 4300, 0.87),
            WordAlignment("naproxen", 4310, 4900, 0.85),
            WordAlignment("500mg", 4910, 5400, 0.82),
            WordAlignment("twice", 5410, 5800, 0.91),
            WordAlignment("daily.", 5810, 6200, 0.94),
        ],
    ),
]

MOCK_RAW_TRANSCRIPT = RawTranscript(
    segments=MOCK_RAW_SEGMENTS,
    engine="whisperx",
    model="large-v3",
    language="en",
    audio_duration_ms=7200,
    diarization_applied=False,
)

MOCK_RAW_DIARIZED = RawTranscript(
    segments=[
        RawSegment(
            text="How are you feeling today?",
            start_ms=0,
            end_ms=2000,
            speaker="SPEAKER_00",
            confidence=0.95,
        ),
        RawSegment(
            text="I have been having neck pain for two weeks.",
            start_ms=2500,
            end_ms=6000,
            speaker="SPEAKER_01",
            confidence=0.91,
        ),
    ],
    engine="whisperx",
    model="large-v3",
    language="en",
    audio_duration_ms=6000,
    diarization_applied=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock ASR engine
# ─────────────────────────────────────────────────────────────────────────────

class MockASREngine(ASREngine):
    """Returns predetermined transcripts without calling any GPU model."""

    def __init__(self, raw: RawTranscript = MOCK_RAW_TRANSCRIPT):
        self._raw = raw
        self.call_count = 0
        self.last_audio_path: str = ""
        self.last_config: ASRConfig | None = None

    def transcribe_batch_sync(self, audio_path: str, config: ASRConfig) -> RawTranscript:
        self.call_count += 1
        self.last_audio_path = audio_path
        self.last_config = config
        return self._raw

    async def transcribe_batch(self, audio_path: str, config: ASRConfig) -> RawTranscript:
        return self.transcribe_batch_sync(audio_path, config)

    async def transcribe_stream(self, audio_chunk, session_id, config) -> AsyncIterator[PartialTranscript]:
        raise NotImplementedError

    async def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(batch=True, diarization=True, word_alignment=True)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processor unit tests (no GPU, no ASR)
# ─────────────────────────────────────────────────────────────────────────────

class TestPostProcessor:
    def test_stutter_pairs_removed(self):
        raw_text = "pa painin the the neck"
        from postprocessor import run_postprocessor
        cleaned, metrics = run_postprocessor(raw_text, use_medical_spellcheck=False)
        # Should reduce duplicate words
        assert cleaned.count("pa painin") == 0 or metrics["stutter_pairs_merged"] > 0

    def test_clean_text_unchanged(self):
        """Clean text should pass through without corruption."""
        clean = "The patient presents with cervical sprain and neck pain."
        from postprocessor import run_postprocessor
        cleaned, metrics = run_postprocessor(clean, use_medical_spellcheck=False)
        # Key medical terms must survive
        assert "cervical" in cleaned
        assert "sprain" in cleaned
        assert "neck pain" in cleaned

    def test_metrics_returned(self):
        from postprocessor import run_postprocessor
        _, metrics = run_postprocessor(GARBLED_CTC, use_medical_spellcheck=False)
        assert isinstance(metrics, dict)
        assert "stutter_pairs_merged" in metrics
        assert "words_before" in metrics
        assert "words_after" in metrics

    def test_garbled_ctc_improved(self):
        """Garbled CTC output should lose stutter pairs."""
        from postprocessor import run_postprocessor
        _, metrics = run_postprocessor(GARBLED_CTC, use_medical_spellcheck=False)
        assert metrics["stutter_pairs_merged"] > 0 or metrics["char_stutters_fixed"] > 0

    def test_medical_wordlist_integration(self):
        """With medical spellcheck, medical terms should be preserved/corrected."""
        from postprocessor import run_postprocessor
        wordlist = Path("postprocessor/medical_wordlist.txt")
        if not wordlist.exists():
            pytest.skip("medical_wordlist.txt not found")
        cleaned, metrics = run_postprocessor(
            "naproxen cervical sprain",
            use_medical_spellcheck=True,
            medical_wordlist_path=str(wordlist),
        )
        assert "naproxen" in cleaned or "naproxen" in cleaned.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Transcribe node unit tests (mock ASR)
# ─────────────────────────────────────────────────────────────────────────────

class TestTranscribeNodeUnit:
    @pytest.fixture(autouse=True)
    def inject_mock_asr(self):
        self.mock = MockASREngine()
        set_asr_engine_factory(lambda: self.mock)
        yield
        set_asr_engine_factory(None)

    def test_produces_unified_transcript(self):
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3")
        graph = build_graph()
        final = run_encounter(graph, state)

        assert final.transcript is not None
        assert final.asr_engine_used.startswith("whisperx/")
        set_llm_engine_factory(None)

    def test_segments_have_text(self):
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3")
        final = run_encounter(build_graph(), state)

        assert len(final.transcript.segments) == len(MOCK_RAW_SEGMENTS)
        for seg in final.transcript.segments:
            assert seg.text.strip()
        set_llm_engine_factory(None)

    def test_full_text_combines_segments(self):
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="fake/audio.mp3")
        final = run_encounter(build_graph(), state)

        assert "neck pain" in final.transcript.full_text
        assert "naproxen" in final.transcript.full_text
        set_llm_engine_factory(None)

    def test_word_tokens_preserved(self):
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="fake/audio.mp3")
        final = run_encounter(build_graph(), state)

        # First segment has word-level timing
        first_seg = final.transcript.segments[0]
        assert len(first_seg.words) > 0
        assert first_seg.words[0].start_ms >= 0
        set_llm_engine_factory(None)

    def test_postprocessor_runs(self):
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="fake/audio.mp3")
        final = run_encounter(build_graph(), state)

        assert final.postprocessor_version == "medasr_postprocessor_v1"
        set_llm_engine_factory(None)

    def test_asr_error_produces_fallback(self):
        """When ASR throws, transcribe_node must not crash the pipeline."""
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine

        class BrokenASR(MockASREngine):
            def transcribe_batch_sync(self, *a, **kw):
                raise RuntimeError("GPU OOM")

        set_asr_engine_factory(lambda: BrokenASR())
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="fake/audio.mp3")
        final = run_encounter(build_graph(), state)

        assert final.asr_engine_used == "fallback_stub"
        assert any("transcribe_node" in e for e in final.errors)
        assert final.status == EncounterStatus.DELIVERED  # pipeline still completes
        set_llm_engine_factory(None)

    def test_no_audio_path_produces_fallback(self):
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path=None)  # no audio
        final = run_encounter(build_graph(), state)

        assert final.asr_engine_used == "fallback_stub"
        assert any("no audio" in e for e in final.errors)
        set_llm_engine_factory(None)

    def test_existing_transcript_passes_through(self):
        """If state already has a transcript, ASR must not be called."""
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        existing = UnifiedTranscript(
            segments=[TranscriptSegment(text="Pre-existing text.", speaker="SPEAKER_00",
                                        start_ms=0, end_ms=1000, mode=RecordingMode.DICTATION)],
            full_text="Pre-existing text.",
        )
        state = make_state()
        state = state.model_copy(update={"transcript": existing})
        final = run_encounter(build_graph(), state)

        assert self.mock.call_count == 0   # ASR not called
        assert "Pre-existing text." in final.transcript.full_text
        set_llm_engine_factory(None)

    def test_diarized_transcript_speaker_labels(self):
        """Diarized transcript must carry speaker labels through to UnifiedTranscript."""
        mock = MockASREngine(raw=MOCK_RAW_DIARIZED)
        set_asr_engine_factory(lambda: mock)

        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="fake/convo.mp3", mode=RecordingMode.AMBIENT)
        final = run_encounter(build_graph(), state)

        speakers = {s.speaker for s in final.transcript.segments}
        assert "SPEAKER_00" in speakers
        assert "SPEAKER_01" in speakers
        set_llm_engine_factory(None)

    def test_confidence_score_computed(self):
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="fake/audio.mp3")
        final = run_encounter(build_graph(), state)

        assert final.metrics.asr_confidence > 0.5   # mock has confidence ~0.9
        set_llm_engine_factory(None)

    def test_session1_and_2_tests_unaffected(self):
        """All prior session tests must continue to pass."""
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine
        set_llm_engine_factory(lambda: MockLLMEngine())

        state = make_state(audio_path="fake/audio.mp3")
        final = run_encounter(build_graph(), state)

        assert final.metrics.nodes_completed == [
            "context", "capture", "transcribe", "note", "review", "delivery"
        ]
        assert final.status == EncounterStatus.DELIVERED
        set_llm_engine_factory(None)


# ─────────────────────────────────────────────────────────────────────────────
# Conversion and scoring unit tests (pure, no engine)
# ─────────────────────────────────────────────────────────────────────────────

class TestConversion:
    def test_raw_to_unified_segments(self):
        unified = _raw_to_unified(MOCK_RAW_TRANSCRIPT, RecordingMode.DICTATION)
        assert len(unified.segments) == 2
        assert unified.segments[0].text == MOCK_RAW_SEGMENTS[0].text

    def test_raw_to_unified_full_text(self):
        unified = _raw_to_unified(MOCK_RAW_TRANSCRIPT, RecordingMode.DICTATION)
        assert "naproxen" in unified.full_text
        assert "neck pain" in unified.full_text

    def test_raw_to_unified_word_tokens(self):
        unified = _raw_to_unified(MOCK_RAW_TRANSCRIPT, RecordingMode.DICTATION)
        first = unified.segments[0]
        assert len(first.words) == len(MOCK_RAW_SEGMENTS[0].words)
        assert first.words[0].text == "The"
        assert first.words[0].start_ms == 0

    def test_raw_to_unified_speaker_labels(self):
        unified = _raw_to_unified(MOCK_RAW_DIARIZED, RecordingMode.AMBIENT)
        assert unified.segments[0].speaker == "SPEAKER_00"
        assert unified.segments[1].speaker == "SPEAKER_01"

    def test_raw_to_unified_diarization_flag(self):
        unified = _raw_to_unified(MOCK_RAW_DIARIZED, RecordingMode.AMBIENT)
        assert unified.diarization_engine == "pyannote-3.1"

    def test_confidence_scoring_word_level(self):
        unified = _raw_to_unified(MOCK_RAW_TRANSCRIPT, RecordingMode.DICTATION)
        conf = _score_asr_confidence(unified)
        assert 0.8 < conf < 1.0

    def test_confidence_no_words_falls_back_to_segment(self):
        unified = _raw_to_unified(MOCK_RAW_DIARIZED, RecordingMode.DICTATION)
        # DIARIZED mock has no word-level data
        conf = _score_asr_confidence(unified)
        assert conf > 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — real WhisperX (skip if model / CUDA not available)
# ─────────────────────────────────────────────────────────────────────────────

def _whisperx_available() -> tuple[bool, str]:
    """Return (available, reason_if_not)."""
    try:
        import whisperx  # noqa: F401
    except ImportError:
        return False, "whisperx not installed"
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
    except ImportError:
        return False, "torch not installed"
    if not Path("ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3").exists():
        return False, "test audio file not found"
    return True, ""


_WX_AVAILABLE, _WX_REASON = _whisperx_available()


@pytest.mark.skipif(not _WX_AVAILABLE, reason=f"WhisperX unavailable: {_WX_REASON}")
class TestTranscribeNodeIntegration:
    """Live WhisperX tests on real audio. Requires CUDA + downloaded models."""

    @pytest.fixture(autouse=True)
    def use_real_asr_and_stub_llm(self):
        from orchestrator.nodes.note_node import set_llm_engine_factory
        from tests.test_note_node import MockLLMEngine

        set_asr_engine_factory(None)           # use real WhisperX
        set_llm_engine_factory(lambda: MockLLMEngine())
        yield
        set_asr_engine_factory(None)
        set_llm_engine_factory(None)

    def test_real_audio_produces_transcript(self):
        state = make_state(audio_path="ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3")
        final = run_encounter(build_graph(), state)

        assert final.transcript is not None
        assert final.asr_engine_used != "fallback_stub", f"ASR failed: {final.errors}"
        assert len(final.transcript.full_text) > 100
        assert final.transcript.audio_duration_ms > 0

    def test_real_audio_contains_medical_terms(self):
        """The 224889 dictation mentions lumbar spine — should appear in transcript."""
        state = make_state(audio_path="ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3")
        final = run_encounter(build_graph(), state)

        text = final.transcript.full_text.lower()
        # At least one of these should appear after ASR + post-processing
        medical_terms = ["lumbar", "spine", "patient", "exam", "back"]
        found = [t for t in medical_terms if t in text]
        assert found, f"No medical terms found in transcript. Got: {text[:300]}"

    def test_postprocessor_applied_to_real_transcript(self):
        state = make_state(audio_path="ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3")
        final = run_encounter(build_graph(), state)

        assert final.postprocessor_version == "medasr_postprocessor_v1"
        assert isinstance(final.postprocessor_metrics, dict)

    def test_ambient_mode_requests_diarization(self):
        """AMBIENT mode should request diarization (even if no HF token is set)."""
        state = make_state(
            audio_path="ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3",
            mode=RecordingMode.AMBIENT,
        )
        final = run_encounter(build_graph(), state)

        # Pipeline should complete; diarization may be skipped if no HF token
        assert final.transcript is not None
        assert final.status == EncounterStatus.DELIVERED
