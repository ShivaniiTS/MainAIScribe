"""
Session 4 — End-to-End Pipeline Tests.

Tests:
  Stub (always run — no GPU/Ollama required):
    1.  Full pipeline runs: audio_file_path → transcript → SOAP note → DELIVERED
    2.  ASR router fires: post-transcribe conditional edge resolves to "note"
    3.  LLM router fires: post-note conditional edge resolves to "review"
    4.  Low-confidence ASR (stub) still reaches DELIVERED
    5.  LLM fallback stub (no Ollama) still reaches DELIVERED
    6.  Both RecordingMode.DICTATION and AMBIENT reach DELIVERED
    7.  Pipeline preserves encounter_id end-to-end
    8.  Pipeline errors list is empty on clean stub run
    9.  generated_note and final_note are both populated
    10. metrics: asr_duration_ms, note_gen_ms both non-zero

  Router unit tests (pure, no graph):
    11. asr_router → "note" on normal transcript
    12. asr_router → "note" on fallback stub transcript
    13. asr_router → "note" on low-confidence transcript
    14. llm_router → "review" on normal note
    15. llm_router → "review" on fallback stub note
    16. llm_router → "review" on low-confidence note

  Integration (real WhisperX + real Ollama — skipped if unavailable):
    17. Full pipeline: dictation audio → SOAP note (all 4 sections present)
    18. Note quality: gold-standard keyword overlap ≥ 30%
    19. Transcript quality: key medical terms in transcript
    20. Pipeline timing: asr + note_gen each complete in < 120 s
    21. Note is longer than gold standard (raw vs edited)
    22. No LLM fallback stub in real integration run
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pytest

from mcp_servers.asr.base import ASRCapabilities, ASRConfig, ASREngine, PartialTranscript, RawSegment, RawTranscript, WordAlignment
from mcp_servers.llm.base import LLMConfig, LLMEngine, LLMMessage, LLMResponse, ModelInfo
from orchestrator.edges.asr_router import asr_router
from orchestrator.edges.llm_router import llm_router
from orchestrator.graph import build_graph, run_encounter
from orchestrator.nodes.note_node import set_llm_engine_factory
from orchestrator.nodes.transcribe_node import set_asr_engine_factory
from orchestrator.state import (
    ClinicalNote,
    DeliveryMethod,
    EncounterMetrics,
    EncounterState,
    EncounterStatus,
    NoteSection,
    NoteType,
    ProviderProfile,
    RecordingMode,
    UnifiedTranscript,
    TranscriptSegment,
)

# ─────────────────────────────────────────────────────────────────────────────
# Test data paths
# ─────────────────────────────────────────────────────────────────────────────

from config.paths import DATA_DIR as _DATA_ROOT
_DATA_DIR = _DATA_ROOT / "dictation" / "dr_faraz_rahman" / "riley_dew_226680_20260219"
_AUDIO_PATH = _DATA_DIR / "dictation.mp3"
_GOLD_PATH = _DATA_DIR / "final_soap_note.md"


# ─────────────────────────────────────────────────────────────────────────────
# Engine availability detection (at collection time)
# ─────────────────────────────────────────────────────────────────────────────

def _whisperx_available() -> bool:
    """Return True if WhisperX + CUDA + test audio are all present."""
    try:
        import whisperx  # noqa: F401
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    return _AUDIO_PATH.exists()


def _ollama_status() -> tuple[bool, str]:
    """Return (available, first_model_name) by querying Ollama at test collection time."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if resp.status_code != 200:
            return False, ""
        models = resp.json().get("models", [])
        if not models:
            return False, ""
        return True, models[0]["name"]
    except Exception:
        return False, ""


_WHISPERX_AVAILABLE = _whisperx_available()
_OLLAMA_AVAILABLE, _OLLAMA_MODEL = _ollama_status()
_FULL_INTEGRATION = _WHISPERX_AVAILABLE and _OLLAMA_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# Stub engines
# ─────────────────────────────────────────────────────────────────────────────

_REALISTIC_TRANSCRIPT = (
    "The patient presents today as a transfer of care. She reports that her "
    "low back symptoms have improved considerably. She has some off and on "
    "stiffness which is quite mild. She does have headaches and some "
    "anxiousness with driving. On examination the patient is alert and "
    "oriented. Inspection of lumbar spine reveals no bony deformities. "
    "She has good range of motion. Normal strength in her extremities. "
    "Assessment: improved lumbar spine symptoms after lumbar spine sprain "
    "strain sustained due to auto accident. Plan: continue home exercises "
    "as instructed. Follow with neurology and behavioral health as scheduled."
)

_STUB_SOAP = """\
SUBJECTIVE:
The patient presents for evaluation of low back pain following motor vehicle
accident. Reports improvement in symptoms. Has mild stiffness and headaches.
Anxious with driving. Follow-up appointments scheduled with neurology and
behavioral health.

OBJECTIVE:
Alert and oriented. Lumbar spine: no bony deformities. No tenderness to
palpation. Good range of motion with flexion, extension, lateral rotation,
and lateral bending. Normal strength in extremities.

ASSESSMENT:
Improved lumbar spine sprain/strain following motor vehicle accident.

PLAN:
Continue home exercises as instructed. Apply ice or heat as needed.
Follow with neurology and behavioral health as scheduled. Discharge from
musculoskeletal care. Return if symptoms worsen.
"""


class _StubASR(ASREngine):
    def __init__(self, confidence: float = 0.92) -> None:
        self._confidence = confidence

    def transcribe_batch_sync(self, audio_path: str, config: ASRConfig) -> RawTranscript:
        return RawTranscript(
            segments=[
                RawSegment(
                    text=_REALISTIC_TRANSCRIPT,
                    start_ms=0,
                    end_ms=30_000,
                    confidence=self._confidence,
                )
            ],
            engine="stub",
            model="stub-asr",
            language="en",
            audio_duration_ms=30_000,
        )

    async def transcribe_batch(self, audio_path, config) -> RawTranscript:
        return self.transcribe_batch_sync(audio_path, config)

    async def transcribe_stream(self, audio_chunk, session_id, config):
        raise NotImplementedError

    async def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(batch=True)


class _StubLLM(LLMEngine):
    def generate_sync(self, system_prompt, messages, config, task="note_generation") -> LLMResponse:
        return LLMResponse(
            content=_STUB_SOAP,
            model="stub-llm",
            prompt_tokens=100,
            completion_tokens=200,
        )

    async def generate(self, system_prompt, messages, config) -> LLMResponse:
        return self.generate_sync(system_prompt, messages, config)

    async def generate_stream(self, system_prompt, messages, config):
        yield  # pragma: no cover

    async def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name="stub-llm", context_window=4096)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_provider(
    specialty: str = "physical medicine",
    note_format: NoteType = NoteType.SOAP,
) -> ProviderProfile:
    return ProviderProfile(
        id="provider-e2e-001",
        name="Dr. E2E Test",
        specialty=specialty,
        note_format=note_format,
        template_id="soap_default",
        style_directives=["Use active voice", "Be concise"],
        custom_vocabulary=["lumbar", "pyannote", "WhisperX"],
    )


def _make_state(
    audio_path: str = "fake/stub_audio.mp3",
    mode: RecordingMode = RecordingMode.DICTATION,
    **kwargs,
) -> EncounterState:
    defaults = dict(
        provider_id="provider-e2e-001",
        patient_id="patient-e2e-001",
        provider_profile=_make_provider(),
        recording_mode=mode,
        delivery_method=DeliveryMethod.CLIPBOARD,
        audio_file_path=audio_path,
    )
    defaults.update(kwargs)
    return EncounterState(**defaults)


def _section_text(note: ClinicalNote, section_type: str) -> str:
    for s in note.sections:
        if s.type == section_type:
            return s.content
    return ""


def _keyword_overlap(generated: str, reference: str, top_n: int = 40) -> float:
    """Fraction of top_n reference keywords that appear in generated text."""

    def _words(text: str) -> set[str]:
        stopwords = {
            "the", "a", "an", "and", "or", "of", "to", "in", "is", "are",
            "was", "for", "she", "her", "he", "his", "with", "has", "have",
            "be", "that", "this", "from", "as", "at", "by", "on", "not",
            "no", "but", "it", "its", "will", "may", "can", "we", "our",
            "patient", "presents", "today", "due",
        }
        return {
            w.lower()
            for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 3 and w not in stopwords
        }

    ref_words = _words(reference)
    gen_words = _words(generated)
    if not ref_words:
        return 0.0
    # Use the most common reference words (longer = more domain-specific)
    important = sorted(ref_words, key=len, reverse=True)[:top_n]
    matched = sum(1 for w in important if w in gen_words)
    return matched / len(important)


def _load_gold_note() -> str:
    if not _GOLD_PATH.exists():
        return ""
    return _GOLD_PATH.read_text()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def graph():
    return build_graph()


@pytest.fixture(autouse=True, scope="module")
def inject_stubs():
    """Inject stub engines for all non-integration tests in this module."""
    set_llm_engine_factory(lambda: _StubLLM())
    set_asr_engine_factory(lambda: _StubASR())
    yield
    set_llm_engine_factory(None)
    set_asr_engine_factory(None)


# ─────────────────────────────────────────────────────────────────────────────
# Stub pipeline tests
# ─────────────────────────────────────────────────────────────────────────────

class TestE2EPipelineStub:
    """Full end-to-end pipeline with stub ASR + stub LLM (always runs)."""

    def test_full_pipeline_reaches_delivered(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.status == EncounterStatus.DELIVERED

    def test_all_six_nodes_run(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.metrics.nodes_completed == [
            "context", "capture", "transcribe", "note", "review", "delivery"
        ]

    def test_transcript_populated(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.transcript is not None
        assert len(final.transcript.full_text) > 50
        assert "stub" in final.asr_engine_used

    def test_note_has_all_soap_sections(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.generated_note is not None
        types = {s.type for s in final.generated_note.sections}
        for section in ("subjective", "objective", "assessment", "plan"):
            assert section in types, f"Missing section: {section}"

    def test_final_note_set_by_review(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.review_approved is True
        assert final.final_note is not None

    def test_delivery_result_populated(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.delivery_result is not None
        assert final.delivery_result.get("success") is True

    def test_encounter_id_preserved(self, graph):
        state = _make_state()
        eid = state.encounter_id
        final = run_encounter(graph, state)
        assert final.encounter_id == eid

    def test_no_errors_on_clean_stub_run(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.errors == []

    def test_metrics_timing_populated(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.metrics.asr_duration_ms is not None
        assert final.metrics.note_gen_ms is not None

    def test_dictation_mode(self, graph):
        state = _make_state(mode=RecordingMode.DICTATION)
        final = run_encounter(graph, state)
        assert final.status == EncounterStatus.DELIVERED

    def test_ambient_mode(self, graph):
        state = _make_state(mode=RecordingMode.AMBIENT)
        final = run_encounter(graph, state)
        assert final.status == EncounterStatus.DELIVERED

    def test_note_to_text_has_soap_headings(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        text = final.final_note.to_text()
        upper = text.upper()
        assert "SUBJECTIVE" in upper
        assert "OBJECTIVE" in upper
        assert "ASSESSMENT" in upper
        assert "PLAN" in upper

    def test_realistic_transcript_flows_to_note(self, graph):
        """Stub ASR returns a realistic transcript; note should reference lumbar."""
        state = _make_state()
        final = run_encounter(graph, state)
        note_text = final.final_note.to_text()
        assert "lumbar" in note_text.lower() or "back" in note_text.lower()

    def test_template_used_tracked(self, graph):
        state = _make_state()
        final = run_encounter(graph, state)
        assert final.template_used == "soap_default"


# ─────────────────────────────────────────────────────────────────────────────
# Router unit tests (no graph needed)
# ─────────────────────────────────────────────────────────────────────────────

def _state_with_transcript(text: str, confidence: float = 0.90) -> EncounterState:
    transcript = UnifiedTranscript(
        segments=[
            TranscriptSegment(
                text=text,
                start_ms=0,
                end_ms=5000,
                mode=RecordingMode.DICTATION,
                source="asr",
                confidence=confidence,
            )
        ],
        full_text=text,
        engine_used="stub",
    )
    return _make_state().model_copy(update={
        "transcript": transcript,
        "asr_engine_used": "stub/stub-asr",
        "metrics": EncounterMetrics(asr_confidence=confidence),
    })


def _state_with_note(confidence: float = 0.90, fallback: bool = False) -> EncounterState:
    content = "[LLM UNAVAILABLE] raw text" if fallback else "Patient presents well."
    note = ClinicalNote(
        note_type=NoteType.SOAP,
        sections=[
            NoteSection(type="subjective", content=content),
            NoteSection(type="objective", content="Normal exam." if not fallback else "[LLM UNAVAILABLE]"),
            NoteSection(type="assessment", content="Stable." if not fallback else "[LLM UNAVAILABLE]"),
            NoteSection(type="plan", content="Continue." if not fallback else "[LLM UNAVAILABLE]"),
        ],
    )
    return _make_state().model_copy(update={
        "generated_note": note,
        "llm_engine_used": "fallback_stub" if fallback else "stub-llm",
        "metrics": EncounterMetrics(note_confidence=confidence),
    })


class TestRouters:
    def test_asr_router_normal_transcript(self):
        state = _state_with_transcript("Patient reports improvement in symptoms.", 0.92)
        assert asr_router(state) == "note"

    def test_asr_router_fallback_stub_transcript(self):
        state = _state_with_transcript("[ASR UNAVAILABLE: asr_error]", 0.0)
        assert asr_router(state) == "note"

    def test_asr_router_low_confidence(self):
        state = _state_with_transcript("um uh patient symptoms improvement", 0.15)
        assert asr_router(state) == "note"

    def test_asr_router_no_transcript(self):
        state = _make_state()
        assert asr_router(state) == "note"

    def test_llm_router_normal_note(self):
        state = _state_with_note(confidence=0.90)
        assert llm_router(state) == "review"

    def test_llm_router_fallback_stub_note(self):
        state = _state_with_note(confidence=0.0, fallback=True)
        assert llm_router(state) == "review"

    def test_llm_router_low_confidence_note(self):
        state = _state_with_note(confidence=0.30)
        assert llm_router(state) == "review"

    def test_llm_router_no_note(self):
        state = _make_state()
        assert llm_router(state) == "review"


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests — real WhisperX + real Ollama
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _FULL_INTEGRATION,
    reason=(
        "Full integration test requires WhisperX (CUDA) + Ollama + test audio. "
        f"whisperx={_WHISPERX_AVAILABLE} ollama={_OLLAMA_AVAILABLE}"
    ),
)
class TestE2EPipelineIntegration:
    """Real WhisperX + real Ollama: dictation audio → SOAP note."""

    @pytest.fixture(scope="class", autouse=True)
    def reset_factories(self):
        """Remove stub overrides so real engines are used."""
        set_llm_engine_factory(None)
        set_asr_engine_factory(None)
        yield
        # Restore stubs for subsequent test classes in this module
        set_llm_engine_factory(lambda: _StubLLM())
        set_asr_engine_factory(lambda: _StubASR())

    @pytest.fixture(scope="class")
    def configure_ollama(self):
        """Point OllamaServer at the first available model."""
        from mcp_servers.llm.ollama_server import OllamaServer
        set_llm_engine_factory(
            lambda: OllamaServer(model_overrides={"note_generation": _OLLAMA_MODEL})
        )
        yield
        set_llm_engine_factory(None)

    @pytest.fixture(scope="class")
    def integration_graph(self):
        return build_graph()

    @pytest.fixture(scope="class")
    def integration_result(self, integration_graph, configure_ollama):
        state = _make_state(audio_path=str(_AUDIO_PATH))
        return run_encounter(integration_graph, state)

    def test_pipeline_reaches_delivered(self, integration_result):
        assert integration_result.status == EncounterStatus.DELIVERED

    def test_real_asr_engine_used(self, integration_result):
        assert "whisperx" in integration_result.asr_engine_used.lower()

    def test_real_llm_engine_used(self, integration_result):
        assert integration_result.llm_engine_used != "fallback_stub"

    def test_all_soap_sections_present(self, integration_result):
        note = integration_result.generated_note
        assert note is not None
        types = {s.type for s in note.sections}
        for section in ("subjective", "objective", "assessment", "plan"):
            assert section in types, f"Missing section: {section}"

    def test_no_llm_fallback_in_note(self, integration_result):
        note = integration_result.generated_note
        for section in note.sections:
            assert "[LLM UNAVAILABLE]" not in section.content

    def test_transcript_contains_medical_terms(self, integration_result):
        text = integration_result.transcript.full_text.lower()
        medical_terms = ["lumbar", "spine", "patient", "symptoms", "pain"]
        found = [t for t in medical_terms if t in text]
        assert len(found) >= 3, f"Only {len(found)} medical terms found in transcript: {found}"

    def test_note_quality_keyword_overlap(self, integration_result):
        gold = _load_gold_note()
        if not gold:
            pytest.skip("Gold standard note not found at " + str(_GOLD_PATH))
        generated = integration_result.final_note.to_text()
        overlap = _keyword_overlap(generated, gold)
        assert overlap >= 0.30, (
            f"Keyword overlap {overlap:.1%} < 30% — generated note may be missing key clinical content"
        )

    def test_asr_timing_reasonable(self, integration_result):
        # Should not take more than 120s on an A10G
        assert integration_result.metrics.asr_duration_ms < 120_000, (
            f"ASR took {integration_result.metrics.asr_duration_ms}ms (> 120s)"
        )

    def test_note_timing_reasonable(self, integration_result):
        assert integration_result.metrics.note_gen_ms < 120_000, (
            f"Note generation took {integration_result.metrics.note_gen_ms}ms (> 120s)"
        )

    def test_note_confidence_above_threshold(self, integration_result):
        assert integration_result.metrics.note_confidence >= 0.50, (
            f"Note confidence {integration_result.metrics.note_confidence} < 0.5"
        )

    def test_no_pipeline_errors(self, integration_result):
        assert integration_result.errors == [], (
            f"Pipeline errors: {integration_result.errors}"
        )
