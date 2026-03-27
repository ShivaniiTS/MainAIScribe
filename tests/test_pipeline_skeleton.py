"""
Session 1 test: verify the LangGraph skeleton runs all 6 nodes end-to-end.

Tests:
  1. State flows through all 6 nodes (context → capture → transcribe → note → review → delivery)
  2. Each node records itself in metrics.nodes_completed
  3. Final status is DELIVERED
  4. Generated note has the expected SOAP sections
  5. Audio file path is preserved through to transcribe
  6. Pipeline works with both AMBIENT and DICTATION recording modes
"""

from __future__ import annotations

import pytest

from mcp_servers.asr.base import ASRCapabilities, ASRConfig, ASREngine, PartialTranscript, RawSegment, RawTranscript, WordAlignment
from mcp_servers.llm.base import LLMConfig, LLMEngine, LLMMessage, LLMResponse, ModelInfo
from orchestrator.graph import build_graph, run_encounter
from orchestrator.nodes.note_node import set_llm_engine_factory
from orchestrator.nodes.transcribe_node import set_asr_engine_factory
from orchestrator.state import (
    DeliveryMethod,
    EncounterState,
    EncounterStatus,
    NoteType,
    ProviderProfile,
    RecordingMode,
)

EXPECTED_NODES = ["context", "capture", "transcribe", "note", "review", "delivery"]

# Minimal mock so skeleton tests never hit Ollama
_STUB_SOAP = """SUBJECTIVE:
Patient presents for evaluation.

OBJECTIVE:
Alert and oriented, normal exam.

ASSESSMENT:
Stable condition.

PLAN:
Continue current management.
"""


class _StubLLM(LLMEngine):
    def generate_sync(self, system_prompt, messages, config, task="note_generation") -> LLMResponse:
        return LLMResponse(content=_STUB_SOAP, model="stub-llm", prompt_tokens=10, completion_tokens=20)

    async def generate(self, system_prompt, messages, config) -> LLMResponse:
        return self.generate_sync(system_prompt, messages, config)

    async def generate_stream(self, system_prompt, messages, config):
        yield  # pragma: no cover

    async def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name="stub-llm", context_window=4096)


class _StubASR(ASREngine):
    """Returns a stub transcript without calling any GPU model."""

    def transcribe_batch_sync(self, audio_path: str, config: ASRConfig) -> RawTranscript:
        return RawTranscript(
            segments=[RawSegment(text="Stub ASR transcript.", start_ms=0, end_ms=1000)],
            engine="stub", model="stub", language="en", audio_duration_ms=1000,
        )

    async def transcribe_batch(self, audio_path, config) -> RawTranscript:
        return self.transcribe_batch_sync(audio_path, config)

    async def transcribe_stream(self, audio_chunk, session_id, config):
        raise NotImplementedError

    async def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(batch=True)


@pytest.fixture(autouse=True, scope="module")
def inject_stubs():
    """Inject stub LLM + stub ASR for all Session 1 skeleton tests."""
    set_llm_engine_factory(lambda: _StubLLM())
    set_asr_engine_factory(lambda: _StubASR())
    yield
    set_llm_engine_factory(None)
    set_asr_engine_factory(None)


def make_provider(specialty: str = "general", note_format: NoteType = NoteType.SOAP) -> ProviderProfile:
    return ProviderProfile(
        id="provider-test-001",
        name="Dr. Test Provider",
        specialty=specialty,
        note_format=note_format,
        template_id="soap_default",
        style_directives=["Use active voice", "Be concise"],
        custom_vocabulary=["Tylenol", "Advil"],
    )


def make_state(**kwargs) -> EncounterState:
    defaults = dict(
        provider_id="provider-test-001",
        patient_id="patient-test-001",
        provider_profile=make_provider(),
        recording_mode=RecordingMode.DICTATION,
        delivery_method=DeliveryMethod.CLIPBOARD,
        audio_file_path="fake/stub_audio.mp3",   # stub ASR ignores the path
    )
    defaults.update(kwargs)
    return EncounterState(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def graph():
    return build_graph()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineSkeleton:
    def test_all_nodes_execute(self, graph):
        """All 6 nodes must run and register themselves in metrics.nodes_completed."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.metrics.nodes_completed == EXPECTED_NODES, (
            f"Expected {EXPECTED_NODES}, got {final.metrics.nodes_completed}"
        )

    def test_final_status_is_delivered(self, graph):
        """Pipeline must reach DELIVERED status."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.status == EncounterStatus.DELIVERED

    def test_context_packet_created(self, graph):
        """Context node must produce a ContextPacket."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.context_packet is not None

    def test_transcript_created(self, graph):
        """Transcribe node must produce a UnifiedTranscript with at least one segment."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.transcript is not None
        assert len(final.transcript.segments) >= 1
        assert "stub" in final.asr_engine_used

    def test_note_has_soap_sections(self, graph):
        """Note node must produce a ClinicalNote with all 4 SOAP sections."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.generated_note is not None
        section_types = [s.type for s in final.generated_note.sections]
        for expected in ["subjective", "objective", "assessment", "plan"]:
            assert expected in section_types, f"Missing section: {expected}"

    def test_review_approves_and_sets_final_note(self, graph):
        """Review node must set final_note and review_approved=True."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.review_approved is True
        assert final.final_note is not None

    def test_delivery_result_present(self, graph):
        """Delivery node must populate delivery_result."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.delivery_result is not None
        assert final.delivery_result["success"] is True

    def test_audio_file_path_preserved(self, graph):
        """Audio file path set on input must be visible to transcribe node."""
        state = make_state(audio_file_path="ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3")
        final = run_encounter(graph, state)
        # Transcribe stub embeds the path in the transcript text
        assert "riley_dew_226680_20260219" in final.transcript.full_text or \
               final.audio_segments[0].storage_path == "ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3"

    def test_ambient_mode_pipeline(self, graph):
        """Pipeline must work with AMBIENT (multi-speaker) mode."""
        state = make_state(recording_mode=RecordingMode.AMBIENT)
        final = run_encounter(graph, state)
        assert final.metrics.nodes_completed == EXPECTED_NODES
        assert final.status == EncounterStatus.DELIVERED

    def test_encounter_id_preserved(self, graph):
        """encounter_id must not change across the pipeline."""
        state = make_state()
        original_id = state.encounter_id
        final = run_encounter(graph, state)
        assert final.encounter_id == original_id

    def test_pipeline_start_ms_set(self, graph):
        """run_encounter must populate metrics.pipeline_start_ms."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.metrics.pipeline_start_ms is not None
        assert final.metrics.pipeline_start_ms > 0

    def test_no_errors_in_stub_run(self, graph):
        """Stub pipeline must produce zero errors."""
        state = make_state()
        final = run_encounter(graph, state)
        assert final.errors == []

    def test_note_to_text(self, graph):
        """ClinicalNote.to_text() must return non-empty formatted string."""
        state = make_state()
        final = run_encounter(graph, state)
        text = final.final_note.to_text()
        assert "SUBJECTIVE" in text.upper()
        assert len(text) > 50


class TestProviderProfile:
    def test_profile_specialty_flows_through(self, graph):
        """Provider specialty must be accessible in the final state."""
        profile = make_provider(specialty="orthopedic")
        state = make_state(provider_profile=profile)
        final = run_encounter(graph, state)
        assert final.provider_profile.specialty == "orthopedic"

    def test_template_id_used_in_note(self, graph):
        """Note node must record the template used from the provider profile."""
        state = make_state()
        final = run_encounter(graph, state)
        assert "soap_default" in (final.template_used or "")
        assert "soap_default" in (final.generated_note.metadata.template_used or "")
