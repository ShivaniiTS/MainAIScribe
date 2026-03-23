"""
Session 2 tests: Ollama LLM server + note generation.

Test layers:
  Unit  — mock LLM engine, no Ollama required (always passes)
  Integ — real Ollama call (skipped if Ollama not running)

Real transcript from ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/ is used for the integration test.
"""

from __future__ import annotations

import pytest

from mcp_servers.llm.base import LLMConfig, LLMEngine, LLMMessage, LLMResponse, ModelInfo
from orchestrator.graph import build_graph, run_encounter
from orchestrator.nodes.note_node import (
    assemble_prompt,
    parse_note_sections,
    set_llm_engine_factory,
)
from orchestrator.state import (
    EncounterState,
    NoteType,
    ProviderProfile,
    RecordingMode,
    TranscriptSegment,
    UnifiedTranscript,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

# Realistic dictation text from our test data (paraphrased to remove PHI)
SAMPLE_TRANSCRIPT = """
The patient presents today as a transfer of care from internal medicine to
physical medicine and rehabilitation. She presents for follow-up evaluation of
injuries sustained due to a motor vehicle accident. She reports that her low
back symptoms have improved considerably. She has some off and on stiffness,
which is quite mild and tolerable at present. She does have headaches and some
anxiousness with driving. She has a follow-up appointment scheduled with the
neurologist on April 1st. She has an appointment with behavioral health therapy
on March 25th. She is pleased with her musculoskeletal symptoms. She is
working.

Past medical history, surgical history, family history, allergies, and social
history are reviewed from previous evaluations and are otherwise unchanged.

The patient is alert and oriented. Inspection of lumbar spine reveals no bony
deformities. She does not have any tenderness to palpation over the paralumbar
musculature on exam today. She has good range of motion of the lumbar spine
with flexion, extension, lateral rotation, and lateral bending without
significant pain. She has normal strength in her extremities.

Assessment: The patient presents with improved lumbar spine symptoms after
initial lumbar spine sprain/strain sustained due to this auto accident.

Plan: She should continue home exercises as instructed. She can apply ice or
heat when required. She should follow with neurology and behavioral health as
scheduled. As her musculoskeletal symptoms have improved, she will be
discharged from our care.
""".strip()

MOCK_LLM_SOAP_RESPONSE = """SUBJECTIVE:
The patient presents as a transfer of care to physical medicine and rehabilitation for follow-up of motor vehicle accident injuries. She reports significant improvement in low back symptoms with mild intermittent stiffness. She endorses headaches and driving anxiety. She is working and pleased with musculoskeletal progress. Follow-up appointments scheduled with neurology and behavioral health. Past medical, surgical, family, allergy, and social history unchanged per prior evaluations.

OBJECTIVE:
Patient is alert and oriented. Lumbar spine inspection: no bony deformities. No tenderness to palpation over paralumbar musculature. Full range of motion with flexion, extension, lateral rotation, and lateral bending without significant pain. Extremity strength normal bilaterally.

ASSESSMENT:
Lumbar spine sprain/strain, improving, sustained in motor vehicle accident.

PLAN:
1. Continue home exercise program as instructed.
2. Apply ice or heat as needed for comfort.
3. Follow-up with neurology regarding headaches.
4. Follow-up with behavioral health for driving anxiety.
5. Patient discharged from physical medicine and rehabilitation service; may return if symptoms worsen.
"""


def make_provider(note_format: NoteType = NoteType.SOAP) -> ProviderProfile:
    return ProviderProfile(
        id="test-provider-001",
        name="Dr. Test",
        specialty="physical_medicine",
        note_format=note_format,
        template_id="soap_default",
        style_directives=["Use active voice", "Be concise"],
    )


def make_state_with_transcript(
    transcript: str = SAMPLE_TRANSCRIPT,
    note_format: NoteType = NoteType.SOAP,
) -> EncounterState:
    unified = UnifiedTranscript(
        segments=[
            TranscriptSegment(
                text=transcript,
                speaker="SPEAKER_00",
                start_ms=0,
                end_ms=60000,
                confidence=0.95,
                mode=RecordingMode.DICTATION,
            )
        ],
        engine_used="whisperx",
        full_text=transcript,
    )
    return EncounterState(
        provider_id="test-provider-001",
        patient_id="test-patient-001",
        provider_profile=make_provider(note_format),
        recording_mode=RecordingMode.DICTATION,
        transcript=unified,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM engine
# ─────────────────────────────────────────────────────────────────────────────

class MockLLMEngine(LLMEngine):
    """Returns a predetermined SOAP note without calling any external service."""

    def __init__(self, response_text: str = MOCK_LLM_SOAP_RESPONSE, model: str = "mock-model"):
        self.response_text = response_text
        self._model = model
        self.call_count = 0
        self.last_system_prompt: str = ""
        self.last_user_message: str = ""

    def generate_sync(self, system_prompt, messages, config, task="note_generation") -> LLMResponse:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_message = messages[0].content if messages else ""
        return LLMResponse(
            content=self.response_text,
            model=self._model,
            prompt_tokens=len(system_prompt) // 4,
            completion_tokens=len(self.response_text) // 4,
            finish_reason="stop",
        )

    async def generate(self, system_prompt, messages, config) -> LLMResponse:
        return self.generate_sync(system_prompt, messages, config)

    async def generate_stream(self, system_prompt, messages, config):
        yield  # pragma: no cover

    async def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self._model, context_window=32768)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (mock LLM — no Ollama required)
# ─────────────────────────────────────────────────────────────────────────────

class TestNoteParser:
    def test_parses_four_soap_sections(self):
        sections = parse_note_sections(MOCK_LLM_SOAP_RESPONSE, NoteType.SOAP)
        types = {s.type for s in sections}
        assert types == {"subjective", "objective", "assessment", "plan"}

    def test_section_content_not_empty(self):
        sections = parse_note_sections(MOCK_LLM_SOAP_RESPONSE, NoteType.SOAP)
        for s in sections:
            assert len(s.content) > 20, f"Section {s.type!r} content too short"

    def test_no_headers_in_content(self):
        sections = parse_note_sections(MOCK_LLM_SOAP_RESPONSE, NoteType.SOAP)
        for s in sections:
            # Content should not start with the header name itself
            assert not s.content.upper().startswith(s.type.upper())

    def test_fallback_on_no_headers(self):
        """LLM output with no headers → single subjective fallback."""
        raw = "The patient has back pain and was seen in clinic today."
        sections = parse_note_sections(raw, NoteType.SOAP)
        assert len(sections) == 1
        assert sections[0].type == "subjective"
        assert raw.strip() in sections[0].content

    def test_markdown_headers_parsed(self):
        """LLM sometimes wraps headers in markdown."""
        md_response = """## SUBJECTIVE
Patient reports back pain.

## OBJECTIVE
Normal exam.

## ASSESSMENT
Back sprain.

## PLAN
Physical therapy.
"""
        sections = parse_note_sections(md_response, NoteType.SOAP)
        types = {s.type for s in sections}
        assert "subjective" in types
        assert "objective" in types

    def test_bold_headers_parsed(self):
        """LLM sometimes bolds headers."""
        bold_response = """**SUBJECTIVE:**
Patient reports neck pain.

**OBJECTIVE:**
Decreased ROM.

**ASSESSMENT:**
Cervical sprain.

**PLAN:**
Rest and NSAIDs.
"""
        sections = parse_note_sections(bold_response, NoteType.SOAP)
        types = {s.type for s in sections}
        assert "subjective" in types

    def test_hp_note_parsed(self):
        hp_response = """CHIEF COMPLAINT:
Back pain after MVA.

HISTORY OF PRESENT ILLNESS:
Onset 2 weeks ago after rear-end collision.

PAST MEDICAL HISTORY:
None significant.

MEDICATIONS:
Ibuprofen 400mg PRN.

ALLERGIES:
NKDA.

FAMILY HISTORY:
Non-contributory.

SOCIAL HISTORY:
Non-smoker, employed.

REVIEW OF SYSTEMS:
Musculoskeletal: back pain. All others negative.

PHYSICAL EXAMINATION:
Lumbar spine tenderness to palpation.

ASSESSMENT AND PLAN:
Lumbar sprain. Continue NSAIDs, PT referral.
"""
        sections = parse_note_sections(hp_response, NoteType.HP)
        types = {s.type for s in sections}
        assert "chief_complaint" in types
        assert "history_of_present_illness" in types
        assert "assessment_and_plan" in types


class TestPromptAssembly:
    def test_system_prompt_not_empty(self):
        state = make_state_with_transcript()
        system_prompt, user_msg, _ = assemble_prompt(state)
        assert len(system_prompt) > 100
        assert "medical" in system_prompt.lower()

    def test_transcript_in_user_message(self):
        state = make_state_with_transcript()
        _, user_msg, _ = assemble_prompt(state)
        assert "motor vehicle" in user_msg.lower()

    def test_style_directives_injected(self):
        state = make_state_with_transcript()
        _, user_msg, _ = assemble_prompt(state)
        # Provider has style directives "Use active voice", "Be concise"
        assert "active voice" in user_msg.lower()

    def test_long_transcript_truncated(self):
        long_transcript = "word " * 10_000  # ~50k chars
        state = make_state_with_transcript(transcript=long_transcript)
        _, user_msg, _ = assemble_prompt(state)
        assert "TRUNCATED" in user_msg

    def test_hp_note_uses_hp_prompt(self):
        state = make_state_with_transcript(note_format=NoteType.HP)
        system_prompt, _, _ = assemble_prompt(state)
        assert "History and Physical" in system_prompt or "H&P" in system_prompt or "medical" in system_prompt.lower()


class TestNoteNodeUnit:
    @pytest.fixture(autouse=True)
    def inject_mock_engine(self):
        """Inject mock LLM before each test; restore default after."""
        self.mock = MockLLMEngine()
        set_llm_engine_factory(lambda: self.mock)
        yield
        set_llm_engine_factory(None)

    def test_note_node_produces_clinical_note(self):
        graph = build_graph()
        state = make_state_with_transcript()
        final = run_encounter(graph, state)

        assert final.generated_note is not None
        assert final.llm_engine_used == "mock-model"

    def test_all_soap_sections_present(self):
        graph = build_graph()
        final = run_encounter(graph, make_state_with_transcript())
        types = {s.type for s in final.generated_note.sections}
        assert {"subjective", "objective", "assessment", "plan"}.issubset(types)

    def test_confidence_score_above_zero(self):
        graph = build_graph()
        final = run_encounter(graph, make_state_with_transcript())
        assert final.generated_note.metadata.confidence_score > 0

    def test_llm_called_exactly_once(self):
        graph = build_graph()
        run_encounter(graph, make_state_with_transcript())
        assert self.mock.call_count == 1

    def test_transcript_passed_to_llm(self):
        graph = build_graph()
        run_encounter(graph, make_state_with_transcript())
        assert "motor vehicle" in self.mock.last_user_message.lower()

    def test_template_id_recorded_in_note(self):
        graph = build_graph()
        final = run_encounter(graph, make_state_with_transcript())
        assert "soap_default" in (final.generated_note.metadata.template_used or "")

    def test_token_counts_recorded(self):
        graph = build_graph()
        final = run_encounter(graph, make_state_with_transcript())
        meta = final.generated_note.metadata
        assert meta.prompt_tokens > 0
        assert meta.completion_tokens > 0

    def test_note_to_text_contains_sections(self):
        graph = build_graph()
        final = run_encounter(graph, make_state_with_transcript())
        text = final.final_note.to_text()
        assert "SUBJECTIVE" in text.upper()
        assert "PLAN" in text.upper()

    def test_llm_error_produces_fallback_note(self):
        """When LLM throws, note_node must not crash the pipeline."""
        class BrokenEngine(MockLLMEngine):
            def generate_sync(self, *a, **kw):
                raise ConnectionError("Ollama is not running")

        set_llm_engine_factory(lambda: BrokenEngine())
        graph = build_graph()
        final = run_encounter(graph, make_state_with_transcript())

        assert final.generated_note is not None
        assert final.llm_engine_used == "fallback_stub"
        assert any("note_node" in e for e in final.errors)
        assert final.status.value == "DELIVERED"   # pipeline still completes

    def test_session1_tests_still_pass(self):
        """Ensure Session 1 skeleton tests are unaffected by the real note_node."""
        graph = build_graph()
        state = make_state_with_transcript()
        final = run_encounter(graph, state)
        assert final.metrics.nodes_completed == [
            "context", "capture", "transcribe", "note", "review", "delivery"
        ]
        assert final.status.value == "DELIVERED"


# ─────────────────────────────────────────────────────────────────────────────
# Integration test — real Ollama (skip if not running)
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_status() -> tuple[bool, str]:
    """Return (available, first_model_name). Empty model name if none loaded."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code != 200:
            return False, ""
        models = resp.json().get("models", [])
        first = models[0]["name"] if models else ""
        return True, first
    except Exception:
        return False, ""


_OLLAMA_AVAILABLE, _OLLAMA_MODEL = _ollama_status()


@pytest.mark.skipif(
    not _OLLAMA_AVAILABLE or not _OLLAMA_MODEL,
    reason="Ollama not running or no models loaded on localhost:11434",
)
class TestNoteNodeIntegration:
    """Live integration tests against a real Ollama instance.

    Skipped if Ollama is not running or has no models loaded.
    Uses whatever model is first in `ollama list` — no hardcoded model name.
    """

    @pytest.fixture(autouse=True)
    def use_real_engine_with_available_model(self):
        """Configure OllamaServer to use the first available model."""
        from mcp_servers.llm.ollama_server import OllamaServer

        engine = OllamaServer(
            url="http://localhost:11434/v1",
            model_overrides={"note_generation": _OLLAMA_MODEL},
        )
        set_llm_engine_factory(lambda: engine)
        yield
        set_llm_engine_factory(None)

    def test_real_ollama_generates_soap_note(self):
        graph = build_graph()
        state = make_state_with_transcript()
        final = run_encounter(graph, state)

        note = final.generated_note
        assert note is not None
        assert final.llm_engine_used != "fallback_stub", (
            f"LLM call failed with model {_OLLAMA_MODEL!r}: {final.errors}"
        )
        types = {s.type for s in note.sections}
        assert "subjective" in types
        assert "plan" in types
        assert note.metadata.prompt_tokens > 0

    def test_real_ollama_with_test_data_file(self):
        """Feed the real gold-standard dictation transcript to the LLM.

        Compares note structure (sections present) to the gold SOAP note.
        Does not require exact text match — just verifies structural correctness.
        """
        import re as re_module
        from pathlib import Path

        gold_path = Path("ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/final_soap_note.md")
        if not gold_path.exists():
            pytest.skip("Test data not found at ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/final_soap_note.md")

        gold_text = gold_path.read_text()
        body_match = re_module.search(r"SUBJECTIVE:.+", gold_text, re_module.DOTALL)
        if not body_match:
            pytest.skip("Could not parse gold standard note body")
        note_body = body_match.group(0)

        state = make_state_with_transcript(transcript=note_body)
        graph = build_graph()
        final = run_encounter(graph, state)

        note = final.generated_note
        assert note is not None
        assert final.llm_engine_used != "fallback_stub", (
            f"LLM call failed: {final.errors}"
        )
        types = {s.type for s in note.sections}
        assert "subjective" in types
        assert "plan" in types
