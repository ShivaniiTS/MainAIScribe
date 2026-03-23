# AI Scribe Enterprise — Implementation Reference

> **Scope:** Sessions 1–8 (complete). The system converts doctor-patient audio into
> structured clinical notes with no cloud dependencies. Every AI component is
> swappable via config; PHI never leaves your infrastructure.

---

## Table of Contents

1. [What Is Built](#1-what-is-built)
2. [Repository Layout](#2-repository-layout)
3. [The Encounter Pipeline](#3-the-encounter-pipeline)
   - 3.1 LangGraph State Graph
   - 3.2 EncounterState Schema
   - 3.3 Node-by-Node Detail
4. [MCP Tool Server Pattern](#4-mcp-tool-server-pattern)
   - 4.1 Engine Registry
   - 4.2 ASR Servers
   - 4.3 LLM Servers
   - 4.4 EHR Adapters
   - 4.5 Data / Reference Servers
5. [ASR Post-Processor](#5-asr-post-processor)
6. [Note Generation — Prompt Assembly](#6-note-generation--prompt-assembly)
7. [Provider Profile System](#7-provider-profile-system)
8. [Template System](#8-template-system)
9. [Quality Evaluation Framework](#9-quality-evaluation-framework)
10. [Configuration Reference](#10-configuration-reference)
11. [Data Flow: Audio → Clinical Note](#11-data-flow-audio--clinical-note)
12. [Quality Results](#12-quality-results)
13. [Running the Pipeline](#13-running-the-pipeline)
14. [Sessions Roadmap](#14-sessions-roadmap)

---

## 1. What Is Built

Eight development sessions have produced a **fully functional end-to-end pipeline**:

| Capability | Status | Where |
|-----------|--------|-------|
| LangGraph 6-node pipeline | ✅ | `orchestrator/` |
| WhisperX ASR (GPU, diarization) | ✅ | `mcp_servers/asr/whisperx_server.py` |
| 12-stage ASR post-processor | ✅ | `postprocessor/medasr_postprocessor.py` |
| Offensive misrecognition filter | ✅ | Stage 0 of post-processor |
| Ollama LLM note generation | ✅ | `mcp_servers/llm/ollama_server.py` |
| Note templates (SOAP, ortho follow-up, initial eval) | ✅ | `config/templates/` |
| EHR context integration (stub) | ✅ | `mcp_servers/ehr/stub_server.py` |
| Patient demographics extraction | ✅ | `scripts/extract_patient_context.py` |
| Provider profiles (YAML, CRUD) | ✅ | `config/providers/`, `config/provider_manager.py` |
| Provider-specific template routing | ✅ | `config/provider_manager.py` |
| Specialty vocabulary injection | ✅ | `mcp_servers/data/medical_dict_server.py` |
| Style directive injection | ✅ | `orchestrator/nodes/note_node.py` |
| Engine registry (discovery, caching, failover) | ✅ | `mcp_servers/registry.py` |
| Quality evaluation (LLM-as-judge) | ✅ | `quality/` |
| Per-provider quality history | ✅ | `config/providers/*.yaml` |
| Regression test suite | ✅ | `tests/quality/test_regression.py` |
| Markdown output (notes, comparisons, reports) | ✅ | `output/` |

**Current quality:** 4.38 / 5.0 across 22 gold-standard orthopedic encounters
(dictation + ambient conversation mode).

---

## 2. Repository Layout

```
ai-scribe-enterprise/
│
├── config/
│   ├── engines.yaml                  ← ALL engine selection; zero code changes to swap
│   ├── quality_baseline.yaml         ← v2 quality baseline (regression gate)
│   ├── loader.py                     ← Cached YAML loaders
│   ├── provider_manager.py           ← ProviderProfile CRUD + template routing
│   ├── providers/
│   │   └── dr_faraz_rahman.yaml      ← One YAML per provider
│   ├── templates/
│   │   ├── soap_default.yaml
│   │   ├── ortho_follow_up.yaml      ← 6 sections, dictation mode
│   │   └── ortho_initial_eval.yaml   ← 12 sections, ambient mode
│   ├── dictionaries/
│   │   ├── orthopedic.txt
│   │   ├── gastroenterology.txt
│   │   ├── cardiology.txt
│   │   └── neurology.txt
│   └── prompts/
│       ├── note_generation.yaml      ← SOAP/H&P/Progress system + user templates
│       └── patient_summary.yaml
│
├── orchestrator/
│   ├── graph.py                      ← LangGraph build_graph() + run_encounter()
│   ├── state.py                      ← EncounterState + all sub-models (Pydantic v2)
│   ├── nodes/
│   │   ├── context_node.py           ← Load patient context via EHR adapter
│   │   ├── capture_node.py           ← Audio pass-through stub (Session 3)
│   │   ├── transcribe_node.py        ← WhisperX + post-processor
│   │   ├── note_node.py              ← Prompt assembly + LLM + parser
│   │   ├── review_node.py            ← Auto-approve stub (HITL in Session 9+)
│   │   └── delivery_node.py          ← Write to clipboard / output
│   └── edges/
│       ├── asr_router.py             ← Conditional edge after transcribe
│       └── llm_router.py             ← Conditional edge after note
│
├── mcp_servers/
│   ├── registry.py                   ← Engine discovery, caching, health checks
│   ├── asr/
│   │   ├── base.py                   ← ASREngine interface + ASRConfig + RawTranscript
│   │   └── whisperx_server.py        ← WhisperX (faster-whisper + pyannote 3.1)
│   ├── llm/
│   │   ├── base.py                   ← LLMEngine interface + LLMConfig + LLMResponse
│   │   └── ollama_server.py          ← Ollama via OpenAI-compatible API
│   ├── ehr/
│   │   ├── base.py                   ← EHRAdapter interface (full read/write)
│   │   └── stub_server.py            ← Reads patient_context.yaml files
│   └── data/
│       ├── medical_dict_server.py    ← 98K base + specialty hotwords
│       └── template_server.py        ← YAML template loader
│
├── postprocessor/
│   ├── medasr_postprocessor.py       ← 12-stage rule-based cleanup (1,723 lines)
│   └── medical_wordlist.txt          ← 98K OpenMedSpel medical dictionary (GPL 3.0)
│
├── quality/
│   ├── evaluator.py                  ← LLM-as-judge scoring (6 dimensions)
│   ├── fact_extractor.py             ← Medication/diagnosis/findings extraction
│   ├── report.py                     ← Per-sample + aggregate Markdown reports
│   ├── dimensions.py                 ← Scoring rubric definitions
│   └── experiments.jsonl             ← All optimization runs (gitignored)
│
├── output/
│   ├── markdown_writer.py            ← Clinical note → formatted .md
│   ├── comparison_writer.py          ← Side-by-side gold vs generated .md
│   ├── batch_report_v*.md            ← Aggregate pipeline reports
│   └── quality_report_v*.md          ← LLM-judge quality sweep reports
│
├── scripts/
│   ├── batch_eval.py                 ← Run pipeline on all samples → v{N} outputs
│   ├── run_quality_sweep.py          ← LLM-as-judge all samples → quality scores
│   └── extract_patient_context.py    ← Extract demographics from gold notes
│
├── tests/
│   └── quality/
│       └── test_regression.py        ← 11 quality regression tests
│
└── data/                             ← Gitignored (PHI)
    ├── dictation/                    ← 21 single-speaker follow-up samples
    └── conversations/                ← 12 multi-speaker initial eval samples
```

---

## 3. The Encounter Pipeline

### 3.1 LangGraph State Graph

The pipeline is a compiled LangGraph `StateGraph` with `EncounterState` as the typed
state object. Every node receives the full state and returns a partial update dict;
LangGraph merges updates via `model_copy`.

```
CONTEXT → CAPTURE → TRANSCRIBE ──[asr_router]──► NOTE ──[llm_router]──► REVIEW → DELIVERY → END
```

**Built in:** [`orchestrator/graph.py`](../orchestrator/graph.py)

```python
graph = StateGraph(EncounterState)
graph.add_node("context",    context_node)
graph.add_node("capture",    capture_node)
graph.add_node("transcribe", transcribe_node)
graph.add_node("note",       note_node)
graph.add_node("review",     review_node)
graph.add_node("delivery",   delivery_node)

graph.set_entry_point("context")
graph.add_edge("context",  "capture")
graph.add_edge("capture",  "transcribe")
graph.add_conditional_edges("transcribe", asr_router,  {"note": "note"})
graph.add_conditional_edges("note",       llm_router,  {"review": "review"})
graph.add_edge("review",   "delivery")
graph.add_edge("delivery", END)

compiled = graph.compile()
```

Entry point: `run_encounter(graph, initial_state) → EncounterState`

### 3.2 EncounterState Schema

**Built in:** [`orchestrator/state.py`](../orchestrator/state.py) (400 lines)

All sub-models are Pydantic v2 `BaseModel`. Key fields:

```python
class EncounterState(BaseModel):
    # Identity
    encounter_id: str                         # UUID, auto-generated
    provider_id: str
    patient_id: str
    status: EncounterStatus                   # CREATED → DELIVERED

    # Provider profile (loaded at start, governs entire pipeline)
    provider_profile: ProviderProfile

    # ── CONTEXT NODE fills these ──────────────────────────────────────
    context_packet: Optional[ContextPacket]   # demographics + encounter + provider + facility

    # ── CAPTURE NODE fills these ──────────────────────────────────────
    audio_file_path: Optional[str]
    recording_mode: RecordingMode             # AMBIENT | DICTATION

    # ── TRANSCRIBE NODE fills these ───────────────────────────────────
    transcript: Optional[UnifiedTranscript]   # segments + full_text + speaker labels
    asr_engine_used: str
    postprocessor_metrics: dict[str, Any]     # corrections per stage

    # ── NOTE NODE fills these ─────────────────────────────────────────
    generated_note: Optional[ClinicalNote]    # sections + citations + metadata
    template_used: str

    # ── REVIEW NODE fills these ───────────────────────────────────────
    corrections: list[Correction]
    final_note: Optional[ClinicalNote]
    review_approved: bool

    # ── Quality metrics (accumulated across all nodes) ─────────────────
    metrics: EncounterMetrics
    errors: list[str]
```

**ProviderProfile** (in the same file):

```python
class ProviderProfile(BaseModel):
    id: str
    name: str
    specialty: str
    credentials: Optional[str]                # "MD", "DO", "NP", etc.
    note_format: NoteType                     # SOAP | H&P | PROGRESS | DISCHARGE
    template_id: str                          # default template (overridden by routing)
    style_directives: list[str]               # injected into LLM system prompt
    custom_vocabulary: list[str]              # injected into ASR + LLM prompt
    asr_override: Optional[str]               # force a specific ASR engine
    llm_override: Optional[str]               # force a specific LLM engine
    quality_scores: dict[str, float]          # version → score, updated by sweep
```

**ContextPacket** (filled by Context Node):

```python
class ContextPacket(BaseModel):
    patient: Optional[PatientDemographics]    # name, DOB, sex, MRN
    encounter: Optional[EncounterContext]     # date_of_service, visit_type, date_of_injury
    provider_context: Optional[ProviderContext] # name, credentials, specialty
    facility: Optional[FacilityContext]       # name, location
    problem_list: list[Problem]
    medications: list[Medication]
    allergies: list[Allergy]
    recent_labs: list[LabResult]
    last_visit_note_summary: Optional[str]
    source: str                               # "stub" | "fhir" | "manual"
```

### 3.3 Node-by-Node Detail

#### CONTEXT NODE
**File:** [`orchestrator/nodes/context_node.py`](../orchestrator/nodes/context_node.py)

1. Looks for `patient_context.yaml` adjacent to the audio file
   (`_find_context_yaml(state) → Optional[Path]`)
2. Gets the EHR adapter via `get_registry().get_ehr()`
3. Calls `ehr.set_context_path(yaml_path)` on `StubEHRServer`
4. Loads patient, encounter, provider, facility data asynchronously
5. Builds and returns a `ContextPacket`
6. Graceful fallback: empty context packet if no YAML or any error

```python
# Simplified flow
ctx_path = _find_context_yaml(state)
if ctx_path:
    ehr = get_registry().get_ehr()
    ehr.set_context_path(ctx_path)
    patient_ehr = await ehr.get_patient(PatientIdentifier(mrn=state.patient_id))
    encounter_data = ehr.get_encounter_context()   # stub-specific accessor
    ...
    return {"context_packet": ContextPacket(patient=..., encounter=..., ...)}
```

#### TRANSCRIBE NODE
**File:** [`orchestrator/nodes/transcribe_node.py`](../orchestrator/nodes/transcribe_node.py)

1. **Pass-through guard:** if `state.transcript` already populated, skip ASR
2. Resolves audio path from `state.audio_file_path` or `state.audio_segments[0]`
3. Builds `ASRConfig` (diarization=True if `AMBIENT`, custom vocab from provider profile)
4. Calls `engine.transcribe_batch_sync(audio_path, asr_cfg) → RawTranscript`
5. Converts `RawTranscript → UnifiedTranscript` (`_raw_to_unified`)
6. Runs `_apply_postprocessor(transcript, mode)` — 12-stage rule-based cleanup
7. Scores ASR confidence (word-level average)
8. Returns updated transcript + metrics

Custom vocabulary from the provider profile is passed directly to WhisperX as hotwords
(injected into the initial prompt / beam search).

#### NOTE NODE
**File:** [`orchestrator/nodes/note_node.py`](../orchestrator/nodes/note_node.py) — 645 lines

This is the most complex node. See [Section 6](#6-note-generation--prompt-assembly)
for full prompt assembly detail.

High-level flow:

```
assemble_prompt(state) → (system_prompt, user_message, template)
    ├── _load_template(state)          ← provider_manager.resolve_template() + template server
    ├── _assemble_context_block(state) ← patient demographics from ContextPacket
    ├── _assemble_vocab_block(state)   ← specialty hotwords + custom vocabulary
    ├── _assemble_template_block(tpl)  ← section headers + prompt hints
    ├── _strip_phi_headers(transcript) ← redact spoken/structured PHI
    ├── _budget_transcript(text)       ← truncate at 24K chars
    └── _assemble_style_block(state)   ← provider style directives

engine.generate_sync(system_prompt, [user_message], cfg)
    → LLMResponse.content

if _is_refusal(response): retry with override message

parse_note_sections(response, note_type, template) → list[NoteSection]
_score_confidence(sections, note_type, template)    → float (0–1)
```

**LLM refusal handling:** When ASR misrecognitions produce words that trigger safety
filters, the model may refuse. `_is_refusal()` detects this and retries with an
explicit override message explaining it is garbled ASR output.

**PHI stripping:** `_strip_phi_headers()` removes structured headers
(`LAST NAME: X`) and inline spoken PHI (`last name is G-R-A-M-B-L-I-N`) before
sending to the LLM, preventing both refusals and PHI leakage into the LLM's context.

#### REVIEW NODE
**File:** [`orchestrator/nodes/review_node.py`](../orchestrator/nodes/review_node.py)

Currently a stub (auto-approve). Copies `generated_note → final_note`, sets
`review_approved = True`. HITL review UI is planned for Session 13.

#### DELIVERY NODE
**File:** [`orchestrator/nodes/delivery_node.py`](../orchestrator/nodes/delivery_node.py)

Writes the final note to the delivery method specified in state
(`DeliveryMethod.CLIPBOARD` in batch eval). Updates encounter status to `DELIVERED`.

---

## 4. MCP Tool Server Pattern

Every external capability is wrapped in an MCP server with a standard base interface.
Nodes call servers via the engine registry — they never import servers directly.

```
node_code                engine_registry              mcp_server
   │                          │                           │
   ├─ get_registry().get_llm()─►                          │
   │                          ├─ lookup config ──────────►│
   │                          ├─ instantiate & cache ────►│
   │  ◄── LLMEngine instance ─┤                           │
   │                          │                           │
   ├─ engine.generate_sync(...)──────────────────────────►│
   │  ◄── LLMResponse ────────────────────────────────────┤
```

### 4.1 Engine Registry

**File:** [`mcp_servers/registry.py`](../mcp_servers/registry.py) — 335 lines

```python
# Module-level singleton
registry = get_registry()

# Typed accessors
llm: LLMEngine  = registry.get_llm(server_name=None)   # uses config default
asr: ASREngine  = registry.get_asr(server_name=None)
ehr: EHRAdapter = registry.get_ehr(adapter_name=None)

# With failover: tries default, falls back to next configured server
engine = registry.get_with_failover("llm")

# Health checks (async)
status = await registry.health_check("llm", "ollama")
all_ok = await registry.health_check_defaults()

# Discovery
registry.list_configured("asr")    # names in engines.yaml
registry.list_available("asr")     # names with implementations registered
```

**Server map** (lazy imports, registered at module load):

```python
_SERVER_MAP = {
    ("asr", "whisperx"):   ("mcp_servers.asr.whisperx_server",  "WhisperXServer"),
    ("llm", "ollama"):     ("mcp_servers.llm.ollama_server",    "OllamaServer"),
    ("ehr", "stub"):       ("mcp_servers.ehr.stub_server",      "StubEHRServer"),
    ("ehr", "manual"):     ("mcp_servers.ehr.stub_server",      "StubEHRServer"),
}
```

Adding a new engine: (1) add entry to `_SERVER_MAP`, (2) add config block to
`engines.yaml`. Zero changes to pipeline nodes.

### 4.2 ASR Servers

**Base interface:** [`mcp_servers/asr/base.py`](../mcp_servers/asr/base.py)

```python
class ASREngine(ABC):
    @property
    def name(self) -> str: ...

    def transcribe_batch_sync(
        self, audio_path: str, config: ASRConfig
    ) -> RawTranscript: ...

    async def transcribe_batch(
        self, audio_path: str, config: ASRConfig
    ) -> RawTranscript: ...
```

**ASRConfig** carries:
- `language` (default `"en"`)
- `diarize: bool` — enables pyannote speaker separation
- `max_speakers: int`
- `custom_vocabulary: list[str]` — provider hotwords injected as WhisperX initial_prompt

**RawTranscript** → **UnifiedTranscript** conversion in `transcribe_node._raw_to_unified()`.
Speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) and word-level timestamps are preserved.

**WhisperX server** (`mcp_servers/asr/whisperx_server.py`):
- Loads `faster-whisper large-v3` (CTranslate2 optimized) on CUDA
- Uses `pyannote/speaker-diarization-3.1` for diarization (requires `HF_TOKEN`)
- `wav2vec2` for word-level timestamp alignment
- Model is loaded once on `from_config()` and reused — prevents CUDA OOM

### 4.3 LLM Servers

**Base interface:** [`mcp_servers/llm/base.py`](../mcp_servers/llm/base.py)

```python
class LLMEngine(ABC):
    def generate_sync(
        self,
        system_prompt: str,
        messages: list[LLMMessage],
        config: LLMConfig,
        task: str = "note_generation",   # selects which model to use
    ) -> LLMResponse: ...

    async def generate(
        self, system_prompt, messages, config, task
    ) -> LLMResponse: ...
```

**LLMConfig**: `model`, `temperature` (default 0.1), `max_tokens` (default 4096),
`top_p`, `stop_sequences`.

**Ollama server** (`mcp_servers/llm/ollama_server.py`):
- POSTs to `http://localhost:11434/v1/chat/completions` (OpenAI-compatible)
- `model_overrides` dict: task → model name
  - `note_generation` → `qwen2.5:32b` (or `llama3.1:latest` in dev)
  - `patient_summary` → `qwen2.5:14b`
  - `command_parse`   → `qwen2.5:7b`
- Switching to vLLM, Claude, or OpenAI requires only a URL + key change

### 4.4 EHR Adapters

**Base interface:** [`mcp_servers/ehr/base.py`](../mcp_servers/ehr/base.py)

```python
class EHRAdapter(ABC):
    # Read (pre-encounter context)
    async def get_patient(self, identifier: PatientIdentifier) -> EHRPatient
    async def get_problem_list(self, patient_id: str) -> list[EHRProblem]
    async def get_medications(self, patient_id: str) -> list[EHRMedication]
    async def get_allergies(self, patient_id: str) -> list[EHRAllergy]
    async def get_recent_labs(self, patient_id: str, days: int) -> list[EHRLabResult]
    async def get_last_visit_note(self, patient_id: str) -> Optional[str]

    # Write (post-encounter delivery)
    async def push_note(self, patient_id: str, note: str) -> PushResult
    async def push_order(self, patient_id: str, order: dict) -> PushResult
    async def navigate(self, command: str) -> NavigationResult
```

**Stub EHR server** (`mcp_servers/ehr/stub_server.py`):
- Implements the full interface; reads from `patient_context.yaml`
- Extra accessors: `get_encounter_context()`, `get_provider_context()`,
  `get_facility_context()` — return raw dicts from YAML

Replacing with a real FHIR adapter later requires only a new class implementing
`EHRAdapter` and a registry entry. The pipeline code is unchanged.

### 4.5 Data / Reference Servers

**Medical dict server** (`mcp_servers/data/medical_dict_server.py`):

```python
srv = get_dict_server()
terms    = srv.get_terms("orthopedic")        # base 98K + specialty terms
hotwords = srv.get_hotwords("orthopedic", 200) # specialty-only, sorted by length
context  = srv.get_specialty_context("orthopedic", 50) # compact string for LLM
```

Specialty dictionaries in `config/dictionaries/`:
`orthopedic.txt`, `gastroenterology.txt`, `cardiology.txt`, `neurology.txt`

**Template server** (`mcp_servers/data/template_server.py`):
Loads all `config/templates/*.yaml` files into `NoteTemplate` objects.
`get_template(specialty, visit_type)` does fuzzy matching on specialty + visit_type.

---

## 5. ASR Post-Processor

**File:** [`postprocessor/medasr_postprocessor.py`](../postprocessor/medasr_postprocessor.py)
— 1,723 lines, production-tested on 10+ real medical transcripts.

### 12-Stage Pipeline

```
Stage 0:  Offensive misrecognition filter (added Session 8)
          → Removes words like "racist", "pedophile" that are ASR misrecognitions
          → Regex with word boundaries, case-insensitive

Stage 1:  System artifact removal
          → [unintelligible], {format commands}, filler noise markers

Stage 2:  CTC stutter pair merging
          → "the the patient" → "the patient"  (90% reduction)

Stage 3:  Character-level stutter fixing
          → "s-s-shoulder" → "shoulder"  (99% reduction)

Stage 4:  Repetition sequence collapse
          → "pain pain pain" → "pain"

Stage 5:  MedASR-specific patterns
          → System-specific artifacts from MedASR CTC output

Stage 6:  Filler word removal
          → "um", "uh", "you know", hesitation markers

Stage 7:  Sentence boundary restoration
          → Capitalise after periods, fix run-on sentences

Stage 8:  Medical abbreviation expansion (selective)
          → Expand only unambiguous abbreviations

Stage 9:  Number and unit normalisation
          → "five milligrams" → "5 mg", range normalisation

Stage 10: Medical dictionary lookup (base 98K)
          → Flag and correct non-dictionary medical terms

Stage 11: Medical spell check (optional, requires symspellpy)
          → Suggestion-based correction for out-of-vocabulary terms
          → Currently SKIPPED (symspellpy not installed)

Stage 12: Final cleanup
          → Trailing whitespace, multiple spaces, empty lines
```

### Stage 0: Offensive Misrecognition Filter

A critical safety stage added after observing that CTC ASR sometimes produces
offensive words as phonetic misrecognitions of medical terms. These words trigger
LLM safety refusals even in clearly clinical contexts.

```python
_OFFENSIVE_MISRECOGNITION_RE = re.compile(
    r"\b(?:"
    r"pedophil(?:e|ia|ic|es)?"
    r"|racist(?:s)?"
    r"|racism"
    r"|rapist(?:s)?"
    r"|neo-?nazi(?:s)?"
    r"|misogynist(?:s|ic)?"
    r"|pornograph(?:y|ic|er)?"
    r"|masturbat(?:ion|ing|e)?"
    r")\b",
    re.IGNORECASE,
)
```

Removed words are counted in `CleanupMetrics.offensive_misrecognitions_removed`.

### Usage

```python
from postprocessor import run_postprocessor

cleaned_text, metrics = run_postprocessor(
    raw_transcript,
    use_medical_spellcheck=True,   # Stage 11; skipped if symspellpy absent
)
# metrics: dict with counts per stage
```

---

## 6. Note Generation — Prompt Assembly

**File:** [`orchestrator/nodes/note_node.py`](../orchestrator/nodes/note_node.py),
function `assemble_prompt(state) → (system_prompt, user_message, template)`

### Prompt Structure

```
SYSTEM PROMPT
  ┌─────────────────────────────────────────────────────────────────┐
  │ You are an expert medical scribe operating in an authorized,     │
  │ HIPAA-compliant clinical documentation workflow...               │
  │ RULES: use only transcript, third person, preserve medical terms │
  └─────────────────────────────────────────────────────────────────┘

USER MESSAGE (assembled from blocks)
  ┌─────────────────────────────────────────────────────────────────┐
  │ {context_block}     ← patient demographics from ContextPacket   │
  │ {template_block}    ← ordered sections with prompt hints        │
  │ {vocab_block}       ← specialty hotwords (up to 80 terms)       │
  │ TRANSCRIPT:                                                      │
  │ {transcript}        ← cleaned, PHI-stripped, truncated ≤24K     │
  │                                                                  │
  │ {style_block}       ← provider style directives (bulleted)      │
  │ Generate the clinical note now.                                  │
  └─────────────────────────────────────────────────────────────────┘
```

### context_block

Rendered from `ContextPacket` — includes patient name/DOB/sex/MRN, date of service,
visit type, date of injury, provider name/credentials, facility name.

```
PATIENT CONTEXT (from EHR — do not repeat verbatim, use only to inform clinical accuracy):
Patient: Jane Doe | DOB: 1985-06-15 | Sex: Female | MRN: 0219-370
Encounter: Date of Service: 2024-03-07 | Visit Type: Initial Evaluation | Date of Injury: 2024-02-19
Provider: Dr. Faraz Rahman, MD, Orthopedic
Facility: Excelsia Injury Care
```

### template_block

Rendered from the selected `NoteTemplate`. Each section includes its label and
`prompt_hint` if defined:

```
NOTE FORMAT — generate ALL of these sections in order, using exactly these headers:
INTERVAL HISTORY:
  [Changes since last visit: symptom progression, treatments received, new symptoms]

PHYSICAL EXAMINATION:
  [Focused exam: ROM in degrees, strength, special tests, neurovascular status]

ASSESSMENT:
  [Numbered diagnoses with current status vs prior visit]

PLAN:
  [Numbered plan: medication changes, therapy, referrals, work status, next follow-up]
```

### vocab_block

Specialty hotwords from `MedicalDictServer.get_hotwords()` (up to 60 terms, sorted
by length descending for specificity) plus provider custom vocabulary (up to 20):

```
SPECIALTY VOCABULARY (Orthopedic) — recognise and use these terms correctly:
motor vehicle accident, occupational therapy, methylprednisolone, spondylolisthesis,
Spurling, Tinel, Phalen, Hawkins, McMurray, Lachman, paralumbar, facet, discectomy,
foraminotomy, WAD, MVA, ...
```

### style_block

Provider-specific style directives from `ProviderProfile.style_directives`:

```
PROVIDER STYLE PREFERENCES:
- Write in third person (e.g. 'The patient presents...' not 'Patient presents')
- Use past tense for examination findings and interval history
- Spell out abbreviations on first use (e.g. 'range of motion (ROM)')
- Include ROM measurements in degrees where documented
- List diagnoses in the Assessment as numbered items
- Mirror Assessment numbering in the Plan
- Use formal clinical language; avoid contractions
- Document work status in the Plan when mentioned
- For MVA cases, reference the accident date and mechanism in the Subjective/HPI
```

### Template Routing

`_load_template(state)` resolves the template in this order:

1. **Provider manager routing table:** `resolve_template(provider_id, visit_type)`
   where `visit_type` comes from `state.context_packet.encounter.visit_type`
2. **Direct filename match:** `template_id.yaml` exists in templates directory
3. **Specialty + visit_type fallback** via template server fuzzy match

For Dr. Rahman:
```yaml
template_routing:
  initial_evaluation: ortho_initial_eval   # 12-section H&P
  follow_up:          ortho_follow_up      # 6-section progress note
  assume_care:        ortho_follow_up
  discharge:          ortho_follow_up
  default:            ortho_follow_up
```

### Note Parsing

`parse_note_sections(llm_output, note_type, template) → list[NoteSection]`

Builds a regex from the template's section labels + standard SOAP/H&P aliases.
Handles `## HEADER`, `**HEADER:**`, and bare `HEADER:` formats.
Falls back to a single "subjective" section if no headers detected.

`_score_confidence(sections, note_type, template) → float`

```python
completeness = len(required_sections_found) / len(required_sections_expected)
score = completeness * 0.6
      + (0.2 if all sections have ≥20 chars content else 0.0)
      + (0.2 if no "[LLM UNAVAILABLE]" stubs else 0.0)
```

---

## 7. Provider Profile System

**Files:**
- [`config/provider_manager.py`](../config/provider_manager.py) — 265 lines
- [`config/providers/dr_faraz_rahman.yaml`](../config/providers/dr_faraz_rahman.yaml)

### YAML Structure

```yaml
# config/providers/{provider_id}.yaml
id: dr_faraz_rahman
name: "Dr. Faraz Rahman"
credentials: "MD"
specialty: orthopedic
practice_id: excelsia_injury_care

note_format: SOAP

template_routing:
  initial_evaluation: ortho_initial_eval
  follow_up:          ortho_follow_up
  assume_care:        ortho_follow_up
  default:            ortho_follow_up

style_directives:
  - "Write in third person..."
  - "Use past tense for examination findings..."
  - "Include ROM measurements in degrees..."
  # (10 directives total)

custom_vocabulary:
  - "Spurling"
  - "Tinel"
  - "paralumbar"
  - "WAD"
  # (22 terms total)

asr_override: null        # null = use registry default
llm_override: null
postprocessor_mode: hybrid

# Updated automatically by run_quality_sweep.py
quality_scores:
  v4: 4.38
quality_history:
  - date: "2026-03-11"
    version: v4
    score: 4.38
    samples: 22
    dimensions:
      medical_accuracy: 4.00
      completeness: 4.59
      no_hallucination: 4.73
      structure_compliance: 4.77
      clinical_language: 4.09
      readability: 4.00
```

### ProviderManager API

```python
from config.provider_manager import get_provider_manager

mgr = get_provider_manager()   # singleton

# CRUD
profile  = mgr.load("dr_faraz_rahman")           # raises FileNotFoundError if absent
profile  = mgr.load_or_default("unknown_id")     # returns minimal default
mgr.save(profile)                                # persist updates
mgr.create(profile, overwrite=False)
mgr.delete("provider_id")
providers = mgr.list_providers()                 # ["dr_faraz_rahman", ...]

# Template routing
template_id = mgr.resolve_template("dr_faraz_rahman", "follow_up")
# → "ortho_follow_up"

# Quality tracking (called by run_quality_sweep.py after each sweep)
mgr.update_quality_score(
    provider_id="dr_faraz_rahman",
    version="v4",
    score=4.38,
    sample_count=22,
    dimension_scores={"completeness": 4.59, ...},
)
trend = mgr.get_quality_trend("dr_faraz_rahman")
# → [{"date": "2026-03-11", "version": "v4", "score": 4.38, ...}]
```

---

## 8. Template System

**Files:** [`config/templates/*.yaml`](../config/templates/),
[`mcp_servers/data/template_server.py`](../mcp_servers/data/template_server.py)

### Template YAML Structure

```yaml
name: "Orthopedic Follow-up"
specialty: orthopedic
visit_type: follow_up

header_fields:
  - patient_name
  - date_of_birth
  - date_of_service
  - provider_name
  - location

sections:
  - id: interval_history
    label: "INTERVAL HISTORY"
    required: true
    prompt_hint: "Changes since last visit: symptom progression, treatments received..."

  - id: physical_examination
    label: "PHYSICAL EXAMINATION"
    required: true
    prompt_hint: "Focused exam: ROM in degrees, strength, special tests..."

  - id: imaging
    label: "IMAGING / DIAGNOSTICS"
    required: false
    prompt_hint: "Any new imaging reviewed; omit if not mentioned"

  - id: assessment
    label: "ASSESSMENT"
    required: true
    prompt_hint: "Numbered diagnoses with current status vs prior visit"

  - id: plan
    label: "PLAN"
    required: true
    prompt_hint: "Numbered plan: medications, therapy, referrals, work status"

formatting:
  voice: active
  tense: past
  person: third
  abbreviations: spell_out
  measurements: include_units
  rom_format: degrees
```

### Available Templates

| File | Specialty | Visit Type | Sections |
|------|-----------|-----------|---------|
| `soap_default.yaml` | any | any | 4 (S/O/A/P) |
| `ortho_follow_up.yaml` | orthopedic | follow_up | 6 (interval hx, meds, PE, imaging, A, P) |
| `ortho_initial_eval.yaml` | orthopedic | initial_evaluation | 12 (full H&P) |

Adding a new template: drop a YAML file into `config/templates/` and add to
`template_routing` in the relevant provider YAML. No code changes.

---

## 9. Quality Evaluation Framework

**Files:** [`quality/`](../quality/)

### Evaluation Dimensions (LLM-as-Judge)

Each dimension is scored 1–5 by `llama3.1:latest` (or any configured model):

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Medical Accuracy | 20% | Clinical facts match the gold note |
| Completeness | 20% | All relevant clinical content captured |
| No Hallucination | 20% | No invented findings, medications, or diagnoses |
| Structure Compliance | 20% | Correct sections in correct order |
| Clinical Language | 10% | Appropriate medical terminology and voice |
| Readability | 10% | Clear, well-formed sentences |

Overall = weighted average of dimension scores.

### Fact Extraction Check

`quality/fact_extractor.py` extracts structured facts from both notes using the LLM,
then computes precision/recall:

- Diagnoses found / total in gold
- Medications found / total in gold
- Plan items found / total in gold
- Exam findings found / total in gold

### Running a Sweep

```bash
python scripts/run_quality_sweep.py --version v4
```

Outputs:
- `output/quality_report_v4.md` — aggregate report with dimension table and per-sample scores
- Per-provider quality history updated in `config/providers/*.yaml`

### Regression Tests

```bash
python -m pytest tests/quality/test_regression.py -v
```

11 tests assert:
- Baseline file exists
- Overall average ≥ baseline
- No sample below minimum threshold
- Each dimension average ≥ baseline
- No zero-score samples
- Report has expected number of samples

---

## 10. Configuration Reference

### engines.yaml — Key Sections

```yaml
llm:
  default_server: ollama        # change to "vllm" or "claude" to swap
  servers:
    ollama:
      url: "http://localhost:11434/v1"
      models:
        note_generation: "qwen2.5:32b"
        patient_summary: "qwen2.5:14b"
        command_parse:   "qwen2.5:7b"

asr:
  default_server: whisperx
  servers:
    whisperx:
      model: "large-v3"
      device: "cuda"
      diarization: true
      hf_token_env: "HF_TOKEN"

ehr:
  default_adapter: manual       # "manual" → StubEHRServer; "fhir" when ready

post_processing:
  mode: hybrid                  # rules_only | ml_only | hybrid
  medical_wordlist: "postprocessor/medical_wordlist.txt"
```

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | Yes (ASR) | HuggingFace token for pyannote diarization model |
| `DATABASE_URL` | Future | PostgreSQL connection string |
| `MINIO_ENDPOINT` | Future | MinIO / S3 endpoint |
| `FHIR_BASE_URL` | Future | FHIR R4 server base URL |
| `ANTHROPIC_API_KEY` | Optional | Claude API (cloud LLM option) |

Store in `.env` file; load with `set -a && source .env && set +a`.

---

## 11. Data Flow: Audio → Clinical Note

```
1. AUDIO FILE
   data/dictation/226748/dictation.mp3   (or conversation.mp3 for ambient mode)

2. CONTEXT NODE
   → Finds: data/dictation/226748/patient_context.yaml
   → StubEHRServer reads YAML
   → Returns ContextPacket {
       patient:  {name: "Elizabeth McQuay", dob: "1974-03-17", sex: "Female"}
       encounter: {date_of_service: "2026-02-19", visit_type: "follow_up"}
       provider:  {name: "Dr. Faraz Rahman", credentials: "MD", specialty: "Orthopedic"}
       facility:  {name: "Excelsia Injury Care"}
     }

3. CAPTURE NODE (pass-through)
   → Sets audio_file_path = "data/dictation/226748/dictation.mp3"
   → Sets recording_mode = DICTATION

4. TRANSCRIBE NODE
   → WhisperX large-v3 on CUDA
   → custom_vocabulary = ["Spurling", "Tinel", "paralumbar", "WAD", ...] (from provider profile)
   → Returns RawTranscript → UnifiedTranscript
   → Post-processor (12 stages): stutter_pairs=3, char_stutters=0, offensive=0
   → full_text: "The patient presents for follow-up evaluation of her cervical
                 and lumbar spine injuries sustained in a motor vehicle accident..."

5. NOTE NODE
   → Provider manager: visit_type="follow_up" → template_id="ortho_follow_up"
   → Template: ortho_follow_up.yaml (6 sections)
   → Prompt assembled (context_block + template_block + vocab_block + transcript + style_block)
   → Ollama (llama3.1:latest / qwen2.5:32b) generates:

     INTERVAL HISTORY
     The patient presents for follow-up evaluation of injuries sustained in a motor
     vehicle accident on the above date. She reports that her cervical spine symptoms...

     PHYSICAL EXAMINATION
     The patient is alert and oriented. Range of motion of the cervical spine with
     flexion is 30 degrees, extension 25 degrees...

     ASSESSMENT
     1. Cervical spine sprain/strain, improving
     2. Lumbar spine sprain/strain, stable

     PLAN
     1. Continue physical therapy 2x/week for cervical spine
     2. Continue home exercises for lumbar spine...

   → parse_note_sections → 5 NoteSection objects
   → confidence_score = 0.85

6. REVIEW NODE (auto-approve)
   → final_note = generated_note

7. DELIVERY NODE
   → write_clinical_note(final_state, path="output/dictation/226748/generated_note_v4.md")
   → write_comparison(gold_note, generated_note, path="comparison_v4.md")

8. OUTPUT
   output/dictation/226748/generated_note_v4.md
   output/dictation/226748/comparison_v4.md
```

---

## 12. Quality Results

### Version History

| Version | Session | Changes | Score |
|---------|---------|---------|-------|
| v1 | 4 | Basic pipeline, no templates | ~3.5 |
| v2 | 5 | Specialty templates + vocabulary | 4.32 |
| v3 | 7 | EHR context + patient demographics | 4.34 |
| v4 | 8 | Provider profiles + vocab + style directives | 4.38 |

### v4 Dimension Detail (22 gold-standard samples)

| Dimension | v2 | v3 | v4 | Δ v2→v4 |
|-----------|----|----|----|----|
| Medical Accuracy | 4.00 | 4.00 | 4.00 | — |
| **Completeness** | 4.36 | 4.41 | **4.59** | **+0.23** |
| No Hallucination | 4.68 | 4.82 | 4.73 | +0.05 |
| Structure Compliance | 4.86 | 4.73 | 4.77 | -0.09 |
| Clinical Language | 4.09 | 4.05 | 4.09 | — |
| **Readability** | — | 3.91 | **4.00** | **+0.09** |
| **Overall** | **4.32** | **4.34** | **4.38** | **+0.06** |

**Key observations:**
- **Completeness** is the most improved dimension (+0.23 over 3 sessions). EHR context
  brought demographics; provider style directives drove more complete clinical detail.
- **Readability** lifted from 3.91 → 4.00 in v4 — Dr. Rahman's style directives
  (third person, past tense, spell out abbreviations) are measurably helping.
- **Medical Accuracy** is plateaued at 4.00 — limited by ASR error propagation and
  the 8B LLM currently in use. Will improve with WhisperX medical fine-tuning and
  Qwen 2.5-32B.
- **Floor raised:** v3 had one sample at 4.05 (readability=3.0). v4 floor is 4.10.

### Fact-Level Performance (v4)

Across 22 gold samples:
- **Diagnoses:** 45/50 correct (90%)
- **Medications:** 29/30 correct (97%)
- **Plan items:** near 100% (all samples scored ≥4.0 on structure)

---

## 13. Running the Pipeline

### Prerequisites

```bash
# 1. Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:latest          # dev/testing
ollama pull qwen2.5:32b              # production quality

# 2. Python environment
pip install -e ".[dev]"
# or: uv venv && uv sync

# 3. Environment
cp .env.example .env
# Set HF_TOKEN (required for pyannote diarization)
```

### Run a Single Encounter

```python
from orchestrator.graph import build_graph, run_encounter
from orchestrator.state import EncounterState, RecordingMode, DeliveryMethod
from config.provider_manager import get_provider_manager

graph = build_graph()
profile = get_provider_manager().load("dr_faraz_rahman")

state = EncounterState(
    provider_id=profile.id,
    patient_id="patient-001",
    provider_profile=profile,
    recording_mode=RecordingMode.DICTATION,
    delivery_method=DeliveryMethod.CLIPBOARD,
    audio_file_path="data/dictation/226748/dictation.mp3",
)

result = run_encounter(graph, state)
print(result.final_note.to_text())
```

### Batch Evaluation

```bash
# Run pipeline on all 33 samples → generated_note_v5.md + comparison_v5.md
set -a && source .env && set +a
python scripts/batch_eval.py --version v5

# Score with LLM-as-judge → quality_report_v5.md
python scripts/run_quality_sweep.py --version v5

# Regression tests (must pass before merging)
python -m pytest tests/quality/test_regression.py -v
```

### Extract Patient Context from Gold Notes

```bash
python scripts/extract_patient_context.py
# Creates patient_context.yaml in each data/*/sample_id/ directory
```

---

## 14. Sessions Roadmap

| Session | Focus | Status |
|---------|-------|--------|
| 1 | Project scaffolding, LangGraph skeleton, state schema | ✅ Done |
| 2 | Ollama LLM server + note generation | ✅ Done |
| 3 | WhisperX ASR + transcribe node | ✅ Done |
| 4 | End-to-end pipeline, batch eval v1 | ✅ Done |
| 5 | Templates + specialty dictionaries, batch eval v2 | ✅ Done |
| 6 | Quality evaluation framework, LLM-as-judge, regression tests | ✅ Done |
| 7 | EHR context integration, patient demographics, batch eval v3 | ✅ Done |
| 8 | Provider profiles, template routing, specialty vocab, batch eval v4 | ✅ Done |
| 9 | Learning loop + correction capture | ⬜ Next |
| 10 | S3 trigger pipeline (production ingestion) | ⬜ |
| 11 | Evidence-linked citations + offline buffering | ⬜ |
| 12 | FastAPI backend + audio endpoints | ⬜ |
| 13 | Web app — recording + review UI | ⬜ |
| 14 | Browser extension (Super Fill MVP) | ⬜ |
| 15+ | Coding suggestions, patient summaries, mobile, voice commands | ⬜ |

---

*Last updated: Session 8 — March 2026*
