# CLAUDE.md — AI Scribe Project Instructions

## What This Project Is

AI Scribe is a fully self-hosted, HIPAA-compliant medical documentation system. It converts doctor-patient conversations (ambient mode) and physician dictations (dictation mode) into structured clinical notes (SOAP, H&P, Progress Notes).

Every AI component is pluggable — ASR engines, LLMs, diarization, noise suppression — all behind MCP interfaces, swappable via config, never code changes.

## Architecture (read docs/architecture.md for full detail)

The encounter pipeline is a **LangGraph state graph** with 6 nodes:

```
CONTEXT → CAPTURE → TRANSCRIBE → NOTE → REVIEW → DELIVERY
                                                    ↓
                                              FEEDBACK LOOP
```

Each node has pre-processing → core execution → post-processing steps.
Each node connects to tools via **MCP tool servers**.
Engine selection is governed by **provider profiles** and an **engine registry**.

### Technology Stack (All Self-Hosted, All Open Source)

| Layer | Default | Alternatives |
|-------|---------|-------------|
| ASR (batch) | WhisperX (faster-whisper + pyannote 3.1 + wav2vec2) | NVIDIA Parakeet |
| ASR (streaming) | NVIDIA Nemotron-Speech-Streaming | WhisperX chunked |
| Diarization | pyannote 3.1 (via WhisperX) | NVIDIA Multitalker-Parakeet |
| Noise suppression | DeepFilterNet | RNNoise, passthrough |
| VAD | Silero VAD | WebRTC VAD |
| LLM inference | Ollama (OpenAI-compatible API) | vLLM, SGLang |
| LLM model | Qwen 2.5-14b (Apache 2.0) | Qwen 2.5-32B, Llama 3.1-70B, Mistral |
| Post-processing | 12-stage rule-based pipeline + 98K medical dict | ByT5 ML model (future) |
| Orchestration | LangGraph | — |
| Database | PostgreSQL | — |
| Object storage | MinIO (S3-compatible) | Local filesystem |
| Cache/queue | Valkey (Redis fork, BSD) | Redis |
| Web app | Next.js + React | — |
| Browser extension | Chrome MV3 | — |
| Monitoring | Grafana + Prometheus | — |

### MCP Tool Server Pattern

Every external capability is an MCP server. Nodes never call engines directly.

```python
# Example: LLM MCP server wraps Ollama's OpenAI-compatible API
# Switching to vLLM or Claude = change the URL in config/engines.yaml
# Same interface, same request/response format

class OllamaLLMServer(BaseLLMServer):
    def generate(self, system_prompt, messages, config) -> LLMResponse:
        # POST to http://localhost:11434/v1/chat/completions
        ...
```

All MCP servers implement a base interface defined in `mcp_servers/{type}/base.py`.

### Engine Configuration

```yaml
# config/engines.yaml controls ALL engine selection
# Zero code changes to swap engines

llm:
  default_server: ollama
  servers:
    ollama:
      url: "http://localhost:11434/v1"
      models:
        note_generation: "qwen2.5:14b"
        coding: "qwen2.5:14b"
        patient_summary: "qwen2.5:14b"
        command_parse: "qwen2.5:7b"

asr:
  default_server: whisperx
  servers:
    whisperx:
      model: "large-v3"
      device: "cuda"
      diarization: true
```

## Test Data

The `data/` folder contains real medical audio and transcripts for testing:

```
data/
├── dictation/           # Single-speaker physician dictations
│   ├── sample_001/
│   │   ├── audio.mp3
│   │   ├── transcript_medasr.txt    # MedASR CTC output (garbled)
│   │   ├── transcript_whisper.txt   # Whisper output (cleaner)
│   │   ├── soap_note.txt            # Gold-standard SOAP note
│   │   ├── patient_context.yaml     # Extracted demographics (Session 7)
│   │   ├── generated_note_v1.md     # Session 4 output (basic pipeline)
│   │   ├── generated_note_v2.md     # Session 5 output (templates + specialty)
│   │   ├── generated_note_v3.md     # Session 7 output (full context + demographics)
│   │   ├── comparison_v3.md         # Side-by-side: gold vs generated (per section table)
│   │   └── quality_report.md        # Quality evaluation with scores + fact check
│   └── ...
├── conversation/        # Multi-speaker ambient encounters
│   ├── sample_001/
│   │   ├── audio.mp3
│   │   ├── transcript.txt
│   │   ├── soap_note.txt            # Gold-standard SOAP note
│   │   ├── patient_context.yaml     # Extracted demographics (Session 7)
│   │   ├── generated_note_v1.md
│   │   ├── generated_note_v2.md
│   │   ├── generated_note_v3.md
│   │   ├── comparison_v3.md
│   │   └── quality_report.md
│   └── ...
└── README.md
```

**Output format:** ALL generated notes, transcripts, quality reports, and comparisons are
Markdown (.md) files with proper headers, sections, and tables. This makes them readable,
diffable, and directly comparable with the gold standard documents.

**Version tracking:** Generated outputs are versioned across sessions:
- `v1` = Session 4 (basic end-to-end, no templates) — ~3.5/5.0
- `v2` = Session 5 (templates + specialty dictionaries) — 4.30/5.0
- `v3` = Session 7 (full patient context + demographics) — 4.34/5.0
- `v4` = Session 8 (provider profiles + vocab + style directives) — 4.38/5.0
- `v5` = Session 10 (ASR inference knobs wired) — 4.35/5.0
- `v6` = Post-session model upgrade (qwen2.5:14b) — **4.44/5.0**
- `v7` = Dual-audio conversation processing — 4.31/5.0 (61 samples incl. new providers)
- `v8` = Multi-provider templates + profiles (chiro, neuro, ortho) — **4.35/5.0** (61 samples, 5 providers) ← current

Each version shows measurable improvement. Quality judged by llama3.1:latest as consistent cross-model evaluator.

**Standalone transcripts:** Each version also saves `audio_transcript_v{N}.txt` alongside the note in `output/{mode}/{sample_id}/`. This separates transcripts from SOAP notes cleanly.

## Existing Code to Integrate

The `postprocessor/` directory contains our existing ASR post-processing pipeline:

- `medasr_postprocessor.py` — 12-stage rule-based transcript cleanup (1,659 lines, production-tested)
- `medical_wordlist.txt` — 98K OpenMedSpel medical terms dictionary
- `ml/` — ML training pipeline for ByT5-based post-processor (future)

This pipeline runs as a post-processing step in the Transcribe Node. It corrects:
- CTC stutter pairs (90% reduction)
- Character-level stutters (99% reduction)  
- Non-dictionary medical terms (87% reduction)
- MedASR system artifacts ([unintelligible], {format commands}, fillers)

## Project Structure

```
ai-scribe/
├── CLAUDE.md                       # THIS FILE
├── README.md
├── pyproject.toml                  # Python project config (use uv or poetry)
├── docker-compose.yml              # Full local stack
│
├── config/
│   ├── engines.yaml                # Engine configuration (plug-and-play)
│   ├── quality_baseline.yaml       # Best-known quality parameters (Session 6)
│   ├── templates/                  # Note templates by specialty + visit type
│   │   ├── soap_default.yaml
│   │   ├── ortho_first_visit.yaml
│   │   ├── ortho_follow_up.yaml
│   │   ├── gi_first_visit.yaml
│   │   └── ... (auto-generated from gold note analysis)
│   ├── dictionaries/               # Specialty-specific medical vocabularies
│   │   ├── base_medical.txt        # 98K OpenMedSpel (symlinked from postprocessor/)
│   │   ├── orthopedic.txt
│   │   ├── gastroenterology.txt
│   │   ├── cardiology.txt
│   │   ├── neurology.txt
│   │   ├── psychiatry.txt
│   │   ├── pediatrics.txt
│   │   └── custom/                 # Provider-specific custom terms
│   │       └── {provider_id}.txt
│   └── prompts/                    # LLM system prompts
│       ├── note_generation.yaml
│       ├── coding_suggestion.yaml
│       └── patient_summary.yaml
│
├── orchestrator/                   # LangGraph encounter pipeline
│   ├── graph.py                    # Top-level encounter graph
│   ├── state.py                    # EncounterState Pydantic schema
│   ├── nodes/                      # Pipeline stage implementations
│   │   ├── context_node.py
│   │   ├── capture_node.py
│   │   ├── transcribe_node.py
│   │   ├── note_node.py
│   │   ├── review_node.py
│   │   └── delivery_node.py
│   ├── edges/                      # Conditional routing
│   │   ├── asr_router.py
│   │   ├── llm_router.py
│   │   ├── confidence_router.py
│   │   └── mode_router.py
│   └── subgraphs/
│       ├── postprocessor_graph.py  # Rules vs ML routing
│       └── note_parallel_graph.py  # Parallel note+coding+summary
│
├── mcp_servers/                    # Plug-and-play tool servers
│   ├── registry.py                 # Engine registry + health checks
│   ├── asr/
│   │   ├── base.py                 # ASREngine interface
│   │   ├── whisperx_server.py      # DEFAULT
│   │   ├── nemo_streaming_server.py
│   │   ├── assemblyai_server.py    # Cloud option
│   │   └── deepgram_server.py      # Cloud option
│   ├── llm/
│   │   ├── base.py                 # LLMEngine interface
│   │   ├── ollama_server.py        # DEFAULT
│   │   ├── vllm_server.py
│   │   └── claude_server.py        # Cloud option
│   ├── ehr/
│   │   ├── base.py                 # EHRAdapter interface
│   │   ├── stub_server.py          # DEFAULT — reads from local YAML files
│   │   ├── fhir_server.py          # Future: live FHIR R4 integration
│   │   ├── extension_server.py     # Future: browser extension bridge
│   │   └── manual_server.py        # Future: manual entry
│   ├── audio/
│   │   ├── deepfilternet_server.py
│   │   ├── silero_vad_server.py
│   │   └── ffmpeg_server.py
│   └── data/
│       ├── medical_dict_server.py  # Base + specialty dictionary loader
│       ├── template_server.py
│       └── icd10_server.py
│
├── postprocessor/                  # ASR post-processing (EXISTING CODE)
│   ├── medasr_postprocessor.py     # 12-stage rule-based pipeline
│   ├── medical_wordlist.txt        # 98K OpenMedSpel dictionary
│   └── ml/                         # ML post-processor (future)
│       ├── generate_training_data.py
│       ├── train_model.py
│       ├── inference.py
│       └── evaluate.py
│
├── learning/                       # Continuous improvement
│   ├── correction_capture.py
│   ├── style_model.py
│   ├── quality_monitor.py
│   └── ab_test.py
│
├── quality/                        # Quality evaluation framework (Session 6)
│   ├── __init__.py
│   ├── evaluator.py                # Central quality scoring (LLM-as-judge + fact extraction)
│   ├── report.py                   # Per-encounter and aggregate quality reports (.md)
│   ├── optimizer.py                # Parameter optimization loop
│   ├── dimensions.py               # Scoring rubric definitions
│   ├── fact_extractor.py           # Extract medications, diagnoses, findings from notes
│   ├── experiments.jsonl            # Log of all optimization experiments
│   └── regression.py               # Quality regression testing
│
├── output/                         # Markdown output writer + batch reports
│   ├── __init__.py
│   ├── markdown_writer.py          # Renders clinical notes as formatted .md files
│   ├── comparison_writer.py        # Renders side-by-side gold vs generated .md files
│   └── batch_report_v{N}.md        # Aggregate quality reports per pipeline version
│
├── trigger/                        # Production ingestion triggers (Session 10)
│   ├── s3_handler.py               # S3 upload → pipeline invocation
│   └── eventbridge_config.json     # EventBridge rule definition
│
├── api/                            # FastAPI backend (Session 12)
│   ├── main.py
│   ├── routes/
│   │   ├── encounters.py
│   │   ├── providers.py
│   │   ├── audio.py
│   │   └── notes.py
│   └── ws/
│       ├── audio_stream.py
│       └── session_events.py
│
├── client/                         # Frontend (Sessions 13-14)
│   ├── web/                        # Next.js web application
│   └── extension/                  # Chrome browser extension
│
├── data/                           # Test data (gitignored, local only)
│   ├── dictation/
│   │   └── sample_001/
│   │       ├── audio.mp3
│   │       ├── soap_note.txt               # Gold standard
│   │       ├── patient_context.yaml        # Extracted demographics (Session 7)
│   │       ├── generated_note_v1.md        # Session 4 output
│   │       ├── generated_note_v2.md        # Session 5 output (templates)
│   │       ├── generated_note_v3.md        # Session 7 output (full context)
│   │       ├── comparison_v3.md            # Side-by-side vs gold
│   │       └── quality_report.md           # Quality evaluation
│   ├── conversation/
│   │   └── (same structure as dictation/)
│   └── README.md
│
├── scripts/
│   ├── setup.sh                    # Project setup
│   ├── pull_models.sh              # ollama pull required models
│   ├── benchmark_asr.py            # ASR engine comparison
│   ├── analyze_gold_notes.py       # Analyze gold notes → generate templates (Session 5)
│   ├── build_specialty_dicts.py    # Generate specialty dictionaries (Session 5)
│   ├── extract_patient_context.py  # Extract demographics from gold notes (Session 7)
│   ├── run_quality_sweep.py        # Run pipeline + evaluate all samples (Session 6)
│   ├── compare_versions.py         # Side-by-side output comparison (Session 6)
│   ├── batch_eval.py               # Batch pipeline runner (use --two-pass for VRAM safety)
│   ├── backfill_transcripts.py     # Backfill audio_transcript_v{N}.txt from cache
│   └── generate_architecture_diagram.py  # Graphviz architecture diagram generator
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── deploy/
│   ├── Dockerfile.api
│   ├── Dockerfile.gpu
│   └── docker-compose.yml
│
└── docs/
    ├── architecture.md             # Full system architecture (14 sections)
    ├── implementation.md           # Implementation notes
    └── images/
        ├── architecture_diagram.png  # Programmatic architecture diagram
        └── architecture_diagram.dot  # Graphviz source
```

## Build Sequence

Follow this order. Each session builds on the previous.

**Sessions 1-8 focus on pipeline quality — getting the best possible clinical note from audio.**
**Sessions 9+ add infrastructure, UI, and integration layers on top of the proven pipeline.**

### Output Format Convention

ALL generated outputs are **Markdown (.md) files** with proper headers, sections, and formatting so they are human-readable and diff-comparable:

```markdown
# Clinical Note — [Patient Name / ID]
**Date:** 2024-03-07 | **Provider:** Dr. Smith | **Visit Type:** Follow-up
**Specialty:** Orthopedic | **Template:** ortho_follow_up

---

## Chief Complaint
Patient presents for follow-up evaluation of cervical spine injury...

## History of Present Illness
...

## Assessment
1. Acute cervical strain, improving
2. Right wrist sprain, stable
...

## Plan
1. Continue physical therapy 2x/week
2. Naproxen 500mg BID as needed
...
```

**Comparison documents** are also Markdown with side-by-side sections:

```markdown
# Quality Comparison — sample_001
**Version:** v3 (Session 7) vs Gold Standard

---

## Chief Complaint
| Gold | Generated | Score |
|------|-----------|-------|
| Patient presents for initial evaluation of injuries sustained in MVA on 2/19... | Patient presents for initial evaluation of motor vehicle accident injuries... | 4.5/5 |

## History of Present Illness
| Gold | Generated | Score |
|------|-----------|-------|
| (gold text) | (generated text) | 4.2/5 |

...

## Summary
| Dimension | Score |
|-----------|-------|
| Medical Accuracy | 4.5 |
| Completeness | 4.0 |
| Hallucination-Free | 5.0 |
| **Overall** | **4.3** |
```

### Session 1: Project Scaffolding + State Schema + LangGraph Skeleton
- Initialize Python project with pyproject.toml (use Python 3.11+)
- Define EncounterState Pydantic schema in `orchestrator/state.py`
- Create the top-level LangGraph in `orchestrator/graph.py` with stub nodes
- Create base interfaces: `mcp_servers/asr/base.py`, `mcp_servers/llm/base.py`, `mcp_servers/ehr/base.py`
- Create `config/engines.yaml` with default configuration
- Wire up basic test: state flows through all 6 stub nodes
- Implement Markdown output writer in `output/markdown_writer.py` — all generated notes are `.md` files

### Session 2: Ollama LLM Server + Note Generation
- Implement `mcp_servers/llm/ollama_server.py` (wraps Ollama OpenAI-compat API)
- Implement `mcp_servers/llm/base.py` LLMEngine interface
- Create prompt templates in `config/prompts/note_generation.yaml`
- Implement `orchestrator/nodes/note_node.py` (prompt assembly → LLM call → parse)
- Output format: generated notes written as `.md` files with proper clinical headers
- Test: feed a transcript string → get a formatted Markdown SOAP note back
- Use test data from `data/dictation/` transcripts as input

### Session 3: WhisperX ASR Server + Transcribe Node
- Implement `mcp_servers/asr/whisperx_server.py`
- Implement `mcp_servers/asr/base.py` ASREngine interface
- Implement `orchestrator/nodes/transcribe_node.py`
- Integrate post-processor: `postprocessor/medasr_postprocessor.py`
- Output format: cleaned transcripts written as `.md` with speaker labels and timestamps
- Test: feed audio file → get cleaned, diarized transcript `.md`
- Use test audio from `data/dictation/` and `data/conversation/`

### Session 4: End-to-End Pipeline
- Wire Transcribe → Note nodes together in the graph
- Implement ASR Router (conditional edge) with single engine initially
- Implement LLM Router with single engine
- Test: audio file → transcript → SOAP note `.md` (full pipeline)
- Generate notes for ALL samples in `data/dictation/` and `data/conversation/`
- Save as `generated_note_v1.md` alongside each sample
- Generate `comparison_v1.md` for each sample showing generated vs gold side-by-side
- Save aggregate report as `output/batch_report_v1.md`

### Session 5: Templates & Specialization
This session makes the note generation specialty-aware and template-driven.

**a. Generic Template Engine**
- Implement template system in `config/templates/` that supports arbitrary note formats
- Each template is a YAML file defining: sections (ordered), required fields, section prompts, formatting rules, and demographic/patient header fields
- Template types include but are not limited to: first_visit, follow_up, procedure_note, consultation, discharge_summary
- The template engine is GENERIC — any new YAML template dropped into `config/templates/` is automatically available
- Templates are selected per-encounter based on: visit type + provider specialty + provider preference
- Implement in `mcp_servers/data/template_server.py` and wire into Note Node's prompt assembly
- Template structure:
  ```yaml
  # config/templates/ortho_first_visit.yaml
  name: "Orthopedic First Visit"
  specialty: orthopedic
  visit_type: first_visit
  header_fields:
    - patient_name
    - date_of_service
    - date_of_birth
    - provider_name
    - referring_provider
    - location
  sections:
    - id: chief_complaint
      label: "Chief Complaint"
      required: true
      prompt_hint: "One-sentence reason for visit"
    - id: hpi
      label: "History of Present Illness"
      required: true
      prompt_hint: "Detailed narrative including: mechanism of injury, date of onset, progression, aggravating/alleviating factors, prior treatment"
    - id: past_medical_history
      label: "Past Medical History"
      required: true
    - id: past_surgical_history
      label: "Past Surgical History"
      required: true
    - id: medications
      label: "Current Medications"
      required: true
    - id: allergies
      label: "Allergies"
      required: true
    - id: social_history
      label: "Social History"
      required: false
    - id: review_of_systems
      label: "Review of Systems"
      required: true
    - id: physical_exam
      label: "Physical Examination"
      required: true
      prompt_hint: "Include inspection, palpation, ROM (in degrees), strength, special tests, neurovascular status"
    - id: imaging
      label: "Imaging / Diagnostics"
      required: false
    - id: assessment
      label: "Assessment"
      required: true
      prompt_hint: "Numbered list of diagnoses with ICD-10 codes if identifiable"
    - id: plan
      label: "Plan"
      required: true
      prompt_hint: "Numbered plan items corresponding to each assessment item"
  formatting:
    voice: active
    abbreviations: spell_out
    measurements: include_units
  ```

**b. Gold Document Analysis + Template Creation**
- Write `scripts/analyze_gold_notes.py` that reads ALL gold-standard notes from both `data/dictation/` and `data/conversation/`
- Analyze the gold documents using the LLM (via Ollama) to extract:
  - Common section structure (which sections appear, in what order)
  - Section-level patterns (how HPI is written, what Assessment typically includes)
  - Style patterns (voice, tense, abbreviation usage, level of detail)
  - Differences between first visit vs follow-up patterns
  - Differences between dictation-sourced vs conversation-sourced notes
  - Patient/demographic fields present in headers
- Generate template YAML files from this analysis — one per detected note type
- Generate example style directives from the analysis
- Run the full pipeline on ALL samples using the generated templates
- Save as `generated_note_v2.md` for each sample
- Generate `comparison_v2.md` with side-by-side Markdown tables (gold vs generated per section)
- Save aggregate report as `output/batch_report_v2.md`

**c. Specialty Dictionaries**
- Create `config/dictionaries/` folder for specialty-specific medical vocabularies
- Implement specialty dictionary loader in `mcp_servers/data/medical_dict_server.py`
- Structure:
  ```
  config/dictionaries/
  ├── base_medical.txt            # 98K OpenMedSpel (already have this)
  ├── orthopedic.txt              # ROM terms, joint anatomy, implants, procedures
  ├── gastroenterology.txt        # Endoscopy, hepatology, motility terms
  ├── cardiology.txt              # Arrhythmias, catheterization, cardiac meds
  ├── neurology.txt               # Cranial nerves, EEG, stroke scales
  ├── psychiatry.txt              # DSM terms, mental status exam vocabulary
  ├── pediatrics.txt              # Growth/development, immunizations
  └── custom/                     # Provider-specific terms
      └── {provider_id}.txt
  ```
- Build specialty dictionaries by:
  1. Mining gold SOAP notes for specialty-specific terms not in base dictionary
  2. Supplementing from public medical terminology databases
  3. Using the LLM to generate comprehensive term lists per specialty
- Wire specialty dictionaries into:
  - Post-processor (Stage 10-11: dictionary matching + medical spellcheck)
  - Note Node (LLM prompt: specialty vocabulary context)
  - ASR (WhisperX initial_prompt / hotwords)

**d. Full Pipeline Execution with Specialty Context**
- Run pipeline on ALL samples with specialty context loaded
- All outputs as properly formatted Markdown clinical notes
- Save all outputs with version tracking for comparison across sessions

### Session 6: Output Quality Measurement & Optimization
This session builds the quality evaluation framework and iteratively tunes the pipeline.

**a. Quality Evaluation Framework**
- Build `quality/evaluator.py` — central quality measurement system
- All quality reports generated as **Markdown files**
- Implement multiple evaluation approaches (word-matching does NOT work for clinical notes):

  **Semantic Similarity (primary metric):**
  - Use the local LLM as a judge: score (gold_note, generated_note) pairs
  - Scoring rubric (each 1-5): Medical accuracy, Completeness, No hallucination, Structure compliance, Clinical language, Readability
  - Overall score = weighted combination

  **Section-Level Evaluation:**
  - Compare section-by-section with Markdown table output
  - For each section: semantic similarity score via LLM judge
  - Identify: missing sections, extra sections, wrong content

  **Factual Extraction Comparison:**
  - Extract key facts from both notes using LLM: medications (name, dose, frequency), diagnoses, physical exam findings, plan items
  - Compute precision/recall on extracted facts
  - Catches: "note reads well but missed a medication"

  **Transcript Coverage:**
  - What % of clinically relevant transcript content is in the note?
  - Flag important transcript segments missing from note

- Quality report per encounter saved as `quality_report.md`:
  ```markdown
  # Quality Report — sample_001
  **Pipeline Version:** v2 | **Date:** 2024-03-08

  ## Overall Score: 4.2 / 5.0

  ## Dimension Scores
  | Dimension | Score | Notes |
  |-----------|-------|-------|
  | Medical accuracy | 4.5 | |
  | Completeness | 3.8 | ⚠ Missing: left knee ROM measurements |
  | No hallucination | 5.0 | |
  | Structure compliance | 4.0 | |
  | Clinical language | 4.5 | |
  | Readability | 4.0 | |

  ## Fact Check
  | Category | Found | Total | Missed |
  |----------|-------|-------|--------|
  | Medications | 5 | 5 | — |
  | Diagnoses | 3 | 3 | — |
  | Exam findings | 8 | 10 | left knee crepitus, patellar grind |
  | Plan items | 4 | 4 | — |

  ## Section Comparison
  (side-by-side Markdown table of gold vs generated per section)
  ```

**b. Parameter Optimization Loop**
- Build `quality/optimizer.py` — systematically tunes all pipeline parameters
- Parameters: post-processor thresholds, LLM temperature/top_p, prompt variations, context budget, few-shot examples, ASR beam_size, VAD sensitivity
- Strategy: run full pipeline → identify weakest dimensions → LLM analyzes failures → suggest parameter changes → re-run → compare → accept/revert
- All experiments logged in `quality/experiments.jsonl`
- NOT grid search — use LLM to reason about failure patterns

**c. Batch Execution & Reporting**
- `scripts/run_quality_sweep.py`: runs pipeline on all samples, evaluates, produces:
  - Per-sample: `quality_report.md` + `comparison_v{N}.md`
  - Aggregate: `output/batch_report_v{N}.md` with summary statistics
  - `output/worst_samples.md` identifying samples needing most improvement
- `scripts/compare_versions.py`: side-by-side across versions as Markdown

**d. Quality Regression Prevention**
- Save best parameters as `config/quality_baseline.yaml`
- `tests/quality/test_regression.py`: assert scores >= baseline before code changes

### Session 7: EHR Context Integration (Stubbed)
This session adds patient demographics and clinical context into the pipeline so generated notes include ALL information present in the gold documents — not just what's in the audio transcript.

**a. Patient Context Extraction from Gold Documents**
- Write `scripts/extract_patient_context.py` that analyzes all gold-standard notes to extract structured patient/demographic data:
  - Patient name (or identifier)
  - Date of birth / age
  - Sex / gender
  - Date of service
  - Date of injury / onset (if applicable)
  - Provider name and credentials
  - Referring provider (if applicable)
  - Clinic / facility name and location
  - Insurance / claim information (if present)
  - Visit type (initial evaluation, follow-up, re-evaluation)
  - Any other header/demographic fields present in gold notes
- Use the LLM to extract this structured data from each gold note
- Save extracted context as `patient_context.yaml` alongside each sample:
  ```yaml
  # data/dictation/sample_001/patient_context.yaml
  patient:
    name: "Jane Doe"
    date_of_birth: "1985-06-15"
    age: 38
    sex: "Female"
    mrn: "0219-370"
  encounter:
    date_of_service: "2024-03-07"
    visit_type: "initial_evaluation"
    date_of_injury: "2024-02-19"
    mechanism_of_injury: "Motor vehicle accident"
  provider:
    name: "Dr. Smith"
    credentials: "MD"
    specialty: "Orthopedic"
    npi: ""
  facility:
    name: "Regional Medical Center"
    location: ""
  referring_provider:
    name: ""
  insurance:
    carrier: ""
    claim_number: ""
  ```

**b. Context Node Implementation**
- Implement `orchestrator/nodes/context_node.py` fully
- For now, the EHR adapter is stubbed: reads patient context from `patient_context.yaml` files
- The stub EHR adapter (`mcp_servers/ehr/stub_server.py`) implements the full EHR interface but reads from local YAML/JSON files instead of a live EHR
- Context is loaded into EncounterState.context_packet before the pipeline runs
- This context flows through to the Note Node's prompt assembly

**c. Template Updates for Demographics**
- Update all templates to include demographic header fields
- The Note Node prompt assembler now includes patient context:
  ```
  PATIENT CONTEXT:
  Name: Jane Doe | DOB: 06/15/1985 | Sex: Female
  Date of Service: 03/07/2024 | Visit Type: Initial Evaluation
  Date of Injury: 02/19/2024 | Mechanism: Motor vehicle accident
  Provider: Dr. Smith, MD | Facility: Regional Medical Center

  TRANSCRIPT:
  (cleaned transcript from Transcribe Node)

  Generate a clinical note using the template below...
  ```
- Generated Markdown notes now include a complete header:
  ```markdown
  # Clinical Note
  **Patient:** Jane Doe | **DOB:** 06/15/1985 | **Sex:** Female
  **Date of Service:** 03/07/2024 | **Visit Type:** Initial Evaluation
  **Date of Injury:** 02/19/2024 | **Mechanism:** MVA
  **Provider:** Dr. Smith, MD | **Specialty:** Orthopedic
  **Facility:** Regional Medical Center

  ---

  ## Chief Complaint
  ...
  ```

**d. Full Pipeline Execution with Complete Context**
- Run the complete pipeline on ALL samples with:
  - Patient context loaded from extracted YAML
  - Specialty dictionary loaded based on provider specialty
  - Appropriate template selected based on visit type + specialty
- The goal: generated notes should include ALL details present in the gold documents
  - Demographics in header ✓
  - Dates, provider info, facility ✓
  - All clinical sections with content from both transcript AND context ✓
- Save as `generated_note_v3.md` for each sample
- Generate `comparison_v3.md` with side-by-side comparison against gold
- Quality evaluation: run full quality sweep → `output/batch_report_v3.md`
- This version should score significantly higher on completeness and structure compliance since it now has the same information the gold notes had

**e. Stub EHR Interface for Future Integration**
- `mcp_servers/ehr/stub_server.py` implements the full EHRAdapter interface:
  ```python
  class StubEHRServer(BaseEHRServer):
      """Reads patient context from local YAML files.
      Replace with FHIRServer, HL7Server, or ExtensionServer later.
      Same interface — zero code changes in the pipeline."""

      async def get_patient(self, identifier) -> PatientDemographics
      async def get_problem_list(self, patient_id) -> list[Problem]
      async def get_medications(self, patient_id) -> list[Medication]
      async def get_allergies(self, patient_id) -> list[Allergy]
      async def get_recent_labs(self, patient_id) -> list[LabResult]
      async def get_last_visit_note(self, patient_id) -> ClinicalNote
      async def push_note(self, patient_id, note) -> PushResult
  ```
- When real EHR integration is built (later sessions), only the MCP server implementation changes — the pipeline code, prompts, and templates are untouched

### Session 8: Provider Profiles + Provider-Specific Tuning
- Implement provider profile CRUD (PostgreSQL or local YAML for now)
- Provider-specific template selection (specialty + visit type → template)
- Provider-specific dictionary loading (specialty + custom vocabulary)
- Style directive injection into LLM prompts (from provider profile)
- Custom vocabulary per provider → ASR keyterm boosting
- Per-provider quality tracking: store quality scores per provider, track improvement

### Session 9: FastAPI Backend + Web UI Viewer
This session wires the pipeline to a web interface so all results can be browsed, reviewed, and triggered from a browser. The UI is a **viewer and dashboard only** — audio upload is supported but live recording is deferred to a later session.

**Design System — Talisman Solutions Style**
- Color palette: primary green `#00B27A`, accent purple/indigo `#6366F1`, neutral dark sidebar `#1E1B4B`
- Card-based KPI tiles, clean sidebar navigation, combo bar+line charts for quality trends
- Typography: Inter (sans-serif), 14px base, generous whitespace
- All charts: Recharts (React); all tables sortable; dark sidebar + light main panel

**a. FastAPI Backend (`api/`)**
- `api/main.py` — FastAPI app with CORS, structured logging, startup/shutdown lifecycle
- Encounter lifecycle routes (`api/routes/encounters.py`):
  - `POST /encounters` — create encounter (provider_id, patient_id, visit_type, mode)
  - `POST /encounters/{id}/upload` — upload audio file (multipart); triggers pipeline async
  - `GET  /encounters/{id}` — poll status (pending / processing / complete / error)
  - `GET  /encounters/{id}/transcript` — return cleaned transcript (Markdown)
  - `GET  /encounters/{id}/note` — return generated clinical note (Markdown)
  - `GET  /encounters/{id}/comparison` — return gold vs generated comparison (Markdown)
  - `GET  /encounters/{id}/quality` — return quality report JSON + Markdown
- Provider routes (`api/routes/providers.py`):
  - `GET  /providers` — list all provider profiles
  - `GET  /providers/{id}` — get profile + quality history
  - `GET  /providers/{id}/quality-trend` — quality history as chart-ready JSON
- Quality routes (`api/routes/quality.py`):
  - `GET  /quality/aggregate` — overall stats across all samples + versions
  - `GET  /quality/samples` — per-sample scores (filterable by version, provider, visit_type)
  - `GET  /quality/dimensions` — dimension breakdown averages
- WebSocket (`api/ws/session_events.py`): `WS /ws/encounters/{id}` — real-time pipeline progress events (stage name, % complete, log lines)
- Background task runner: upload handler enqueues pipeline run via `asyncio.create_task`; sends WS events at each LangGraph node transition
- Data source: reads existing `output/` directory structure + provider YAML files; no database required for this session

**b. Next.js Web App (`client/web/`)**
- `npx create-next-app@latest` with TypeScript + Tailwind CSS
- Package additions: `recharts`, `react-markdown`, `react-syntax-highlighter`, `@radix-ui/react-tabs`, `lucide-react`
- Sidebar navigation: Dashboard | Samples | Providers | Upload

**Dashboard (`/`)**
- KPI cards (4 across): Total Samples, Average Quality Score, Best Sample, Latest Version
- Quality trend chart: line chart of avg score per version (v1 → v2 → v3 → v4)
- Dimension radar chart: accuracy / completeness / no-hallucination / structure / language / readability
- Recent encounters table: sample_id, visit_type, version, score, date

**Sample Browser (`/samples`)**
- Filterable table: version, visit_type (dictation/conversation), score range
- Row click → Sample Detail page

**Sample Detail (`/samples/[id]`)**
- Tabs: Transcript | Clinical Note | Comparison | Quality Report
- Transcript tab: Markdown renderer with speaker labels (DOCTOR / PATIENT) highlighted in green/indigo
- Clinical Note tab: rendered Markdown with section headers styled as cards
- Comparison tab: two-column table (Gold | Generated) with per-section score badge
- Quality Report tab: dimension score bar chart + fact-check table
- "Re-run Pipeline" button → `POST /encounters` then polls WS for progress

**Upload (`/upload`)**
- Drag-and-drop audio file zone (MP3, WAV, M4A)
- Provider selector (dropdown from `/providers`)
- Visit type selector (initial_evaluation / follow_up)
- Submit → `POST /encounters` + `POST /encounters/{id}/upload` → redirects to Sample Detail with live WS progress bar

**Providers (`/providers`)**
- Provider cards: name, specialty, latest quality score badge
- Provider detail: style directives list, custom vocabulary chips, quality history line chart

**c. Integration**
- API base URL configured via `NEXT_PUBLIC_API_URL` env var (default `http://localhost:8000`)
- All Markdown content rendered with `react-markdown` + `remark-gfm` for tables
- WebSocket connection managed with `useWebSocket` custom hook; progress shown as animated step indicator

**d. Dev Setup**
- `api/`: `uvicorn api.main:app --reload --port 8000`
- `client/web/`: `npm run dev` (port 3000)
- Add both to `docker-compose.yml` for one-command startup

### Session 10: ASR Quality Improvement — Physician-Specific Fine-Tuning ✓ COMPLETE
This session improves transcription accuracy for individual physicians by fine-tuning Whisper on provider-specific audio and notes, capturing idiosyncratic pronunciation, accent, vocabulary, and dictation style.

**Session 10 outcomes:**
- LoRA fine-tuning implemented and evaluated. **Verdict: opt-in only** (`use_lora=True` required).
  - Dictation: +23–27% WER degradation (labels were SOAP notes, not verbatim transcripts — LoRA learned to summarise, not transcribe)
  - Ambient: −11–13% WER improvement (multi-speaker data gave better verbatim signal)
  - Data flywheel thresholds: ≥30 min verbatim = minimum viable; 1–2 hr = sweet spot; 2–5 hr = strong adaptation
- Dictation-specific inference knobs wired end-to-end through `ASRConfig` → `WhisperXServer.transcribe()`:
  - `condition_on_previous_text=True` for dictation, `False` for ambient
  - `hotwords` = top 100 provider custom vocabulary terms (direct logit boost)
  - `beam_size`, `no_speech_threshold`, `compression_ratio_threshold`, `vad_threshold` all per-request
  - Per-request options swap uses `dataclasses.replace()` + `threading.Lock()` — thread-safe for concurrent encounters
- Pipeline v5: **4.35/5.0** quality (stable vs v4=4.38; delta within judge noise ±0.1)
- Ambient-mode ASR optimizations deferred to Session 10b (see below)

**NOTE:** Ambient ASR optimizations (`condition_on_previous_text=False` is already done; remaining: `max_speakers` provider-tunable, ambient-specific `vad_threshold` defaults) are deferred to a standalone session.

**a. Data Preparation Pipeline (`scripts/prepare_asr_training_data.py`)**
- For each sample pair (audio + gold note / transcript):
  - Align gold transcript words to audio using WhisperX forced alignment (wav2vec2)
  - Produce word-level `(start_time, end_time, word)` segments
  - Segment audio into 10–30 second chunks with corresponding text
  - Output: HuggingFace `datasets`-compatible `DatasetDict` with `{audio, sentence}` fields
- Filtering: skip chunks with alignment confidence < 0.6 or duration > 30s
- Augmentation: speed perturbation ±10%, additive noise at SNR 15–25 dB (using `audiomentations`)
- Save to `data/asr_training/{provider_id}/` in HuggingFace Arrow format

**b. LoRA Adapter Fine-Tuning (`scripts/finetune_whisper_lora.py`)**
- Base model: `openai/whisper-large-v3` (same as production)
- Method: LoRA via `peft` library — fine-tune only attention + MLP projection layers
  - `r=8`, `lora_alpha=32`, `target_modules=["q_proj","v_proj"]`, `lora_dropout=0.05`
- Training: HuggingFace `Seq2SeqTrainer`
  - `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`, `fp16=True`
  - `learning_rate=1e-4`, `warmup_steps=50`, cosine LR schedule
  - `max_steps=200` (sufficient for ~21 samples with augmentation = ~1,000 chunks)
  - Eval every 50 steps on held-out 10% split; early stop if WER stops improving
- Saves adapter weights to `models/whisper_lora/{provider_id}/` (< 50 MB per provider)
- Full model NOT saved — only LoRA delta weights; base model shared across all providers

**c. Provider-Specific ASR Server (`mcp_servers/asr/whisperx_lora_server.py`)**
- Extends `WhisperXServer`; on initialization loads LoRA adapter via `peft.PeftModel.from_pretrained`
- Engine registry detects if `models/whisper_lora/{provider_id}/` exists → uses LoRA server
- Falls back to base WhisperX server if no adapter exists
- Zero pipeline code changes — same `ASREngine` interface

**d. Evaluation & Comparison (`scripts/eval_asr_quality.py`)**
- Metric: Word Error Rate (WER) and Character Error Rate (CER) on held-out audio
- Compare: base Whisper vs LoRA-fine-tuned on same audio
- Output: `output/asr_eval_{provider_id}.md` — WER/CER table, worst-error examples, medical term accuracy breakdown
- Medical term accuracy: compute separately for terms in provider's custom_vocabulary + specialty dict

**e. Hotword / Keyword Boosting (lighter-weight alternative)**
- For providers where < 5 minutes of audio is available (insufficient for fine-tuning):
  - Use WhisperX `initial_prompt` with 200-word provider-specific vocabulary
  - Use `initial_prompt` to prime the model with the provider's name, clinic, common phrases
  - No training required; immediate improvement for rare medical terms
- `mcp_servers/asr/whisperx_server.py`: `_build_initial_prompt(provider_profile)` constructs the prompt string from `provider.custom_vocabulary + specialty_hotwords[:50]`

**f. Continuous Improvement Hook**
- After each completed encounter, if provider has manually corrected the transcript:
  - New audio + corrected text is added to the training dataset
  - Re-fine-tuning triggered automatically when `new_samples >= 5` (configurable)
  - New LoRA adapter replaces old one; quality metrics compared before activation

### Session 10b: Ambient ASR Optimizations
Deferred from Session 10. Wire ambient-specific inference knobs that were left incomplete:
- `max_speakers`: make provider-tunable (currently hardcoded to 5 for ambient); add to provider profile `asr_overrides`
- `vad_threshold` defaults: lower default (0.3) for ambient to capture soft-spoken patients
- Evaluate WER improvement on ambient samples after these changes
- (LoRA for ambient: also deferred until ≥30 min verbatim ambient transcripts are accumulated via correction-capture loop)

### Session 10c: LLM Model Upgrade + Multi-Version UI ✓ COMPLETE
This session benchmarked alternative LLM models and upgraded the default, then added full
version-comparison capabilities to the web UI.

**Model Benchmarking:**
- Tested: openbiollm:8b, MedGemma, mistral:latest, qwen2.5:14b, qwen2.5:32b
- openbiollm and MedGemma ruled out: medical Q&A chatbots, ignore system prompt formatting
- Mistral below baseline (25% keyword overlap vs 33% for llama3.1)
- qwen2.5:14b = qwen2.5:32b in quality but 5× faster; selected as default
- Fair comparison methodology: always use llama3.1:latest as judge (--judge-model flag)
- Result: **v6 = 4.44/5.0** (+0.09 vs v5=4.35)

**Two-Pass Batch Architecture (VRAM management):**
- A10G (23 GB): WhisperX peaks 10-12 GB + qwen2.5:14b ~8 GB = cannot coexist
- Solution: `--two-pass` flag in `scripts/batch_eval.py`
  - Pass 1: ASR only — `_NoOpLLM` stub (zero VRAM for LLM), saves `transcript_cache_v{N}.json`
  - Pass 2: LLM only from cache — WhisperX fully freed before LLM loads
- `keep_alive=0` in OllamaServer forces model unload from VRAM after each response

**Standalone Transcript Files:**
- All generated notes now also save `audio_transcript_v{N}.txt` per sample per version
- `scripts/backfill_transcripts.py`: backfills v1-v5 from transcript cache for all 33 samples
- API: new endpoints `GET /encounters/{id}/transcript` and `GET /encounters/{id}/audio`

**Multi-Version Web UI:**
- Sample Detail page now has 6 tabs: Transcript, Clinical Note, Comparison, Gold Standard, Quality Scores, Compare Versions
- Transcript tab: HTML5 audio player (play/pause, scrubber, time) + version picker (v1–v6)
- Clinical Note tab: version picker with client-side switching via fetchNote()
- Compare Versions tab: dual version pickers + LCS-based line diff (green/red/gray)
- Architecture diagram: `scripts/generate_architecture_diagram.py` → `docs/images/architecture_diagram.png`

### Session 11: Onboard New Practice Types, Templates, and Providers
This session adds admin-facing UX for managing specialties, templates, and providers — making the platform self-service for onboarding without config file edits.

**a. Specialty Management**
- Add "Specialties" link to left navigation sidebar
- Specialties list page: shows all supported specialty types (orthopedic, chiropractic, neurology, etc.)
- Specialty detail page: read-only view of the keyword dictionary associated with that specialty (`config/dictionaries/{specialty}.txt`)
- Inline editing: toggle to edit the dictionary directly in the browser (saves back to the dictionary file)
- Add new specialty: form to create a new specialty — user can enter keywords manually or upload a `.txt` file
- Backend API: `GET/POST/PUT /specialties`, `GET/PUT /specialties/{id}/dictionary`
- Creating a specialty creates the dictionary file in `config/dictionaries/`

**b. Template Management**
- Add "Templates" link to left navigation sidebar
- Templates list page: shows all templates with attributes (specialty, visit type, associated providers)
- Template detail: shows sections, formatting rules, and which providers use this template
- Create template form: select specialty (dropdown), visit type, and define header/section list
  - Section list builder: add/remove/reorder sections, each with id, label, required flag, and prompt hint
  - Available section types pre-populated from existing templates (e.g., chief_complaint, history_of_present_illness, physical_examination, assessment, plan, etc.)
- Backend API: `GET/POST/PUT/DELETE /templates`
- Templates saved as YAML files in `config/templates/`

**c. Provider Onboarding & Editing**
- Update existing Provider detail page (`/providers/[id]`) to include an "Edit" button for editing provider attributes:
  - Specialty (dropdown — only lists currently supported specialties)
  - Template routing (dropdowns filtered to templates matching the provider's specialty)
  - Custom vocabulary (editable list)
  - Style directives (editable list)
  - Credentials, noise suppression level, practice ID
- Add "New Provider" panel on the Providers list page (`/providers`)
  - Name, credentials, specialty (dropdown of existing specialties)
  - Template mapping: dropdowns filtered to templates available for selected specialty
  - Data directory location (file path for physician-specific data files)
  - Future features noted: LoRA fine-tuning trigger, correction history, quality threshold alerts
- Backend API: `GET/POST/PUT /providers`
- Provider profiles saved as YAML in `config/providers/`

**d. Data Consistency Enforcement**
- Audit all existing specialties, templates, and providers for consistency:
  - Every template must reference a valid specialty that has a dictionary
  - Every provider must map to templates that match their specialty
  - Fix inconsistencies (e.g., "chiropractic" specialty exists in templates but missing from `config/dictionaries/`)
- Create missing dictionary files
- Add validation in the API: reject template/provider creation that violates specialty constraints

### Session 12: On-Demand Note Generation
This session adds the ability to trigger the transcription and note generation pipeline on demand from the UI, with asynchronous notification of completion.

**a. Backend On-Demand Pipeline**
- `POST /encounters` — create encounter + trigger pipeline asynchronously
- `POST /encounters/{id}/upload` — upload audio, triggers pipeline via `asyncio.create_task`
- Pipeline status tracked in encounter state: `pending → processing → complete | error`
- WebSocket notifications (`WS /ws/encounters/{id}`): emits events at each node transition and on completion
- No polling required — client subscribes to WS channel and receives push notifications
- Auto-generates encounter folder in `ai-scribe-data/{mode}/{provider_id}/{patient_name}_{encounter_id}_{date}/` with:
  - Audio file (uploaded or recorded)
  - `patient_demographics.json` — generated from patient selection (name, DOB, sex, MRN)
  - `encounter_details.json` — generated from form inputs (provider_id, visit_type, date_of_service, mode)
- Folder creation + file placement is sufficient to trigger the pipeline — same path as batch processing

**b. Patient Selection via EHR**
- **Patient search dropdown** (searchable, typeahead): provider selects an existing patient before recording
  - Searches by name, MRN, or DOB
  - Returns patient demographics (name, DOB, sex, MRN) and recent encounter history
  - Data source: EHR MCP server (`mcp_servers/ehr/stub_server.py`)
  - Dropdown populates on focus with default patient list (no typing required)
- **Stub EHR patient roster**: `StubEHRServer.search_patients(query)` returns dummy patient data
  - Pre-populated with ~20 dummy patients per practice (realistic names, MRNs, demographics)
  - Stored in `config/ehr_stub/patient_roster.json`
  - **Test patients**: all stub patients are tagged with `_TEST` suffix in last name (e.g., "Dotson_TEST") for clear identification
  - Test patient encounters create folders with `_test` in the name, routed to standard `ai-scribe-data/` and `output/` directories
  - Future: replaced by live EHR integration (FHIR R4, HL7) — same interface, zero pipeline changes
- API endpoint: `GET /patients/search?q={query}&provider_id={id}` — returns matching patients (empty query returns first 10)
- After patient is selected, their demographics auto-populate `patient_demographics.json`

**c. Frontend Capture Page**
- Change "Upload" nav item to "Capture" (renamed)
- Capture page flow:
  1. **Provider selector**: dropdown showing provider name with specialty (e.g., "Dr. Rahman (Orthopedic)")
  2. **Patient selector**: searchable dropdown — typeahead queries `GET /patients/search`; shows name, DOB, MRN
  3. **Visit type selector**: initial, follow_up, assume_care, discharge
  4. **Audio input** (two modes):
     - **Upload**: drag-and-drop zone for MP3/WAV/M4A files
     - **Live dictation**: browser-based audio recording (MediaRecorder API) — record, pause, stop, submit
- After submission:
  - Backend creates encounter folder in `ai-scribe-data/` with audio + `patient_demographics.json` + `encounter_details.json`
  - Pipeline triggers automatically
  - Sample appears in Samples list with status "Processing" (animated indicator)
  - When WS notification arrives with completion, status changes to "Complete"
  - Transcript and note become viewable immediately
- Sample detail page shows pipeline progress during processing (step indicator: Transcribe → Generate → Review)

**d. Golden Source Handling for Live Encounters**
- Live-recorded encounters do **NOT** have a golden source (gold-standard note) at creation time
- Pipeline runs without quality evaluation — transcript and note are generated, but no comparison or quality score is computed
- Quality report is omitted (or marked "N/A — no gold standard available")
- **Future workflow**: after the provider reviews and finalizes the generated note, the finalized note becomes the golden source
  - Finalized note is saved as `final_soap_note.md` in the encounter folder
  - Quality evaluation can then be run retroactively against the finalized note
  - This creates a feedback loop: each finalized note improves the quality baseline
  - See Session 14 (Learning Loop + Correction Capture) for the correction-driven improvement cycle

**e. On-Demand Re-run from Samples Page**
- Any existing sample can be re-run on demand via `POST /encounters/{sample_id}/rerun`
- Auto-detects next version number by scanning existing `generated_note_v*.md` files in the sample's output directory
- Re-uses the original audio file from `ai-scribe-data/`; no re-upload required
- Pipeline runs asynchronously with WebSocket progress events (same as new encounters)
- Output saved as `generated_note_v{N+1}.md` and `audio_transcript_v{N+1}.txt`
- New version appears in the sample detail page version selector after completion
- UI: "Re-run Pipeline" button on each sample detail page header
- Dynamic version discovery: `api/data_loader.py` scans output/ for all version numbers (no hardcoded version list)

**f. Batch Run Support**
- Existing batch pipeline (`scripts/batch_eval.py`) continues to work for bulk processing
- On-demand is for individual encounters triggered from the UI
- Both paths use the same underlying pipeline code

### Session 13: iPhone, iPad and Android Application
This session builds a cross-platform mobile application for AI Scribe using React Native (or Expo), providing core functionality on iPhone, iPad, and Android.

**a. Application Design**
- App name: "AI Scribe" with Talisman Solutions logo
- Cross-platform: React Native / Expo for iPhone, iPad, and Android feature parity
- Connects to the same FastAPI backend as the web app
- Authentication: JWT token-based (same as web API)
- Designed for future extensibility: modular navigation, theme system, offline-first architecture

**b. Core Features**
- **Audio recording**: Record audio directly from the mobile device (microphone capture)
  - Start/stop/pause recording controls
  - Visual audio waveform during recording
  - Provider and visit type selection before recording
  - On recording completion: triggers on-demand note generation pipeline (Session 12 API)
- **Transcript & note viewer**: View latest transcripts and clinical notes filtered by provider
  - List view: recent encounters with provider, date, status, quality score
  - Detail view: rendered clinical note with section headers
  - No version history required (latest version only)
- **Status tracking**: Shows pipeline status (Processing → Complete) with push notifications
- **Provider selector**: Filter encounters by provider

**c. Architecture**
- Shared component library between iOS and Android
- API client: typed SDK matching the web `lib/api.ts` interface
- State management: React Context or Zustand for session state
- Audio: `expo-av` for cross-platform audio recording
- Notifications: push notifications via Firebase Cloud Messaging (FCM) for pipeline completion
- Offline: audio files cached locally (AsyncStorage) and uploaded when connectivity is restored

**d. Future Mobile Features (documented, not implemented)**
- Live streaming transcription (NeMo Streaming ASR via WebSocket)
- Voice commands ("new section", "assessment")
- Offline-first full pipeline execution (on-device models)
- Biometric authentication (Face ID / fingerprint)
- EHR integration via mobile deep links

### Session 14: Production Deployment Architecture & Resource Optimization
This session re-architects the system for production deployment where provider-facing components
run on-prem (or on a separate server) and GPU-intensive pipeline components run on AWS. Both
can initially be deployed on one server but configuration allows splitting across two.

**Key Objective:** Support hybrid deployment (on-prem provider-facing + cloud GPU pipeline) with
strict PHI/EHR data isolation — sensitive data NEVER leaves the provider-facing server.

**a. Provider-Facing Server (CPU, on-prem or cloud)**
Components that interact with providers, EHR systems, and serve the client-facing UI:

- **3A. Client Web Server:** Serves the provider-facing web app — dashboard, physicians,
  specialties, samples, templates (read-only). No template/provider editing (admin-only on
  processing server). Built from the existing Next.js app with a `client` role flag.
- **3B. Mobile API Gateway:** Serves the mobile app. May share the same server as 3A or be
  separate. All mobile app API calls route through this server.
- **3C. Provider API Server:** FastAPI backend that:
  - Exposes APIs to fetch data from local EHR and provider-specific resources
  - Proxies pipeline trigger requests (transcription, note generation) to the processing server
  - Retrieves generated transcripts and SOAP notes from the processing server
  - Periodically syncs templates and provider details from the processing server (every 2 hours)
  - Serves synced template/provider data to the frontend
  - Supports async pipeline progress tracking (WebSocket forwarding)
  - **PHI isolation:** EHR data is accessed locally and NEVER sent to the processing server.
    Only audio files and de-identified encounter metadata are sent for processing.
- **3D. Data Sync & Storage:**
  - Audio files and samples stored locally in the standard folder structure
  - Generated outputs (transcripts, notes) stored in local `output/` folders for web/mobile serving
  - Sync script copies audio + encounter data to the processing server for batch processing
  - Sync script copies generated outputs back from the processing server
  - Supports both on-demand (single encounter) and batch sync modes
  - File overwrite policy: only overwrite if source timestamp > local timestamp

**b. Processing Pipeline Server (GPU, AWS)**
Components that run GPU-intensive AI workloads and admin operations:

- **4A. Pipeline Engine:** Transcription and note generation pipeline (LangGraph). Triggered
  on-demand or batch. Reads from standard input folders (audio, demographics, encounter details,
  gold standard). Produces transcripts and clinical notes.
- **4B. Pipeline API Server:** FastAPI backend that:
  - Accepts file uploads from the provider-facing server into the correct folder structure
  - Triggers pipeline execution (single or batch)
  - Supports async progress tracking and result retrieval
  - Batch upload: handles large volumes; skips files where local timestamp >= uploaded timestamp
  - Batch retrieval: serves generated output files back to the provider-facing server
- **4C. Admin Web Server:** Full-featured web UI with all current capabilities plus:
  - Create/edit/delete specialties, templates, providers
  - Dashboard with quality metrics, pipeline monitoring
  - Used by Talisman to manage the practice and its providers

**c. Configuration & Deployment**
- **`config/deployment.yaml`:** Central deployment config specifying:
  - Server roles (`provider-facing`, `processing-pipeline`, or `both`)
  - Network addresses for each server component
  - Sync schedule (default: every 2 hours for templates/providers)
  - Data directory paths for each server
  - Feature flags per server role (e.g., `allow_template_edit: false` for provider-facing)
- **Deployment scripts:** Automated setup for both server types:
  - `deploy/setup_provider_server.sh` — installs CPU dependencies, configures nginx, systemd
  - `deploy/setup_pipeline_server.sh` — installs GPU dependencies (CUDA, WhisperX, Ollama)
  - `deploy/docker-compose.provider.yml` + `deploy/docker-compose.pipeline.yml`
  - Single-server mode: one docker-compose that runs both roles
  - Split-server mode: separate compose files with network config pointing to each other

**d. Data Sync Infrastructure**
- `scripts/sync_to_pipeline.py` — Push audio + encounter data from provider to pipeline server
- `scripts/sync_from_pipeline.py` — Pull generated outputs from pipeline to provider server
- `scripts/sync_templates.py` — Periodic pull of templates/providers from pipeline server
- All sync operations use the Pipeline API (4B), not direct file access
- Sync supports incremental transfer (timestamp comparison, skip unchanged files)
- Batch sync with progress reporting and error recovery

**e. Comprehensive Testing**
- **Unit tests:** Every component, API endpoint, sync function, config loader
- **Component tests:** Each web page/screen, each API route, each pipeline node
- **Integration tests:** Provider → Pipeline server communication, sync workflows
- **End-to-end tests:** Full workflow from audio upload → pipeline → note retrieval
- Pre-populated test data on both servers (replicated from existing `data/` and `output/`)
- **CI/CD test strategy:** Define test tiers (fast unit → slow E2E), run appropriate tier per change
- Auto-commit on test pass

**f. Implementation Order**
1. `config/deployment.yaml` + config loader
2. Split API into provider-facing + pipeline roles
3. Pipeline API (4B) — file upload, trigger, progress, retrieval
4. Provider API (3C) — proxy to pipeline, EHR local access, template sync
5. Admin vs client web server role split (4C vs 3A)
6. Data sync scripts (3D)
7. Deployment scripts + Docker compose files
8. Comprehensive test suite
9. Documentation

### Session 15: Learning Loop + Correction Capture
- Capture provider corrections (diff: AI output vs provider-edited)
- Classify corrections (ASR_ERROR, STYLE, CONTENT, CODING, TEMPLATE)
- Generate training pairs for post-processor ML model
- Update provider style model from edit patterns
- Feed quality evaluator with real correction data
- Trigger ASR re-fine-tuning when enough new corrected transcripts accumulate (see Session 10f)

### Session 16: S3 Trigger Pipeline (Production Ingestion)
- Implement S3 upload watcher: audio files uploaded to designated S3 bucket trigger pipeline
- EventBridge rule: S3 PutObject → invoke pipeline Lambda/container
- Pipeline reads audio from S3, processes through full graph, writes output `.md` back to S3
- Folder structure in S3 mirrors local data structure:
  ```
  s3://ai-scribe-encounters/
  ├── incoming/                  # Drop audio files here → triggers pipeline
  │   └── {encounter_id}.mp3
  ├── context/                   # Patient context YAML (uploaded with audio or pre-populated)
  │   └── {encounter_id}.yaml
  ├── output/                    # Pipeline writes results here
  │   ├── {encounter_id}/
  │   │   ├── transcript.md
  │   │   ├── clinical_note.md
  │   │   ├── quality_report.md
  │   │   └── metadata.json
  └── archive/                   # Completed encounters moved here
  ```
- Implement `trigger/s3_handler.py` — the entry point invoked by EventBridge
- All output files are Markdown, consistent with local pipeline output format

### Session 17: Evidence-Linked Citations + Audio Recording UI
- Audio indexer: word-level timestamp mapping for citation links
- Citation metadata embedded in note Markdown (link note sentences → audio timestamps)
- Offline audio buffering strategy for client-side capture (IndexedDB ring buffer)

### Session 18: Browser Extension (Super Fill MVP)
- Chrome MV3 extension scaffold
- EHR detection, field mapping, note injection

### Sessions 19+: Advanced features (coding suggestions, patient summaries, voice commands)

## Coding Standards

- **Python**: 3.11+, type hints everywhere, Pydantic for data models
- **Async**: Use async/await for all I/O (API calls, file operations)
- **Error handling**: Every MCP server call wrapped in try/except with fallback logic
- **Configuration**: All engine/model selection via `config/engines.yaml`, never hardcoded
- **Testing**: pytest, test each MCP server independently, test each node independently
- **Logging**: structured logging (structlog), every pipeline step logged with encounter_id
- **Secrets**: environment variables only, never in config files
- **Output format**: ALL generated notes, transcripts, quality reports, and comparisons as Markdown (.md) files with proper headers, sections, and tables

## Important Context

- All audio test data contains PHI — never commit to git, never send to external APIs
- The post-processor in `postprocessor/` is production-tested on 10 real transcripts
- The 98K medical wordlist is GPL-licensed — include attribution
- pyannote requires a HuggingFace token (free) for the diarization model
- Ollama must be installed and running locally before testing LLM nodes
- For GPU nodes (WhisperX, NeMo), CUDA toolkit must be installed
- Gold-standard notes in `data/` are the quality benchmark — every generated note is compared against them
- Generated output versions are tracked: v1 (Session 4), v2 (Session 5), v3 (Session 7), v4 (Session 8), v5 (Session 10), v6 (model upgrade) — each should show measurable improvement
- Current default LLM: qwen2.5:14b via Ollama. Always use --judge-model llama3.1:latest for quality sweeps
- VRAM constraint on A10G (23 GB): use --two-pass flag for batch_eval.py to prevent OOM
- Virtual env: always `source .venv/bin/activate` before running scripts
