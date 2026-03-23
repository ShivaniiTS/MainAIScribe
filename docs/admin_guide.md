# AI Scribe Enterprise — Administrator Guide

This guide covers platform configuration, onboarding, and management tasks for administrators. Configuration is currently file-based (YAML). A web-based admin portal for specialties, templates, and provider management is planned for Session 11.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Onboarding a New Physician](#3-onboarding-a-new-physician)
4. [Managing Templates](#4-managing-templates)
5. [Adding a New Specialty](#5-adding-a-new-specialty)
6. [Medical Dictionary Management](#6-medical-dictionary-management)
7. [Engine Configuration](#7-engine-configuration)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Quality Evaluation](#9-quality-evaluation)
10. [Test Dataset](#10-test-dataset)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Overview

AI Scribe converts doctor-patient conversations and physician dictations into structured clinical notes. The system is fully self-hosted and HIPAA-compliant.

**Two encounter modes:**

| Mode | Input | Description |
|------|-------|-------------|
| **Dictation** | Single audio file | Physician dictates the note after seeing the patient |
| **Conversation (Ambient)** | One or two audio files | Records the doctor-patient conversation; optionally includes a separate physician dictation |

**Processing pipeline:** Audio → Transcription (WhisperX) → Post-processing (12-stage cleanup) → Note Generation (LLM) → Quality Review

**Key configuration files:**

| File | Purpose |
|------|---------|
| `config/engines.yaml` | ASR and LLM engine selection |
| `config/providers/{id}.yaml` | Per-physician profiles |
| `config/templates/{name}.yaml` | Note structure templates |
| `config/dictionaries/*.txt` | Medical terminology dictionaries |
| `config/prompts/*.yaml` | LLM system prompts |
| `config/quality_baseline.yaml` | Quality score thresholds |

---

## 2. Directory Structure

```
config/
├── engines.yaml                  # Engine configuration (ASR, LLM, audio)
├── quality_baseline.yaml         # Quality thresholds for regression testing
├── providers/                    # One YAML file per physician
│   ├── dr_faraz_rahman.yaml
│   ├── dr_caleb_ademiloye.yaml
│   ├── dr_mohammed_alwahaidy.yaml
│   ├── dr_mark_reischer.yaml
│   └── dr_paul_peace.yaml
├── templates/                    # Note structure templates
│   ├── soap_default.yaml         # Generic 4-section fallback
│   ├── ortho_initial_eval.yaml   # Orthopedic initial (12 sections)
│   ├── ortho_follow_up.yaml      # Orthopedic follow-up (6 sections)
│   ├── chiro_initial_eval.yaml   # Chiropractic initial (11 sections)
│   ├── chiro_follow_up.yaml      # Chiropractic follow-up (4 sections)
│   ├── neuro_initial_eval.yaml   # Neurology initial (8 sections)
│   └── neuro_follow_up.yaml      # Neurology follow-up (6 sections)
├── dictionaries/                 # Medical terminology
│   ├── base_medical.txt          # 98K OpenMedSpel terms
│   ├── orthopedic.txt
│   ├── chiropractic.txt
│   ├── neurology.txt
│   └── custom/                   # Provider-specific terms
│       └── {provider_id}.txt
└── prompts/                      # LLM prompt templates
    ├── note_generation.yaml
    ├── coding_suggestion.yaml
    └── patient_summary.yaml
```

---

## 3. Onboarding a New Physician

### 3.1 Create the Provider Profile

Create a new file at `config/providers/{provider_id}.yaml`. The `provider_id` must be a lowercase, underscore-separated identifier (e.g., `dr_jane_smith`).

**Minimal profile:**

```yaml
id: dr_jane_smith
name: Dr. Jane Smith
credentials: MD
specialty: orthopedic
practice_id: excelsia_injury_care
note_format: SOAP
noise_suppression_level: moderate
postprocessor_mode: hybrid

custom_vocabulary:
- TermSpecificToThisDoctor
- ClinicName

style_directives:
- Write in third person
- Use past tense for examination findings
- List diagnoses as numbered items in the Assessment
- Mirror Assessment numbering in the Plan

template_routing:
  default: ortho_follow_up
  follow_up: ortho_follow_up
  initial: ortho_initial_eval
  initial_evaluation: ortho_initial_eval

# Auto-populated fields (do not edit manually):
asr_override: null
llm_override: null
correction_count: 0
quality_history: []
quality_scores: {}
style_model_version: v0
npi: ''
```

### 3.2 Profile Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier. Must match filename. |
| `name` | string | Display name (e.g., "Dr. Jane Smith") |
| `credentials` | string | MD, DO, DC, NP, PA, etc. |
| `specialty` | string | Must match available templates (orthopedic, chiropractic, neurology, etc.) |
| `practice_id` | string | Practice/clinic identifier |
| `note_format` | string | Note format preference (SOAP is default) |
| `noise_suppression_level` | string | off, mild, moderate, aggressive |
| `postprocessor_mode` | string | hybrid, rules_only, ml_only |
| `custom_vocabulary` | list | Provider-specific medical terms for ASR boosting and LLM context |
| `style_directives` | list | Instructions injected into the LLM prompt to control note style |
| `template_routing` | map | Maps visit_type → template_id |
| `asr_override` | string/null | Override default ASR engine for this provider |
| `llm_override` | string/null | Override default LLM model for this provider |
| `quality_history` | list | Auto-populated by quality sweeps |
| `quality_scores` | map | Auto-populated: version → score |

### 3.3 Template Routing

The `template_routing` field maps encounter visit types to template filenames. Common visit types:

| Visit Type Key | When Used |
|----------------|-----------|
| `initial` | First visit / new patient evaluation |
| `initial_evaluation` | Alias for initial |
| `follow_up` | Return visit |
| `assume_care` | Patient transferring from another provider |
| `discharge` | Final visit / discharge evaluation |
| `default` | Fallback when visit type is unknown |

The value is the template filename without `.yaml` (e.g., `ortho_follow_up` maps to `config/templates/ortho_follow_up.yaml`).

### 3.4 Custom Vocabulary

Custom vocabulary terms serve three purposes:
1. **ASR hotword boosting** — The top 100 terms are passed to WhisperX as logit-boosted hotwords, improving recognition of uncommon medical terms
2. **LLM prompt context** — Terms are injected into the note generation prompt so the LLM uses correct terminology
3. **Post-processor dictionary** — Terms are added to the medical spell-check dictionary

**Best practices:**
- Include the practice/clinic name
- Add procedure names the physician commonly performs
- Add medication names they frequently prescribe
- Add anatomical terms specific to their specialty
- Add eponymous tests (Spurling, Tinel, Phalen, etc.)

### 3.5 Style Directives

Style directives are natural-language instructions injected into the LLM system prompt. They control how the generated note reads.

**Common directives:**

```yaml
style_directives:
- Write in third person (e.g. 'The patient presents...' not 'Patient presents')
- Use past tense for examination findings and interval history
- Spell out abbreviations on first use (e.g. 'range of motion (ROM)')
- Include ROM measurements in degrees where documented
- List diagnoses in the Assessment as numbered items
- Mirror Assessment numbering in the Plan
- Use formal clinical language; avoid contractions
- Document work status in the Plan when mentioned
- For MVA cases, reference the accident date and mechanism in the HPI
```

**Tips:**
- Be specific — "Include ROM in degrees" is better than "Be detailed"
- Include examples in parentheses where helpful
- Order from most important to least important
- Limit to 10-15 directives to avoid prompt bloat

---

## 4. Managing Templates

### 4.1 Template Structure

Each template is a YAML file in `config/templates/` that defines:
- **Sections** — What sections the note should contain, in what order
- **Required/optional** — Whether each section must be populated
- **Prompt hints** — Guidance for the LLM on what to include in each section
- **Formatting rules** — Voice, tense, abbreviation handling

### 4.2 Creating a New Template

```yaml
# config/templates/cardio_follow_up.yaml
name: "Cardiology Follow-up"
specialty: cardiology
visit_type: follow_up

header_fields:
  - patient_name
  - date_of_birth
  - date_of_service
  - provider_name
  - location

sections:
  - id: history_of_present_illness
    label: "HISTORY OF PRESENT ILLNESS"
    required: true
    prompt_hint: "Interval changes since last visit: symptoms, medication response, functional status"

  - id: cardiovascular_review
    label: "CARDIOVASCULAR REVIEW"
    required: true
    prompt_hint: "Chest pain, dyspnea, palpitations, edema, syncope"

  - id: physical_examination
    label: "PHYSICAL EXAMINATION"
    required: true
    prompt_hint: "Vitals, heart auscultation, lung exam, peripheral pulses, edema assessment"

  - id: diagnostics
    label: "DIAGNOSTICS"
    required: false
    prompt_hint: "ECG, echocardiogram, stress test, lab results if reviewed"

  - id: assessment
    label: "ASSESSMENT"
    required: true
    prompt_hint: "Numbered diagnoses with current status"

  - id: plan
    label: "PLAN"
    required: true
    prompt_hint: "Numbered plan: medication changes, follow-up studies, referrals, lifestyle modifications"

formatting:
  voice: active
  tense: past
  person: third
  abbreviations: spell_out
  measurements: include_units
```

### 4.3 Section Design Guidelines

- **Follow-up templates** typically have 4-6 sections (HPI, PE, Assessment, Plan + optional sections)
- **Initial evaluation templates** typically have 8-12 sections (full history, ROS, exam, imaging, assessment, plan)
- **Section IDs** must be `snake_case` and unique within the template
- **Labels** are displayed as-is in the generated note (use ALL CAPS for clinical convention)
- **Prompt hints** are critical — they tell the LLM exactly what clinical details to extract and include

### 4.4 Template Naming Convention

`{specialty}_{visit_type}.yaml`

Examples:
- `ortho_initial_eval.yaml`
- `chiro_follow_up.yaml`
- `neuro_initial_eval.yaml`
- `cardio_follow_up.yaml`
- `psych_initial_eval.yaml`

### 4.5 Template Selection Chain

When an encounter is processed, the template is selected via this chain:

```
encounter_details.json (visit_type) + provider profile (template_routing)
    ↓
ProviderManager.resolve_template(provider_id, visit_type)
    ↓
Lookup: template_routing[visit_type] → template_id
    ↓
Fallback: template_routing["default"] → template_id
    ↓
Fallback: "soap_default"
    ↓
TemplateServer loads: config/templates/{template_id}.yaml
```

---

## 5. Adding a New Specialty

Adding a specialty requires three components: templates, a dictionary, and provider profiles.

### 5.1 Step-by-Step

1. **Create templates** — At minimum, create `{specialty}_initial_eval.yaml` and `{specialty}_follow_up.yaml` in `config/templates/`. Analyze gold-standard notes from physicians of this specialty to determine the correct section structure.

2. **Create a specialty dictionary** — Create `config/dictionaries/{specialty}.txt` with one term per line. Include anatomy, procedures, diagnoses, medications, instruments, and scales specific to this specialty.

3. **Create provider profiles** — For each physician in this specialty, create a profile YAML with `specialty: {specialty_name}` and `template_routing` entries pointing to the new templates.

### 5.2 Specialty Dictionary Sources

- **Gold notes** — Extract medical terms from existing gold-standard notes
- **LLM generation** — Prompt the LLM: "List 200 medical terms specific to {specialty}"
- **Public databases** — UMLS Metathesaurus, SNOMED CT, ICD-10 terminology
- **Physician input** — Ask the physician for terms they commonly use

### 5.3 Currently Supported Specialties

| Specialty | Templates | Dictionary | Providers |
|-----------|-----------|------------|-----------|
| Orthopedic | ortho_initial_eval, ortho_follow_up | orthopedic.txt | Dr. Rahman, Dr. Ademiloye |
| Chiropractic | chiro_initial_eval, chiro_follow_up | chiropractic.txt | Dr. Alwahaidy, Dr. Reischer |
| Neurology | neuro_initial_eval, neuro_follow_up | neurology.txt | Dr. Peace |

---

## 6. Medical Dictionary Management

### 6.1 Dictionary Tiers

```
Tier 1: config/dictionaries/base_medical.txt     (98K terms, shared by all)
Tier 2: config/dictionaries/{specialty}.txt       (specialty-specific terms)
Tier 3: config/dictionaries/custom/{provider}.txt (provider-specific terms)
Tier 4: Provider YAML → custom_vocabulary field   (also provider-specific)
```

All tiers are merged and deduplicated at runtime. Changes take effect on the next pipeline run — no restart required.

### 6.2 Adding Terms

**To a specialty dictionary:** Edit `config/dictionaries/{specialty}.txt` — one term per line, case-sensitive.

**To a provider's vocabulary:** Either:
- Edit `config/dictionaries/custom/{provider_id}.txt` (one term per line), or
- Add to the `custom_vocabulary` list in `config/providers/{provider_id}.yaml`

Both methods work. The YAML `custom_vocabulary` list is preferred because it also feeds ASR hotword boosting and LLM prompt injection.

### 6.3 Dictionary Format

Plain text, one term per line:
```
foraminotomy
discectomy
laminectomy
spondylolisthesis
radiculopathy
```

No headers, no comments, no blank lines between terms.

---

## 7. Engine Configuration

All engine selection is controlled by `config/engines.yaml`. Changing engines is a config change, never a code change.

### 7.1 LLM Configuration

```yaml
llm:
  default_server: ollama
  servers:
    ollama:
      url: "http://localhost:11434/v1"
      models:
        note_generation: "qwen2.5:14b"    # Primary note generation model
        coding: "qwen2.5:14b"
        patient_summary: "qwen2.5:14b"
        command_parse: "qwen2.5:7b"
      keep_alive: "0"                       # Unload model after each request (VRAM management)
```

**Changing the LLM model:**
1. Pull the new model: `ollama pull model_name`
2. Update `config/engines.yaml` → `llm.servers.ollama.models.note_generation`
3. Run a quality sweep to verify the new model meets quality thresholds

### 7.2 ASR Configuration

```yaml
asr:
  default_server: whisperx
  servers:
    whisperx:
      model: "large-v3"
      device: "cuda"
      compute_type: "float16"
      diarization: true
      batch_size: 16
      beam_size: 5
```

### 7.3 Per-Provider Engine Overrides

Individual providers can override the default engine via their profile YAML:

```yaml
# In config/providers/dr_jane_smith.yaml
asr_override: whisperx_lora    # Use LoRA-fine-tuned ASR for this provider
llm_override: qwen2.5:32b      # Use larger LLM for this provider
```

---

## 8. Running the Pipeline

### 8.1 Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure Ollama is running with required models
ollama list    # Should show qwen2.5:14b, llama3.1:latest

# Set environment variables
set -a && source .env && set +a
```

### 8.2 Batch Processing

Process all samples through the pipeline:

```bash
# Two-pass mode (required for A10G 23GB VRAM constraint)
python scripts/batch_eval.py --two-pass --version v8 --data-dir ai-scribe-data

# Filter by mode
python scripts/batch_eval.py --two-pass --version v8 --mode dictation
python scripts/batch_eval.py --two-pass --version v8 --mode ambient
```

**Two-pass mode explained:**
- Pass 1: ASR only — transcribes all audio, saves to `output/transcript_cache_{version}.json`
- Pass 2: LLM only — generates notes from cached transcripts (WhisperX freed from VRAM)
- Required because WhisperX (10-12 GB) + qwen2.5:14b (8 GB) exceed 23 GB VRAM

### 8.3 Single Encounter Processing

```bash
# Via API
curl -X POST http://localhost:8000/encounters \
  -H "Content-Type: application/json" \
  -d '{"provider_id": "dr_faraz_rahman", "visit_type": "follow_up", "mode": "dictation"}'

# Upload audio
curl -X POST http://localhost:8000/encounters/{id}/upload \
  -F "audio=@path/to/audio.mp3"
```

---

## 9. Quality Evaluation

### 9.1 Running a Quality Sweep

```bash
python scripts/run_quality_sweep.py \
  --version v8 \
  --judge-model llama3.1:latest \
  --data-dir ai-scribe-data
```

**Important:** Always use `--judge-model llama3.1:latest` for consistent cross-version comparison. The judge model must be different from the generation model to avoid self-evaluation bias.

### 9.2 Quality Dimensions

Each note is scored on 6 dimensions (1-5 scale):

| Dimension | What It Measures |
|-----------|-----------------|
| **Medical Accuracy** | Correct medical facts, no errors in diagnoses/medications/findings |
| **Completeness** | All relevant information from the transcript is captured |
| **No Hallucination** | Nothing is fabricated or added beyond what's in the source material |
| **Structure Compliance** | Note follows the template structure correctly |
| **Clinical Language** | Appropriate medical terminology and professional writing |
| **Readability** | Clear, well-organized, easy to review |

**Overall score** = weighted average of all dimensions.

### 9.3 Quality History

Quality scores are tracked per provider, per version:

```yaml
# In config/providers/dr_faraz_rahman.yaml (auto-populated)
quality_history:
- date: '2026-03-13'
  samples: 22
  score: 4.443
  version: v6
quality_scores:
  v6: 4.443
  v7: 4.296
```

### 9.4 Quality Baseline

`config/quality_baseline.yaml` defines minimum acceptable scores. The regression test suite (`tests/quality/test_regression.py`) ensures new pipeline versions don't degrade below these thresholds.

### 9.5 Golden Source (Gold Standard Notes)

Quality evaluation requires a **golden source** — a clinician-verified reference note to compare against.

**Batch-processed encounters** (historical data in `ai-scribe-data/`):
- Gold standard note is `final_soap_note.md` in each encounter folder
- These were provided with the original dataset and are the authoritative reference

**Live-recorded encounters** (created via the Capture page or mobile app):
- Do **NOT** have a golden source at creation time
- The pipeline generates a transcript and note, but **no quality score is computed**
- Quality report is omitted or marked "N/A — no gold standard available"
- **Future workflow (Session 14):** after the provider reviews and finalizes the AI-generated note, the finalized version is saved as `final_soap_note.md` in the encounter folder
  - Quality evaluation can then be run retroactively
  - Each finalized note strengthens the quality baseline for that provider
  - Provider corrections (diff between AI output and finalized note) feed the learning loop for continuous improvement

### 9.6 On-Demand Pipeline Re-run

Any sample visible in the Samples page can be re-run on demand:

1. Navigate to the sample detail page
2. Click **"Re-run Pipeline"** in the top-right corner
3. The pipeline re-processes the original audio file and generates a new version
4. Version number is automatically incremented (e.g., v7 → v8)
5. Progress is shown in real-time via WebSocket events
6. The new version appears in the version selector once complete

API: `POST /encounters/{sample_id}/rerun` — returns `{ encounter_id, sample_id, version, status }`.

### 9.7 Test Patient Data

For development and testing, the stub EHR roster provides ~20 dummy patients:

- All test patients have `_TEST` suffix in their last name (e.g., "James Dotson_TEST")
- Test encounters create folders with `_test` in the name for easy identification
- The patient dropdown on the Capture page shows all test patients by default (no typing required)
- Test encounters are fully processed by the pipeline and appear alongside real data in the Samples list
- Roster file: `config/ehr_stub/patient_roster.json`

---

## 10. Test Dataset

### 10.1 Download

Audio files for testing are available on SharePoint:

**[Download Test Dataset](https://talismansolutionscom.sharepoint.com/:f:/s/ExcelsiaITprojects-AIScribe/IgBCWV3umPwaRowR3nC8dGVGAV5_VlQBAOBqJuDOewniM58?e=s9fYUE)**

The repository includes encounter metadata (patient demographics, gold-standard notes, encounter details) but **excludes audio files** (`.mp3`) due to size and PHI concerns. After downloading, place the audio files into their corresponding `ai-scribe-data/` encounter folders — the folder names in the SharePoint archive match the repo structure.

### 10.2 Expected Folder Structure

```
ai-scribe-data/
├── conversation/                           # Ambient (multi-speaker) encounters
│   └── <physician_id>/
│       └── <patient>_<mrn>_<date>/
│           ├── conversation_audio.mp3      # Doctor-patient conversation
│           ├── note_audio.mp3              # Optional: physician dictation after visit
│           ├── final_soap_note.md          # Gold-standard clinical note
│           ├── patient_demographics.json   # Patient name, DOB, sex, MRN
│           └── encounter_details.json      # Visit type, provider, date, mode
│
├── dictation/                              # Single-speaker physician dictations
│   └── <physician_id>/
│       └── <patient>_<mrn>_<date>/
│           ├── dictation.mp3               # Physician dictation audio
│           ├── final_soap_note.md          # Gold-standard clinical note
│           ├── patient_demographics.json
│           └── encounter_details.json
```

### 10.3 Configurable Data Paths

All data directory paths can be overridden via environment variables (defined in `config/paths.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_SCRIBE_ROOT` | Auto-detected project root | Base project root |
| `AI_SCRIBE_DATA_DIR` | `{root}/ai-scribe-data` | Patient encounter data |
| `AI_SCRIBE_OUTPUT_DIR` | `{root}/output` | Pipeline output |
| `AI_SCRIBE_CONFIG_DIR` | `{root}/config` | Config files |

Set these in your `.env` file or export before running:

```bash
export AI_SCRIBE_DATA_DIR=/mnt/efs/ai-scribe-data
export AI_SCRIBE_OUTPUT_DIR=/mnt/efs/output
```

---

## 11. Troubleshooting

### Common Issues

**Pipeline produces empty or generic notes:**
- Check that the provider profile exists and has correct `template_routing`
- Verify the template file exists in `config/templates/`
- Check that `encounter_details.json` has a valid `visit_type`

**ASR produces garbled output:**
- Verify audio quality (noise suppression level may need adjustment)
- Check VRAM availability: `nvidia-smi`
- Try increasing `beam_size` in `config/engines.yaml`

**Template not found:**
- The template filename in `template_routing` must exactly match a file in `config/templates/` (without `.yaml`)
- Fallback chain: provider routing → default routing → `soap_default`

**Quality scores dropped after model change:**
- Always compare using the same judge model (`--judge-model llama3.1:latest`)
- Run on the full sample set, not a subset
- Check `quality/experiments.jsonl` for the full experiment log

**VRAM out of memory:**
- Use `--two-pass` flag for batch processing
- Set `keep_alive: "0"` in engines.yaml to unload models after each request
- Monitor with `nvidia-smi -l 1`

**Provider not showing in UI:**
- Ensure the provider YAML exists in `config/providers/`
- Ensure sample data exists in `ai-scribe-data/{mode}/{provider_id}/`
- The API discovers providers from both YAML files and data directories
