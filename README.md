# AI Scribe Enterprise

Fully self-hosted, HIPAA-compliant AI medical scribe. Converts doctor-patient
conversations (ambient mode) and physician dictations (dictation mode) into
structured clinical notes — with zero cloud dependencies and zero PHI egress.

**Current quality:** 4.35 / 5.0 across 61 encounters, 5 providers, 3 specialties
(LLM-as-judge evaluation against hand-authored notes).

---

## Dual-Server Architecture

AI Scribe runs as two server roles from a single codebase:

```
┌──────────────────────────────────────────────────────────┐
│                      CLIENTS                             │
│  Web App (:3000)  ·  Mobile App (Expo/RN)                │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│       PROVIDER-FACING SERVER  (port 8000)                │
│  FastAPI + Next.js — encounter browsing, EHR access,     │
│  audio capture, proxy to pipeline. Holds PHI locally.    │
└────────────┬─────────────────────────────────────────────┘
             │  audio + de-identified metadata (NO PHI)
             ▼
┌──────────────────────────────────────────────────────────┐
│    PROCESSING-PIPELINE SERVER  (port 8100)               │
│  FastAPI + Next.js — GPU pipeline (ASR + LLM), admin     │
│  CRUD for providers/templates/specialties.               │
└──────────────────────────────────────────────────────────┘
```

**Development:** Run two instances on the same machine — one provider-facing on port 8000, one pipeline on port 8100.

| | Provider-Facing | Pipeline |
|---|---|---|
| **Port** | API 8000, Web 3000 | API 8100, Admin 3100 |
| **GPU** | No | Yes (WhisperX, Ollama) |
| **PHI** | Stored locally | Never receives |
| **Admin CRUD** | Read-only | Full |
| **Mobile app** | Connects here | N/A |

> See [docs/dual_server_guide.md](docs/dual_server_guide.md) for architecture details
> and [docs/team_responsibilities.md](docs/team_responsibilities.md) for team ownership.

---

## Encounter Pipeline

```
AUDIO FILE
    │
    ▼
┌─────────┐   ┌─────────┐   ┌───────────┐   ┌──────┐   ┌────────┐   ┌──────────┐
│ CONTEXT │──►│ CAPTURE │──►│ TRANSCRIBE│──►│ NOTE │──►│ REVIEW │──►│ DELIVERY │
│  NODE   │   │  NODE   │   │   NODE    │   │ NODE │   │  NODE  │   │  NODE    │
└─────────┘   └─────────┘   └───────────┘   └──────┘   └────────┘   └──────────┘
     │                            │                │
  EHR/YAML                  WhisperX +           Ollama +
  patient                   pyannote +           templates +
  context                   12-stage             specialty
                            post-processor       vocab +
                                                 style
```

Every AI component is pluggable — ASR, LLM, EHR adapters, audio processing —
all behind MCP tool server interfaces. Swap engines in `config/engines.yaml`;
no code changes required.

---

## Technology Stack (All Self-Hosted)

| Layer | Default | Alternatives |
|-------|---------|-------------|
| ASR (batch) | WhisperX (faster-whisper + pyannote 3.1 + wav2vec2) | NVIDIA Parakeet |
| ASR (streaming) | NVIDIA Nemotron-Speech-Streaming | WhisperX chunked |
| Diarization | pyannote 3.1 | NVIDIA Multitalker-Parakeet |
| Noise suppression | DeepFilterNet | RNNoise, passthrough |
| LLM inference | Ollama (OpenAI-compatible API) | vLLM, SGLang |
| LLM model | Qwen 2.5-14B (Apache 2.0) | Qwen 2.5-32B, Llama 3.1-70B |
| Post-processing | 12-stage rule-based + 98K medical dict | ByT5 ML model (future) |
| Orchestration | LangGraph | — |
| Web app | Next.js + React | — |
| Mobile app | React Native / Expo | — |
| EHR integration | Stub (YAML) | FHIR R4, HL7v2, Browser extension |

**License cost: $0/month.** Every component is open-source with permissive licenses.

---

## Providers & Specialties

5 onboarded physicians across 3 specialties:

| Provider | Specialty | Samples |
|----------|-----------|---------|
| Dr. Faraz Rahman (MD) | Orthopedic | 22 (dictation + ambient) |
| Dr. Caleb Ademiloye (DO) | Orthopedic | 9 dictation |
| Dr. Mohammed Alwahaidy (DC) | Chiropractic | 23 (ambient + dictation) |
| Dr. Mark Reischer (DC) | Chiropractic | 5 dictation |
| Dr. Paul Peace (MD) | Neurology | 4 dictation |

7 note templates: `soap_default`, `ortho_initial_eval`, `ortho_follow_up`,
`chiro_initial_eval`, `chiro_follow_up`, `neuro_initial_eval`, `neuro_follow_up`

---

## Test Dataset

Audio files for testing are available on SharePoint:

**[Download Test Dataset](https://talismansolutionscom.sharepoint.com/:f:/s/ExcelsiaITprojects-AIScribe/IgBCWV3umPwaRowR3nC8dGVGAV5_VlQBAOBqJuDOewniM58?e=s9fYUE)**

The repository includes encounter metadata (patient demographics, gold-standard notes,
encounter details) but **excludes audio files** (`.mp3`) due to size and PHI concerns.
After downloading, place the audio files into their corresponding `ai-scribe-data/`
encounter folders — the folder names in the SharePoint archive match the repo structure.

---

## Quick Start

```bash
# 1. Install Ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:14b
ollama pull llama3.1:latest   # used as quality judge

# 2. Install Python dependencies
pip install -e ".[dev]"

# 3. Set environment variables
cp .env.example .env
# Set HF_TOKEN (required for pyannote diarization)
source .env

# 4. Start the provider-facing API
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --reload --port 8000

# 5. Start the processing-pipeline API (separate terminal)
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --reload --port 8100

# 6. Start the web UI
cd client/web && npm install && npm run dev   # port 3000

# 7. Start the mobile app
cd client/mobile && npm install && npx expo start
```

### Split-Server Mode

```bash
# Provider-facing server (CPU, on-prem)
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --port 8000

# Pipeline server (GPU, cloud)
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --port 8100
```

---

## Batch Evaluation

```bash
# Run pipeline on all samples (two-pass for VRAM safety)
python scripts/batch_eval.py --two-pass --version v9

# Score with LLM-as-judge (6 dimensions)
python scripts/run_quality_sweep.py --version v9 --judge-model llama3.1:latest
```

---

## Quality Progress

| Version | What was added | Score | Samples |
|---------|---------------|-------|---------|
| v1 | Basic end-to-end pipeline | ~3.5 | 22 |
| v2 | Specialty templates + medical vocabulary | 4.30 | 22 |
| v3 | EHR context + patient demographics | 4.34 | 22 |
| v4 | Provider profiles + style directives | 4.38 | 22 |
| v5 | ASR inference knobs | 4.35 | 22 |
| v6 | Model upgrade (qwen2.5:14b) | **4.44** | 22 |
| v7 | Dual-audio conversation processing | 4.31 | 61 |
| v8 | Multi-provider templates (3 specialties) | **4.35** | 61 |

Judged by `llama3.1:latest` across 6 dimensions: medical accuracy, completeness,
hallucination-free, structure compliance, clinical language, readability.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System architecture — design principles, deployment, pipeline, MCP servers, routing |
| [docs/dual_server_guide.md](docs/dual_server_guide.md) | Dual-server architecture — per-server responsibilities, APIs, data flow |
| [docs/team_responsibilities.md](docs/team_responsibilities.md) | Team ownership — what each team builds, owns, and deploys |
| [docs/implementation.md](docs/implementation.md) | Implementation reference — pipeline internals, prompt assembly, quality framework |
| [CLAUDE.md](CLAUDE.md) | Build sequence and session-by-session instructions |

---

## Infrastructure Sizing

| Tier | Cost | GPU | Capacity |
|------|------|-----|----------|
| Proof of concept | ~$260/mo | 1x T4 16GB | 20–30 notes/day |
| **Production start** | **~$630/mo** | **T4 + A10 24GB** | **100+ notes/day** |
| Quality maximum | ~$1,250/mo | T4 + A100 40GB | 500+ notes/day |

---

## License

Proprietary. All third-party components are open-source:
- WhisperX, faster-whisper, pyannote: MIT / CC-BY-4.0
- OpenMedSpel (98K medical wordlist): GPL 3.0
- LangGraph, LangChain: MIT
- Qwen 2.5: Apache 2.0
- Llama 3.1: Llama Community License
- Ollama, vLLM: MIT / Apache 2.0
