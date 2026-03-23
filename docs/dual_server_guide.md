# AI Scribe Enterprise — Dual-Server Architecture Guide

This document describes how AI Scribe Enterprise is split across two server roles. Each section tells one team exactly what their server owns, what APIs it exposes, how it communicates with the other server, and where the code lives.

> **Development:** Run two instances on the same machine — one provider-facing on port 8000, one pipeline on port 8100. Use the `AI_SCRIBE_SERVER_ROLE` env var for each process.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLIENTS                           │
│  Web App (Next.js :3000)  ·  Mobile App (Expo)      │
└──────────────┬──────────────────────────────────────┘
               │ HTTPS
               ▼
┌─────────────────────────────────────────────────────┐
│         PROVIDER-FACING SERVER (port 8000)           │
│                                                     │
│  FastAPI — client UI backend, EHR access, proxy      │
│  Next.js — read-only web UI (:3000)                  │
│                                                     │
│  PHI stays here. Audio + de-identified metadata      │
│  are forwarded to the pipeline server.               │
└──────────────┬──────────────────────────────────────┘
               │ HTTP (internal network)
               ▼
┌─────────────────────────────────────────────────────┐
│       PROCESSING-PIPELINE SERVER (port 8100)         │
│                                                     │
│  FastAPI — pipeline execution, admin CRUD            │
│  Next.js — admin web UI (:3100)                      │
│  GPU workloads: WhisperX ASR, Ollama LLM             │
│                                                     │
│  Authoritative source for templates, providers,      │
│  specialties, dictionaries. No PHI.                  │
└─────────────────────────────────────────────────────┘
```

---

## 2. Provider-Facing Server

**Owner:** Provider-facing team
**Port:** API 8000, Web UI 3000
**Role value:** `provider-facing`
**Start command:**
```bash
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --port 8000
```

### 2.1 Responsibilities

| Responsibility | Description |
|---|---|
| Serve client web app | Next.js UI for providers to view encounters, notes, transcripts |
| Serve mobile app API | Same FastAPI endpoints consumed by the React Native mobile app |
| EHR access (local) | Patient search, demographics — reads from `config/ehr_stub/patient_roster.json` (stub) or live EHR in production |
| Audio capture | Accept audio uploads from web/mobile, store locally |
| Proxy to pipeline | Forward audio + de-identified metadata to pipeline server for processing |
| Config sync | Background task pulls templates/providers/specialties from pipeline server every 2 hours |
| Output sync | Pull generated notes and transcripts back from pipeline server |
| PHI isolation | Patient demographics, gold notes, and finalized notes NEVER leave this server |

### 2.2 API Routes

| Method | Path | Description |
|---|---|---|
| `GET` | `/encounters` | List all encounters/samples |
| `GET` | `/encounters/{id}` | Encounter detail |
| `GET` | `/encounters/{id}/note` | Generated clinical note (Markdown) |
| `GET` | `/encounters/{id}/transcript` | Cleaned transcript |
| `GET` | `/encounters/{id}/audio` | Audio file stream |
| `GET` | `/encounters/{id}/gold` | Gold-standard note |
| `GET` | `/encounters/{id}/quality` | Quality scores |
| `GET` | `/encounters/{id}/comparison` | Gold vs generated comparison |
| `POST` | `/encounters` | Create encounter → proxied to pipeline |
| `POST` | `/encounters/{id}/upload` | Upload audio → proxied to pipeline |
| `GET` | `/encounters/{id}/status` | Poll pipeline status → proxied |
| `GET` | `/providers` | List provider profiles |
| `GET` | `/providers/{id}` | Provider detail + quality history |
| `GET` | `/patients/search?q=` | Search patient roster (EHR stub) |
| `GET` | `/specialties` | List specialties (read-only, synced) |
| `GET` | `/templates` | List templates (read-only, synced) |
| `GET` | `/quality/aggregate` | Aggregate quality stats |
| `GET` | `/quality/samples` | Per-sample quality scores |
| `WS` | `/ws/encounters/{id}` | Real-time pipeline progress events |
| `GET` | `/health` | Health check |
| `GET` | `/config/features` | Active feature flags |
| `GET` | `/config/role` | Server role info |

### 2.3 Feature Flags (what's enabled)

```yaml
# Enabled on provider-facing
dashboard: true
view_encounters: true
view_providers: true
view_specialties: true        # read-only
view_templates: true          # read-only
view_quality: true
record_audio: true
trigger_pipeline: true        # proxied to pipeline server
ehr_access: true
patient_search: true

# DISABLED on provider-facing
create_providers: false
edit_providers: false
create_templates: false
edit_templates: false
create_specialties: false
edit_specialties: false
run_pipeline: false           # pipeline runs on GPU server
batch_processing: false
```

### 2.4 Key Code Files

| File | Purpose |
|---|---|
| `api/main.py` | FastAPI app — mounts routes based on `cfg.is_provider_facing` |
| `api/routes/encounters.py` | Encounter CRUD — reads from local `output/`, proxies pipeline ops |
| `api/routes/providers.py` | Provider list/detail — reads from `config/providers/*.yaml` |
| `api/routes/patients.py` | Patient search — reads from `config/ehr_stub/patient_roster.json` |
| `api/routes/specialties.py` | Specialties list (read-only in this role) |
| `api/routes/templates.py` | Templates list (read-only in this role) |
| `api/routes/quality.py` | Quality aggregate/per-sample scores |
| `api/proxy.py` | HTTP client that forwards pipeline requests to the pipeline server |
| `api/sync.py` | Background config sync — pulls providers/templates/specialties every 2h |
| `api/data_loader.py` | Reads `output/` and `data/` directories for encounter data |
| `api/ws/session_events.py` | WebSocket for real-time pipeline progress |
| `config/deployment.py` | `DeploymentConfig`, `ServerRole`, `FeatureFlags`, `require_feature()` |
| `config/deployment.yaml` | Central config — network, data paths, sync schedule, features |
| `config/paths.py` | Resolves `DATA_DIR`, `OUTPUT_DIR`, `CONFIG_DIR` per server role |

### 2.5 Data Directories

```
ai-scribe-data/                    # Patient encounter data (audio, demographics)
├── dictation/
│   └── {provider_id}/
│       └── {sample_id}/
│           ├── dictation.mp3
│           ├── patient_demographics.json    ← PHI, stays local
│           ├── encounter_details.json       ← de-identified, sent to pipeline
│           └── patient_context.yaml         ← PHI, stays local
└── conversation/
    └── (same structure)

output/                            # Generated notes + transcripts (synced FROM pipeline)
├── dictation/
│   └── {provider_id}/
│       └── {sample_id}/
│           ├── generated_note_v8.md
│           ├── audio_transcript_v8.txt
│           └── quality_report.md
└── conversation/
    └── (same structure)

config/                            # Configuration (synced FROM pipeline)
├── providers/*.yaml               # Provider profiles
├── templates/*.yaml               # Note templates
├── dictionaries/*.txt             # Specialty keyword dictionaries
└── ehr_stub/patient_roster.json   # Stub patient roster (local EHR)
```

### 2.6 Proxy Pattern

When a provider-facing server receives a pipeline operation (create encounter, upload audio, poll status), it proxies the request to the pipeline server:

```python
# api/proxy.py — simplified flow
async def proxy_upload(audio_bytes, sample_id, mode, provider_id, encounter_details):
    client = httpx.AsyncClient(base_url=cfg.pipeline_api_url)  # http://pipeline:8100
    resp = await client.post("/pipeline/upload", files={"audio": ...}, data={...})
    return resp.json()
```

The proxy:
- Sends audio + `encounter_details.json` (de-identified)
- Does NOT send `patient_demographics.json` or `patient_context.yaml` (PHI)
- Uses shared-secret auth when `inter_server_auth.enabled: true`
- Reuses a persistent `httpx.AsyncClient` for connection pooling

### 2.7 Config Sync

The `api/sync.py` module runs a background `asyncio` task that periodically pulls from the pipeline server:

```
Pipeline Server ──GET /providers──→ Provider-facing writes config/providers/*.yaml
Pipeline Server ──GET /templates──→ Provider-facing writes config/templates/*.yaml
Pipeline Server ──GET /specialties─→ Provider-facing writes config/dictionaries/*.txt
```

- **Interval:** Every 2 hours (configurable via `sync.config_sync.interval_seconds`)
- **Policy:** Only creates missing files — does not overwrite existing local configs
- **Startup/shutdown:** Started in FastAPI lifespan, cancelled on shutdown

### 2.8 Web UI (Provider-Facing)

The Next.js web app at `client/web/` serves a **read-only dashboard** when in provider-facing mode:

| Page | Path | Capabilities |
|---|---|---|
| Dashboard | `/` | KPI cards, quality trend chart, recent encounters |
| Encounters | `/samples` | Browse all encounters, filter by version/mode/score |
| Encounter Detail | `/samples/[id]` | Tabs: Transcript, Note, Comparison, Gold, Quality, Compare Versions |
| Providers | `/providers` | Provider cards with quality scores |
| Provider Detail | `/providers/[id]` | Quality history, style directives (read-only) |
| Specialties | `/specialties` | List specialties (read-only) |
| Templates | `/templates` | List templates (read-only) |
| Capture | `/capture` | Record/upload audio → trigger pipeline |

Admin features (create/edit providers, templates, specialties) are **disabled** — those are managed from the pipeline server's admin UI.

### 2.9 Mobile App

The React Native/Expo mobile app (`client/mobile/`) connects to the provider-facing server:

| Screen | API Endpoints Used |
|---|---|
| Record | `GET /providers`, `GET /patients/search`, `POST /encounters`, `POST /encounters/{id}/upload`, `WS /ws/encounters/{id}` |
| Encounters | `GET /encounters` |
| Encounter Detail | `GET /encounters/{id}`, `GET /encounters/{id}/note`, `GET /encounters/{id}/transcript`, `GET /encounters/{id}/audio` |
| Providers | `GET /providers` |
| Settings | Configures API URL (supports cloudflare tunnel URLs for remote access) |

The mobile app's API URL is configurable at runtime via the Settings screen. On first launch it auto-detects the Expo dev server's LAN IP. For remote access (e.g., testing on a physical device), use a Cloudflare tunnel pointed at port 8000.

---

## 3. Processing-Pipeline Server

**Owner:** Pipeline/admin team
**Port:** API 8100, Admin Web UI 3100
**Role value:** `processing-pipeline`
**Start command:**
```bash
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --port 8100
```

### 3.1 Responsibilities

| Responsibility | Description |
|---|---|
| Run ASR pipeline | WhisperX transcription on GPU (CUDA) |
| Run LLM pipeline | Ollama (qwen2.5:14b) note generation |
| Full encounter pipeline | LangGraph state graph: CONTEXT → CAPTURE → TRANSCRIBE → NOTE → REVIEW → DELIVERY |
| Quality evaluation | LLM-as-judge scoring, fact extraction, comparison reports |
| Batch processing | Process all samples with `--two-pass` for VRAM management |
| Admin CRUD | Create/edit providers, templates, specialties |
| Serve admin web UI | Full-featured Next.js UI with all management capabilities |
| Authoritative config | This server is the source of truth for templates, providers, dictionaries |

### 3.2 API Routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/pipeline/upload` | Receive audio + metadata from provider server |
| `POST` | `/pipeline/trigger/{job_id}` | Trigger pipeline execution |
| `GET` | `/pipeline/status/{job_id}` | Poll pipeline job status |
| `GET` | `/pipeline/output/{sample_id}` | List output files for a sample |
| `GET` | `/pipeline/output/{sample_id}/note` | Get generated note |
| `GET` | `/pipeline/output/{sample_id}/transcript` | Get transcript |
| `POST` | `/pipeline/batch/upload` | Batch upload encounter files |
| `POST` | `/pipeline/batch/trigger` | Trigger batch pipeline execution |
| `GET` | `/pipeline/outputs/batch` | Batch retrieve output manifests |
| `GET` | `/encounters` | List encounters (reads local output/) |
| `GET` | `/encounters/{id}` | Encounter detail |
| `GET` | `/encounters/{id}/note` | Generated note |
| `GET` | `/encounters/{id}/transcript` | Transcript |
| `GET` | `/encounters/{id}/quality` | Quality scores |
| `GET` | `/providers` | List providers |
| `GET/POST/PUT` | `/providers/{id}` | Provider CRUD |
| `GET/POST/PUT/DELETE` | `/specialties` | Specialties CRUD |
| `GET/PUT` | `/specialties/{id}/dictionary` | Specialty dictionary CRUD |
| `GET` | `/specialties/audit/consistency` | Cross-reference audit |
| `GET/POST/PUT/DELETE` | `/templates` | Templates CRUD |
| `GET` | `/quality/aggregate` | Aggregate quality stats |
| `GET` | `/quality/samples` | Per-sample quality scores |
| `WS` | `/ws/encounters/{id}` | Real-time pipeline progress |
| `GET` | `/health` | Health check |

### 3.3 Feature Flags (what's enabled)

```yaml
# Enabled on processing-pipeline
dashboard: true
view_encounters: true
view_providers: true
view_specialties: true
view_templates: true
view_quality: true
run_pipeline: true              # GPU pipeline execution
batch_processing: true          # Batch process all samples
create_providers: true          # Admin CRUD
edit_providers: true
create_templates: true
edit_templates: true
create_specialties: true
edit_specialties: true

# DISABLED on processing-pipeline
ehr_access: false               # No EHR — PHI stays on provider server
patient_search: false
record_audio: false             # No audio capture — receives uploads
trigger_pipeline: false         # Uses run_pipeline instead
```

### 3.4 Key Code Files

| File | Purpose |
|---|---|
| `api/main.py` | FastAPI app — mounts pipeline + admin routes based on `cfg.is_processing_pipeline` |
| `api/pipeline/routes.py` | Pipeline API: upload, trigger, status, output retrieval, batch ops |
| `api/routes/specialties.py` | Specialties CRUD + audit endpoint |
| `api/routes/templates.py` | Templates CRUD with specialty validation |
| `api/routes/providers.py` | Providers CRUD with template/specialty validation |
| `api/routes/encounters.py` | Encounter list/detail (reads local output/) |
| `api/routes/quality.py` | Quality metrics |
| `api/quality_runner.py` | Quality evaluation orchestrator |
| `orchestrator/graph.py` | LangGraph encounter pipeline definition |
| `orchestrator/state.py` | `EncounterState` Pydantic schema |
| `orchestrator/nodes/*.py` | Pipeline node implementations (context, transcribe, note, review, delivery) |
| `mcp_servers/asr/whisperx_server.py` | WhisperX ASR server (GPU) |
| `mcp_servers/llm/ollama_server.py` | Ollama LLM server |
| `mcp_servers/registry.py` | Engine registry + health checks |
| `config/deployment.py` | Deployment config loader |
| `config/provider_manager.py` | Provider profile loading |
| `scripts/batch_eval.py` | Batch pipeline runner (`--two-pass --version vN`) |

### 3.5 Data Directories

```
pipeline-data/                     # Audio + metadata received from provider server
├── dictation/
│   └── {provider_id}/
│       └── {sample_id}/
│           ├── dictation.mp3              ← audio from provider server
│           └── encounter_details.json     ← de-identified metadata
└── conversation/
    └── (same structure)

pipeline-output/                   # Pipeline-generated outputs
├── dictation/
│   └── {provider_id}/
│       └── {sample_id}/
│           ├── generated_note_v8.md
│           ├── audio_transcript_v8.txt
│           ├── quality_report.md
│           └── comparison_v8.md
└── conversation/
    └── (same structure)

config/                            # AUTHORITATIVE configuration
├── providers/*.yaml               # Provider profiles (editable here only)
├── templates/*.yaml               # Note templates (editable here only)
├── dictionaries/*.txt             # Specialty dictionaries (editable here only)
├── engines.yaml                   # Engine configuration (ASR, LLM models)
├── quality_baseline.yaml          # Quality baseline thresholds
└── prompts/*.yaml                 # LLM system prompts
```

### 3.6 Pipeline Execution Flow

When the pipeline server receives an upload + trigger request:

```
1. POST /pipeline/upload
   → Saves audio + encounter_details.json to pipeline-data/{mode}/{provider_id}/{sample_id}/
   → Returns job_id

2. POST /pipeline/trigger/{job_id}
   → Loads provider profile from config/providers/
   → Creates EncounterState
   → Launches async pipeline task:
       a. Load ASR (WhisperX, GPU)
       b. Transcribe audio → transcript
       c. Unload ASR from GPU
       d. Load LLM (Ollama qwen2.5:14b)
       e. Generate clinical note from transcript + template + provider style
       f. Run quality evaluation (if gold standard exists)
   → Writes outputs to pipeline-output/{mode}/{provider_id}/{sample_id}/
   → Updates job status: pending → processing → complete | error

3. GET /pipeline/status/{job_id}
   → Returns current stage, progress %, message
```

### 3.7 GPU/VRAM Management

The A10G GPU has 23 GB VRAM. WhisperX peaks at 10-12 GB and qwen2.5:14b needs ~8 GB — they cannot coexist.

**Single encounter:** Pipeline runs ASR first, unloads it, then loads LLM.

**Batch mode (`--two-pass`):**
- Pass 1: ASR only for all samples → saves transcript cache
- Pass 2: LLM only from cache → WhisperX fully freed before LLM loads
- `keep_alive=0` in Ollama config forces model unload after each response

### 3.8 Admin Web UI (Pipeline Server)

The Next.js web app at `client/web/` serves a **full admin UI** on port 3100:

| Page | Path | Capabilities |
|---|---|---|
| Dashboard | `/` | KPI cards, quality trends, dimension radar |
| Encounters | `/samples` | Browse + re-run pipeline on any sample |
| Encounter Detail | `/samples/[id]` | Full viewer + "Re-run Pipeline" button |
| Providers | `/providers` | List + **create/edit** providers |
| Provider Detail | `/providers/[id]` | Edit specialty, templates, vocabulary, style directives |
| Specialties | `/specialties` | List + **create/edit** specialties with inline dictionary editor |
| Templates | `/templates` | List + **create/edit** templates with section builder |
| Capture | `/capture` | Not applicable (no EHR/patient access) |

---

## 4. Inter-Server Communication

### 4.1 Request Flow: New Encounter

```
Mobile/Web App
    │
    ├─ POST /encounters ──────────→ Provider-Facing Server
    │                                   │
    │                                   ├── Saves patient_demographics.json locally (PHI)
    │                                   ├── Saves encounter_details.json locally
    │                                   │
    │                                   ├── proxy_upload() ──→ Pipeline Server
    │                                   │     POST /pipeline/upload
    │                                   │     (audio + encounter_details only, NO PHI)
    │                                   │
    │                                   ├── proxy_trigger() ──→ Pipeline Server
    │                                   │     POST /pipeline/trigger/{job_id}
    │                                   │
    │                                   └── WebSocket progress → Client
    │
    └─ GET /encounters/{id}/note ──→ Provider-Facing Server
                                        │
                                        └── Reads from local output/
                                            (synced from pipeline server)
```

### 4.2 Data Sync Flows

```
Pipeline Server ───(every 2h)───→ Provider-Facing Server
  GET /providers                    writes config/providers/*.yaml
  GET /templates                    writes config/templates/*.yaml
  GET /specialties                  writes config/dictionaries/*.txt

Pipeline Server ───(on demand)──→ Provider-Facing Server
  GET /pipeline/outputs/batch       manifest of available outputs
  GET /pipeline/output/{id}/note    generated note content
  GET /pipeline/output/{id}/transcript  transcript content
                                    ↓
                            writes to local output/ directory
```

### 4.3 Authentication

Inter-server auth is controlled by `config/deployment.yaml`:

```yaml
security:
  inter_server_auth:
    enabled: true                 # Set true in production
    secret_env_var: "AI_SCRIBE_INTER_SERVER_SECRET"
```

When enabled:
- Provider-facing adds `X-Inter-Server-Auth: {secret}` header to all proxy requests
- Pipeline server validates the header on `/pipeline/*` routes
- Secret is read from the environment variable, never stored in config files

### 4.4 PHI Isolation Rules

| Data | Stays on Provider Server | Sent to Pipeline Server |
|---|---|---|
| `patient_demographics.json` | YES | NO |
| `patient_context.yaml` | YES | NO |
| `final_soap_note.md` | YES | NO |
| Audio files (`.mp3`) | YES (stored) | YES (for ASR) |
| `encounter_details.json` | YES (stored) | YES (de-identified) |
| Generated notes | Synced back | YES (generated here) |
| Templates, providers | Synced from pipeline | YES (authoritative source) |

---

## 5. Configuration

### 5.1 `config/deployment.yaml`

This is the central deployment configuration file. Key sections:

```yaml
server:
  role: "provider-facing"          # "provider-facing" | "processing-pipeline"
  instance_id: "ai-scribe-dev-01"

network:
  provider_facing:
    api_port: 8000
    web_port: 3000
    public_url: "http://localhost:8000"
  processing_pipeline:
    api_port: 8100
    web_port: 3100
    internal_url: "http://localhost:8100"

data:
  provider_facing:
    data_dir: "ai-scribe-data"     # encounters + PHI
    output_dir: "output"           # synced from pipeline
  processing_pipeline:
    data_dir: "pipeline-data"      # received audio + metadata
    output_dir: "pipeline-output"  # generated outputs

sync:
  config_sync:
    enabled: true
    interval_seconds: 7200         # 2 hours

features:
  provider_facing:
    ehr_access: true
    trigger_pipeline: true
    # admin CRUD: false
  processing_pipeline:
    run_pipeline: true
    batch_processing: true
    # all admin CRUD: true
```

### 5.2 `config/deployment.py`

The Python module that loads and validates the YAML config:

- `ServerRole` enum: `PROVIDER_FACING`, `PROCESSING_PIPELINE`
- `DeploymentConfig` Pydantic model with convenience properties:
  - `is_provider_facing` / `is_processing_pipeline` — exactly one is True
  - `active_features` — feature flags for the current role
  - `api_port` / `web_port` — correct port for the current role
  - `pipeline_api_url` — URL to reach the pipeline server
- `get_deployment_config()` — singleton loader, supports env var overrides
- `require_feature(name)` — raises `HTTPException(403)` if feature not enabled

### 5.3 Environment Variable Overrides

| Variable | Purpose |
|---|---|
| `AI_SCRIBE_SERVER_ROLE` | Override server role (bypasses YAML) |
| `AI_SCRIBE_DEPLOYMENT_CONFIG` | Custom path to deployment.yaml |
| `AI_SCRIBE_INTER_SERVER_SECRET` | Shared secret for inter-server auth |

---

## 6. Development Setup

### Single Machine (two instances, same machine)

```bash
# Terminal 1: Provider-facing server
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --port 8000

# Terminal 2: Processing-pipeline server
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --port 8100

# Terminal 3: Provider web UI
cd client/web && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev -- --port 3000

# Terminal 4: Admin web UI
cd client/web && NEXT_PUBLIC_API_URL=http://localhost:8100 npm run dev -- --port 3100
```

### Two Machines (production)

1. Set `config/deployment.yaml` on each machine with the correct `role`
2. Update `network.processing_pipeline.internal_url` to point to the pipeline server's IP
3. Update `network.provider_facing.public_url` to the provider server's public URL
4. Enable `inter_server_auth` and set the shared secret on both machines
5. Start each server with its role:
   ```bash
   # Provider machine:
   AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --port 8000

   # Pipeline machine (with GPU):
   AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --port 8100
   ```

---

## 7. Adding New Features — Which Server?

| Feature Type | Server | Why |
|---|---|---|
| New read-only data view | Provider-facing | Serves the client UI |
| New admin CRUD operation | Processing-pipeline | Authoritative config owner |
| New pipeline stage/node | Processing-pipeline | Runs GPU workloads |
| New EHR integration | Provider-facing | PHI access is local only |
| New mobile screen | Provider-facing API | Mobile connects here |
| New quality metric | Processing-pipeline | Quality eval runs on GPU |
| New sync item | Both servers | Sync module on provider, API on pipeline |

---

## 8. Testing

Both servers share the same codebase (`api/`, `config/`, `orchestrator/`). Test files are in `tests/`:

```
tests/
├── unit/
│   ├── test_deployment_config.py   # ServerRole, feature flags, config loading
│   ├── test_proxy.py               # Proxy functions, needs_proxy()
│   ├── test_sync.py                # Config sync logic
│   ├── test_pipeline_routes.py     # Pipeline API endpoints
│   ├── test_encounter_routes.py    # Encounter routes
│   ├── test_provider_routes.py     # Provider CRUD
│   ├── test_template_routes.py     # Template CRUD
│   └── test_specialty_routes.py    # Specialty CRUD
├── integration/
│   └── test_inter_server.py        # Provider ↔ Pipeline communication
└── e2e/
    └── test_full_pipeline.py       # Audio → transcript → note → quality
```

Run tests with:
```bash
source .venv/bin/activate
pytest tests/unit/ -v
```
