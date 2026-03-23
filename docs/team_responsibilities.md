# AI Scribe Enterprise — Team Responsibilities

Two teams work on this codebase. This document defines exactly what each team owns, builds, and is responsible for.

---

## Team A: Provider Server + Mobile App

**Mission:** Everything the clinician touches — the provider-facing server, the mobile app, client deployment, and connectivity between all pieces.

### What You Own

| Area | Key Files | Description |
|------|-----------|-------------|
| **Provider-facing API** | `api/main.py` (provider routes), `api/routes/encounters.py`, `api/routes/patients.py` | FastAPI server on port 8000 that serves the web and mobile clients |
| **Proxy layer** | `api/proxy.py` | Forwards pipeline requests (upload, trigger, status) to the pipeline server. You own the client side of this boundary. |
| **Config sync** | `api/sync.py` | Background task that pulls templates/providers/specialties from the pipeline server every 2 hours. You own the pull side. |
| **Mobile app** | `client/mobile/` (entire directory) | React Native/Expo app — recording, encounter list, provider list, settings |
| **Provider web UI** | `client/web/` (when connected to port 8000) | Next.js app running as a read-only provider dashboard on port 3000 |
| **EHR stub / integration** | `mcp_servers/ehr/stub_server.py`, `config/ehr_stub/` | Patient roster, patient search. Future: live EHR (FHIR R4, HL7) |
| **Client deployment** | `config/deployment.yaml` (provider-facing settings), `deploy/` | Installing, configuring, and running the provider server at the client site |
| **Data directories** | `ai-scribe-data/`, `output/` | Audio + PHI storage, synced outputs from pipeline |

### What You Build Next

1. **Deployment scripts** — Automate standing up the provider-facing server at a client site:
   - Install dependencies (CPU only, no GPU)
   - Configure `deployment.yaml` with the pipeline server's address
   - Set up `AI_SCRIBE_INTER_SERVER_SECRET` for auth
   - Configure nginx/reverse proxy for HTTPS
   - systemd service for the FastAPI process
   - Verify connectivity to the pipeline server (`GET /health`)

2. **Mobile app hardening** — The app works but needs production polish:
   - Persistent API URL configuration (already done via Settings screen)
   - Offline audio caching (store recorded audio locally, upload when online)
   - Push notifications for pipeline completion
   - Error handling for network failures and timeouts

3. **EHR integration** — Replace the stub patient roster with live EHR:
   - `mcp_servers/ehr/fhir_server.py` — FHIR R4 adapter (Epic, Cerner, Athena)
   - Patient search, demographics pull, problem list, medications
   - Note push-back to EHR after provider approval
   - PHI stays on this server — never sent to pipeline

4. **Output sync improvements** — Currently basic; needs:
   - On-demand sync when a pipeline job completes (triggered by WebSocket event)
   - Incremental sync (only fetch new/changed files)
   - Conflict resolution for outputs modified on both sides

5. **Provider web UI enhancements** — Currently read-only; add:
   - Note review + editing (provider corrects generated note)
   - Correction capture (diff: AI output vs provider edit) → save as training data
   - Provider-friendly labels and branding per practice

### Your API Endpoints

These routes run **only on the provider-facing server** (port 8000):

| Endpoint | What it does |
|----------|--------------|
| `GET /patients/search?q=` | Search patient roster (local EHR data) |
| `POST /encounters` | Create encounter → saves PHI locally, proxies audio to pipeline |
| `POST /encounters/{id}/upload` | Upload audio → proxies to pipeline |
| `GET /encounters/{id}/status` | Poll pipeline progress → proxies to pipeline |

These routes run on **both servers** (you read from local `output/`; pipeline reads from `pipeline-output/`):

| Endpoint | What it does |
|----------|--------------|
| `GET /encounters` | List encounters from local output directory |
| `GET /encounters/{id}` | Encounter detail |
| `GET /encounters/{id}/note` | Generated clinical note |
| `GET /encounters/{id}/transcript` | Transcript |
| `GET /providers` | List providers (read-only on your server) |
| `GET /quality/aggregate` | Quality stats |

### PHI Rules (Critical)

You are the **only server that holds PHI**. Enforce these rules:

- `patient_demographics.json` — NEVER sent to the pipeline server
- `patient_context.yaml` — NEVER sent to the pipeline server
- `final_soap_note.md` — NEVER sent to the pipeline server
- Only `audio files` and `encounter_details.json` (de-identified) cross the boundary
- The proxy layer (`api/proxy.py`) enforces this — do not bypass it

### How You Connect to the Pipeline Server

```yaml
# config/deployment.yaml — your key settings
server:
  role: "provider-facing"

network:
  processing_pipeline:
    internal_url: "http://<pipeline-server-ip>:8100"  # ← set this

security:
  inter_server_auth:
    enabled: true  # ← enable in production
    # Secret via env var: AI_SCRIBE_INTER_SERVER_SECRET
```

Start command:
```bash
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### How the Mobile App Connects

The mobile app connects to your server (port 8000). For remote access:

1. Expose port 8000 via HTTPS (nginx + TLS cert, or Cloudflare tunnel)
2. Users set the URL in the mobile app's Settings screen
3. The app auto-detects the LAN IP during Expo development

---

## Team B: Pipeline Server + Shared API

**Mission:** Everything that processes audio and generates notes — the GPU pipeline, the admin UI, the shared API layer, quality evaluation, and all backend infrastructure.

### What You Own

| Area | Key Files | Description |
|------|-----------|-------------|
| **Pipeline API** | `api/pipeline/routes.py` | Upload, trigger, status, output retrieval, batch operations on port 8100 |
| **Shared API routes** | `api/routes/encounters.py`, `api/routes/quality.py`, `api/routes/providers.py`, `api/routes/specialties.py`, `api/routes/templates.py` | CRUD endpoints used by both servers. You own the implementation. |
| **FastAPI app** | `api/main.py` | Role-based route mounting, lifespan, CORS. You own the app structure. |
| **Encounter pipeline** | `orchestrator/` (entire directory) | LangGraph state graph: CONTEXT → CAPTURE → TRANSCRIBE → NOTE → REVIEW → DELIVERY |
| **MCP servers** | `mcp_servers/` (entire directory) | ASR (WhisperX), LLM (Ollama), audio processing, data/reference servers |
| **Post-processor** | `postprocessor/` | 12-stage rule-based transcript cleanup + ML pipeline |
| **Quality framework** | `quality/` | LLM-as-judge evaluator, fact extraction, dimension scoring, optimization |
| **Batch processing** | `scripts/batch_eval.py`, `scripts/run_quality_sweep.py` | Bulk pipeline execution and quality evaluation |
| **Admin web UI** | `client/web/` (when connected to port 8100) | Next.js app on port 3100 — full CRUD for providers, templates, specialties |
| **Config (authoritative)** | `config/` (all files) | Templates, providers, dictionaries, engines, prompts — you are the source of truth |
| **Deployment config** | `config/deployment.py`, `config/deployment.yaml`, `config/paths.py` | Server role system, feature flags, data directory resolution |
| **Data directories** | `pipeline-data/`, `pipeline-output/` | Received audio, generated outputs |
| **Output generation** | `output/` | Markdown writer, comparison writer, batch reports |
| **Tests** | `tests/` (entire directory) | Unit, integration, E2E tests for all backend code |

### What You Build Next

1. **Shared API contract** — You define and maintain the API endpoints that both servers use:
   - Document request/response schemas (already Pydantic models)
   - Version the API if breaking changes are needed
   - Ensure the provider team's proxy layer (`api/proxy.py`) stays compatible with your pipeline routes

2. **Pipeline improvements** — Ongoing quality work:
   - Better ASR accuracy (ambient mode optimization, provider-specific tuning)
   - LLM prompt engineering for note quality
   - New MCP server integrations (NeMo streaming, vLLM)
   - LoRA fine-tuning pipeline (when verbatim training data is available)

3. **Admin CRUD hardening** — The admin UI works but needs:
   - Validation improvements (circular template references, orphaned providers)
   - Audit logging for config changes
   - Rollback capability for config changes

4. **Batch processing improvements**:
   - API-driven batch execution (currently uses `scripts/batch_eval.py`)
   - Progress tracking for batch jobs
   - Automatic quality regression testing after pipeline changes

5. **Learning loop** — Session 15:
   - Capture provider corrections from Team A's review UI
   - Classify corrections (ASR_ERROR, STYLE, CONTENT, CODING, TEMPLATE)
   - Generate training pairs for post-processor ML model
   - Update provider style models from edit patterns

6. **Infrastructure**:
   - GPU memory management (two-pass batch, model unloading)
   - Monitoring (Grafana + Prometheus dashboards)
   - Docker deployment for the pipeline server

### Your API Endpoints

These routes run **only on the pipeline server** (port 8100):

| Endpoint | What it does |
|----------|--------------|
| `POST /pipeline/upload` | Receive audio + metadata from provider server |
| `POST /pipeline/trigger/{job_id}` | Launch ASR → LLM pipeline (async, GPU) |
| `GET /pipeline/status/{job_id}` | Pipeline job progress |
| `GET /pipeline/output/{id}/note` | Serve generated note back to provider server |
| `GET /pipeline/output/{id}/transcript` | Serve transcript back to provider server |
| `POST /pipeline/batch/upload` | Batch upload from provider server |
| `POST /pipeline/batch/trigger` | Batch pipeline execution |
| `GET/POST/PUT /providers` | Full provider CRUD |
| `GET/POST/PUT/DELETE /templates` | Full template CRUD |
| `GET/POST/PUT/DELETE /specialties` | Full specialty CRUD |
| `GET /specialties/audit/consistency` | Cross-reference validation |

### GPU / VRAM Constraints

The pipeline server runs on an A10G (23 GB VRAM):
- WhisperX peaks at 10-12 GB
- qwen2.5:14b needs ~8 GB
- They **cannot coexist** — pipeline unloads ASR before loading LLM
- Batch mode uses `--two-pass`: ASR all samples first, then LLM all samples
- `keep_alive=0` in Ollama config forces model unload after each response

### How Provider Server Connects to You

Team A's proxy sends requests to your `/pipeline/*` endpoints. You need to:
- Accept uploads at `POST /pipeline/upload` (audio + encounter_details.json)
- Return `job_id` for status tracking
- Run pipeline asynchronously and update job status
- Serve outputs at `GET /pipeline/output/{id}/note` and `/transcript`
- Support the config sync: expose `GET /providers`, `GET /templates`, `GET /specialties` with full data

Start command:
```bash
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --host 0.0.0.0 --port 8100
```

---

## Shared Boundaries

### Code Both Teams Touch

| File | Team A Uses | Team B Owns |
|------|-------------|-------------|
| `api/routes/encounters.py` | Reads encounter data, proxies pipeline ops | Implements the route logic, data loading |
| `api/routes/providers.py` | Read-only access | Full CRUD implementation |
| `api/data_loader.py` | Reads local output/ | Implements the scanning/loading logic |
| `config/deployment.py` | Reads role, feature flags | Defines the role system and feature flags |
| `config/deployment.yaml` | Sets provider-facing config | Defines the full config schema |

**Rule:** Team B owns the implementation of shared routes. Team A consumes them. If Team A needs a new endpoint or a change to an existing one, coordinate with Team B.

### API Contract

The inter-server API contract is defined by these files:

| File | Defines |
|------|---------|
| `api/pipeline/routes.py` | Pipeline API schemas (`PipelineUploadResponse`, `PipelineTriggerRequest`, `PipelineStatusResponse`) |
| `api/proxy.py` | What the provider server sends and expects back |

**Breaking change protocol:** If Team B changes a pipeline API schema, Team A's proxy must be updated in the same PR. Both teams should review changes to `api/proxy.py` and `api/pipeline/routes.py`.

### Feature Flags

Feature flags (`config/deployment.py`) gate what each server can do:

```python
# Team A's server has these enabled:
ehr_access: True
patient_search: True
trigger_pipeline: True      # proxied to Team B's server
record_audio: True

# Team B's server has these enabled:
run_pipeline: True
batch_processing: True
create_providers: True
edit_templates: True
# etc.
```

Team B owns the feature flag system. Team A uses `require_feature()` in any new routes.

---

## Development Workflow

### Development Setup

For local development, run two instances on the same machine:

### Two Instances (same machine)
```bash
# Terminal 1: Team A's server
AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --port 8000

# Terminal 2: Team B's server
AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --port 8100

# Terminal 3: Provider web UI
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev -- --port 3000

# Terminal 4: Admin web UI
NEXT_PUBLIC_API_URL=http://localhost:8100 npm run dev -- --port 3100
```

### Running Tests

```bash
source .venv/bin/activate
pytest tests/unit/ -v                    # All unit tests (213 passing)
pytest tests/unit/test_proxy.py -v       # Team A: proxy tests
pytest tests/unit/test_pipeline_routes.py -v  # Team B: pipeline tests
```

---

## Deployment Checklist (Team A)

When deploying at a client site:

- [ ] Provision CPU server (no GPU needed)
- [ ] Install Python 3.11+, Node.js 18+
- [ ] Clone repo, install dependencies (`pip install -e .`, `npm install`)
- [ ] Set `AI_SCRIBE_SERVER_ROLE=provider-facing`
- [ ] Edit `config/deployment.yaml`:
  - `network.processing_pipeline.internal_url` → pipeline server address
  - `network.provider_facing.public_url` → this server's public URL
- [ ] Set env vars:
  - `AI_SCRIBE_INTER_SERVER_SECRET` (match pipeline server)
- [ ] Start API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- [ ] Start web UI: `NEXT_PUBLIC_API_URL=https://this-server npm run build && npm start`
- [ ] Verify: `curl http://localhost:8000/health`
- [ ] Verify pipeline connectivity: `curl http://localhost:8000/config/role`
- [ ] Configure HTTPS (nginx + TLS) for mobile app access
- [ ] Share the HTTPS URL with mobile app users

## Deployment Checklist (Team B)

When standing up the pipeline server:

- [ ] Provision GPU server (A10G 24GB recommended)
- [ ] Install CUDA toolkit, Python 3.11+, Node.js 18+
- [ ] Install Ollama, pull models: `ollama pull qwen2.5:14b`, `ollama pull llama3.1:latest`
- [ ] Clone repo, install dependencies
- [ ] Set `AI_SCRIBE_SERVER_ROLE=processing-pipeline`
- [ ] Set env vars:
  - `AI_SCRIBE_INTER_SERVER_SECRET` (match provider server)
  - `HF_TOKEN` (for pyannote diarization model)
- [ ] Start API: `uvicorn api.main:app --host 0.0.0.0 --port 8100`
- [ ] Start admin UI: `NEXT_PUBLIC_API_URL=http://localhost:8100 npm run build && npm start -- --port 3100`
- [ ] Verify: `curl http://localhost:8100/health`
- [ ] Verify GPU: pipeline should load WhisperX on first encounter
- [ ] Open port 8100 to the provider server's IP (firewall rule)
