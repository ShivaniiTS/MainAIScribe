"""
api/main.py — FastAPI application entry point (role-aware).

The server role is determined by config/deployment.yaml or the
AI_SCRIBE_SERVER_ROLE environment variable:

  - "provider-facing"          — Client UI, EHR access, proxies to pipeline server
  - "processing-pipeline"      — Pipeline API, admin UI, GPU workloads

Start with:
    # Provider-facing server:
    AI_SCRIBE_SERVER_ROLE=provider-facing uvicorn api.main:app --port 8000

    # Processing pipeline server:
    AI_SCRIBE_SERVER_ROLE=processing-pipeline uvicorn api.main:app --port 8100
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config.deployment import get_deployment_config, ServerRole

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Lifespan — starts/stops background tasks based on role
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_deployment_config()
    log.info("ai_scribe_api_starting", role=cfg.role.value, instance=cfg.instance_id)

    # Start config sync on provider-facing server
    if cfg.is_provider_facing:
        from api.sync import start_config_sync
        await start_config_sync()

    yield

    # Cleanup
    if cfg.is_provider_facing:
        from api.sync import stop_config_sync
        await stop_config_sync()
        from api.proxy import close
        await close()

    log.info("ai_scribe_api_shutdown", role=cfg.role.value)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
cfg = get_deployment_config()

app = FastAPI(
    title=f"AI Scribe Enterprise API ({cfg.role.value})",
    description=(
        "REST + WebSocket API for the AI medical scribe pipeline. "
        f"Server role: {cfg.role.value}."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Inter-server auth middleware (when enabled)
# ---------------------------------------------------------------------------
if cfg.inter_server_auth.enabled:
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def inter_server_auth_middleware(request: Request, call_next):
        # Only protect /pipeline/* routes
        if request.url.path.startswith("/pipeline/"):
            expected = cfg.inter_server_auth.secret
            provided = request.headers.get("X-Inter-Server-Auth")
            if expected and provided != expected:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid inter-server auth token"},
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Routes — included based on server role
# ---------------------------------------------------------------------------

# Shared routes (all roles): encounters (read), quality, websocket
from api.routes import encounters, quality
from api.ws import session_events

app.include_router(encounters.router)
app.include_router(quality.router)
app.include_router(session_events.router)

# Audio streaming: pipeline server runs NeMo directly; provider-facing proxies
if cfg.is_processing_pipeline:
    from api.ws import audio_stream
    app.include_router(audio_stream.router)

if cfg.is_provider_facing:
    from api.ws.asr_proxy import router as asr_proxy_router
    app.include_router(asr_proxy_router)

# Provider-facing routes: patients (EHR), providers (read)
if cfg.is_provider_facing:
    from api.routes import patients, providers, specialties, templates
    app.include_router(patients.router)
    app.include_router(providers.router)
    app.include_router(specialties.router)
    app.include_router(templates.router)

# Processing pipeline routes: pipeline API, admin CRUD
if cfg.is_processing_pipeline:
    from api.pipeline.routes import router as pipeline_router
    from api.routes import providers as admin_providers
    from api.routes import specialties, templates
    app.include_router(pipeline_router)
    app.include_router(admin_providers.router)
    app.include_router(specialties.router)
    app.include_router(templates.router)


# ---------------------------------------------------------------------------
# Health + info endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["health"])
def root():
    return {
        "service": "AI Scribe Enterprise API",
        "version": "1.0.0",
        "role": cfg.role.value,
        "instance_id": cfg.instance_id,
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "role": cfg.role.value,
    }


@app.get("/config/features", tags=["config"])
def get_features():
    """Return the active feature flags for this server instance."""
    return cfg.active_features.model_dump()


@app.get("/config/role", tags=["config"])
def get_role():
    """Return the server role and deployment info."""
    return {
        "role": cfg.role.value,
        "instance_id": cfg.instance_id,
        "is_provider_facing": cfg.is_provider_facing,
        "is_processing_pipeline": cfg.is_processing_pipeline,
        "pipeline_api_url": cfg.pipeline_api_url if cfg.is_provider_facing else None,
    }


@app.get("/config/latest-version", tags=["config"])
def get_latest_version():
    """Return the latest pipeline version (dynamically discovered from output files)."""
    from api.data_loader import get_latest_version as _glv, get_versions
    return {
        "latest": _glv(),
        "versions": get_versions(),
    }
