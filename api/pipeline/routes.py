"""
api/pipeline/routes.py — Pipeline API routes (processing-pipeline server).

Accepts file uploads from the provider-facing server, triggers pipeline
execution, tracks progress, and serves generated outputs back.

Endpoints:
    POST /pipeline/upload           — Upload encounter files (audio + metadata)
    POST /pipeline/trigger/{id}     — Trigger pipeline for an uploaded encounter
    POST /pipeline/batch/upload     — Upload batch of encounter files
    POST /pipeline/batch/trigger    — Trigger batch pipeline execution
    GET  /pipeline/status/{id}      — Poll pipeline status
    GET  /pipeline/output/{id}      — Retrieve generated outputs
    GET  /pipeline/output/{id}/note — Retrieve generated note
    GET  /pipeline/output/{id}/transcript — Retrieve transcript
    GET  /pipeline/outputs/batch    — Batch retrieval of outputs
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from config.deployment import get_deployment_config, require_feature

logger = structlog.get_logger()
router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# ---------------------------------------------------------------------------
# In-memory pipeline job tracking
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}


class PipelineUploadResponse(BaseModel):
    job_id: str
    sample_id: str
    status: str
    message: str


class PipelineTriggerRequest(BaseModel):
    mode: str = "dictation"  # "dictation" | "ambient"
    provider_id: str = ""
    visit_type: str = "follow_up"
    version: str | None = None  # Auto-detect if not specified


class PipelineStatusResponse(BaseModel):
    job_id: str
    sample_id: str
    status: str  # "pending" | "processing" | "complete" | "error"
    stage: str | None = None
    pct: int = 0
    message: str = ""
    version: str | None = None


class BatchUploadItem(BaseModel):
    sample_id: str
    mode: str
    provider_id: str
    has_audio: bool = True


class BatchUploadResponse(BaseModel):
    uploaded: int
    skipped: int
    errors: list[str]


class BatchTriggerRequest(BaseModel):
    sample_ids: list[str] | None = None  # None = all pending
    version: str | None = None
    two_pass: bool = True


class OutputFile(BaseModel):
    filename: str
    size: int
    modified: str


class OutputListResponse(BaseModel):
    sample_id: str
    files: list[OutputFile]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_pipeline_data_dir() -> Path:
    """Return the data directory for this server role (resolved by config/paths.py)."""
    from config.paths import DATA_DIR
    return DATA_DIR


def _get_pipeline_output_dir() -> Path:
    """Return the output directory for this server role (resolved by config/paths.py)."""
    from config.paths import OUTPUT_DIR
    return OUTPUT_DIR


def _find_sample_dirs(sample_id: str) -> tuple[Path | None, Path | None, str, str]:
    """Find data and output directories for a sample. Returns (data_dir, output_dir, mode, physician)."""
    data_root = _get_pipeline_data_dir()
    output_root = _get_pipeline_output_dir()

    for mode_dir in ["dictation", "conversation"]:
        mode_path = data_root / mode_dir
        if not mode_path.exists():
            continue
        for physician_dir in mode_path.iterdir():
            if not physician_dir.is_dir():
                continue
            sample_dir = physician_dir / sample_id
            if sample_dir.exists():
                out_dir = output_root / mode_dir / physician_dir.name / sample_id
                return sample_dir, out_dir, mode_dir, physician_dir.name

    return None, None, "", ""


# ---------------------------------------------------------------------------
# Upload encounter files
# ---------------------------------------------------------------------------
@router.post("/upload", response_model=PipelineUploadResponse)
async def upload_encounter(
    audio: UploadFile = File(...),
    sample_id: str = Form(...),
    mode: str = Form("dictation"),
    provider_id: str = Form(""),
    encounter_details: str = Form("{}"),
):
    """
    Upload audio + metadata for a single encounter.

    The provider-facing server sends audio + encounter metadata here.
    PHI (demographics, patient names) is NOT sent — only audio and
    de-identified encounter details.
    """
    require_feature("run_pipeline")

    data_root = _get_pipeline_data_dir()

    data_mode = "conversation" if mode == "ambient" else mode
    encounter_dir = data_root / data_mode / provider_id / sample_id
    encounter_dir.mkdir(parents=True, exist_ok=True)

    # Save audio file
    audio_filename = "dictation.mp3" if mode == "dictation" else "conversation_audio.mp3"
    audio_path = encounter_dir / audio_filename
    content = await audio.read()
    audio_path.write_bytes(content)

    # Save encounter details (de-identified)
    try:
        details = json.loads(encounter_details)
    except json.JSONDecodeError:
        details = {}
    details.setdefault("mode", mode)
    details.setdefault("provider_id", provider_id)
    details.setdefault("audio_file", audio_filename)
    (encounter_dir / "encounter_details.json").write_text(json.dumps(details, indent=2))

    job_id = f"job-{str(uuid.uuid4())[:8]}"
    _jobs[job_id] = {
        "job_id": job_id,
        "sample_id": sample_id,
        "status": "pending",
        "stage": None,
        "pct": 0,
        "message": "Files uploaded — ready for pipeline trigger",
        "mode": mode,
        "provider_id": provider_id,
        "data_dir": str(encounter_dir),
    }

    logger.info("pipeline_upload", job_id=job_id, sample_id=sample_id)

    return PipelineUploadResponse(
        job_id=job_id,
        sample_id=sample_id,
        status="pending",
        message="Files uploaded — ready for pipeline trigger",
    )


# ---------------------------------------------------------------------------
# Trigger pipeline
# ---------------------------------------------------------------------------
@router.post("/trigger/{job_id}", response_model=PipelineStatusResponse)
async def trigger_pipeline(job_id: str, req: PipelineTriggerRequest | None = None):
    """Trigger the pipeline for an uploaded encounter."""
    require_feature("run_pipeline")

    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Pipeline already running for this job")

    mode = req.mode if req else job.get("mode", "dictation")
    provider_id = req.provider_id if req and req.provider_id else job.get("provider_id", "")
    visit_type = req.visit_type if req else "follow_up"

    # Find audio file
    data_dir = Path(job["data_dir"])
    audio_path = None
    for candidate in ["dictation.mp3", "conversation_audio.mp3", "notes.mp3"]:
        p = data_dir / candidate
        if p.exists():
            audio_path = str(p)
            break
    if not audio_path:
        raise HTTPException(status_code=404, detail="No audio file found in uploaded data")

    # Determine output directory
    cfg = get_deployment_config()
    output_root = _get_pipeline_output_dir()
    sample_id = job["sample_id"]
    data_mode = "conversation" if mode == "ambient" else "dictation"

    out_dir = output_root / data_mode / provider_id / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect version
    version = req.version if req and req.version else None
    if not version:
        existing = sorted(
            [int(f.stem.split("_v")[1]) for f in out_dir.glob("generated_note_v*.md")
             if f.stem.split("_v")[1].isdigit()],
            reverse=True,
        )
        if existing:
            next_num = existing[0] + 1
        else:
            from api.data_loader import get_latest_version
            latest = get_latest_version()
            next_num = int(latest[1:]) + 1 if latest != "v1" else 1
        version = f"v{next_num}"

    job["status"] = "processing"
    job["message"] = f"Pipeline started → {version}"
    job["version"] = version

    # Check for note audio (conversation mode)
    note_audio_path = None
    if data_mode == "conversation":
        for candidate in ["notes.mp3", "note_audio.mp3"]:
            p = data_dir / candidate
            if p.exists():
                note_audio_path = str(p)
                break

    asyncio.create_task(
        _run_pipeline(
            job_id=job_id,
            sample_id=sample_id,
            audio_path=audio_path,
            note_audio_path=note_audio_path,
            mode=mode,
            provider_id=provider_id,
            visit_type=visit_type,
            output_dir=str(out_dir),
            version=version,
        )
    )

    return PipelineStatusResponse(
        job_id=job_id,
        sample_id=sample_id,
        status="processing",
        stage="init",
        pct=0,
        message=f"Pipeline started → {version}",
        version=version,
    )


# ---------------------------------------------------------------------------
# Pipeline execution (background)
# ---------------------------------------------------------------------------
async def _run_pipeline(
    job_id: str,
    sample_id: str,
    audio_path: str,
    note_audio_path: str | None,
    mode: str,
    provider_id: str,
    visit_type: str,
    output_dir: str,
    version: str,
) -> None:
    """Run the pipeline in the background, updating job status."""
    job = _jobs.get(job_id, {})

    try:
        job["stage"] = "init"
        job["pct"] = 5
        job["message"] = "Loading provider profile..."

        from config.provider_manager import get_provider_manager
        mgr = get_provider_manager()
        profile = mgr.load_or_default(provider_id)

        job["pct"] = 10
        job["message"] = "Building pipeline..."

        from orchestrator.state import (
            DeliveryMethod,
            EncounterState,
            RecordingMode,
        )

        recording_mode = RecordingMode.AMBIENT if mode == "ambient" else RecordingMode.DICTATION

        state = EncounterState(
            provider_id=profile.id,
            patient_id=f"patient-{sample_id}",
            provider_profile=profile,
            recording_mode=recording_mode,
            delivery_method=DeliveryMethod.CLIPBOARD,
            audio_file_path=audio_path,
            note_audio_file_path=note_audio_path,
        )

        job["stage"] = "transcribe"
        job["pct"] = 20
        job["message"] = "Starting ASR transcription..."

        from orchestrator.graph import build_graph, run_encounter
        graph = build_graph()

        final = await asyncio.to_thread(run_encounter, graph, state)

        # Free ASR model from GPU memory before LLM quality evaluation
        try:
            from mcp_servers.registry import get_registry
            get_registry().unload_engine("asr")
        except Exception:
            pass  # Non-critical — just memory optimization

        job["pct"] = 50
        job["message"] = "Transcription complete"

        out_path = Path(output_dir)

        if final.final_note:
            from output.markdown_writer import write_clinical_note
            write_clinical_note(
                final,
                path=out_path / f"generated_note_{version}.md",
                version=version,
                sample_id=sample_id,
            )
            job["stage"] = "note"
            job["pct"] = 80
            job["message"] = "Clinical note generated"

        if final.transcript and final.transcript.full_text.strip():
            transcript_path = out_path / f"audio_transcript_{version}.txt"
            transcript_path.write_text(final.transcript.full_text.strip())
            job["pct"] = 85
            job["message"] = "Transcript saved"

        # Run quality evaluation if gold standard exists
        job["stage"] = "quality"
        job["pct"] = 90
        job["message"] = "Running quality evaluation..."

        try:
            from api.quality_runner import evaluate_sample
            from api.data_loader import get_gold_note

            gold = get_gold_note(sample_id)
            gen_text = (out_path / f"generated_note_{version}.md").read_text() if (out_path / f"generated_note_{version}.md").exists() else ""
            tx_text = final.transcript.full_text.strip() if final.transcript else ""

            if gold:
                await asyncio.to_thread(
                    evaluate_sample,
                    sample_id=sample_id,
                    generated_note=gen_text,
                    gold_note=gold,
                    transcript=tx_text,
                    version=version,
                    output_dir=out_path,
                )
        except Exception as qe:
            logger.warning("quality_eval_skipped", job_id=job_id, error=str(qe))

        job["stage"] = "delivery"
        job["pct"] = 100
        job["status"] = "complete"
        job["message"] = f"Pipeline complete — {version} generated"

        logger.info("pipeline_job_complete", job_id=job_id, sample_id=sample_id, version=version)

    except Exception as e:
        logger.error("pipeline_job_error", job_id=job_id, error=str(e))
        job["status"] = "error"
        job["message"] = f"Pipeline error: {str(e)}"


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------
@router.get("/status/{job_id}", response_model=PipelineStatusResponse)
def get_pipeline_status(job_id: str):
    """Poll pipeline execution status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return PipelineStatusResponse(
        job_id=job["job_id"],
        sample_id=job["sample_id"],
        status=job["status"],
        stage=job.get("stage"),
        pct=job.get("pct", 0),
        message=job.get("message", ""),
        version=job.get("version"),
    )


# ---------------------------------------------------------------------------
# Output retrieval
# ---------------------------------------------------------------------------
@router.get("/output/{sample_id}", response_model=OutputListResponse)
def list_outputs(sample_id: str):
    """List all generated output files for a sample."""
    _, out_dir, _, _ = _find_sample_dirs(sample_id)
    if out_dir is None or not out_dir.exists():
        raise HTTPException(status_code=404, detail=f"No outputs for '{sample_id}'")

    files = []
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            stat = f.stat()
            files.append(OutputFile(
                filename=f.name,
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            ))
    return OutputListResponse(sample_id=sample_id, files=files)


@router.get("/output/{sample_id}/note")
def get_output_note(sample_id: str, version: str = Query("latest")):
    """Retrieve the generated clinical note for a sample."""
    _, out_dir, _, _ = _find_sample_dirs(sample_id)
    if out_dir is None or not out_dir.exists():
        raise HTTPException(status_code=404, detail=f"No outputs for '{sample_id}'")

    if version == "latest":
        notes = sorted(out_dir.glob("generated_note_v*.md"), reverse=True)
        if not notes:
            raise HTTPException(status_code=404, detail="No generated notes found")
        note_path = notes[0]
    else:
        note_path = out_dir / f"generated_note_{version}.md"

    if not note_path.exists():
        raise HTTPException(status_code=404, detail=f"Note {version} not found")

    return {"content": note_path.read_text(), "version": version}


@router.get("/output/{sample_id}/transcript")
def get_output_transcript(sample_id: str, version: str = Query("latest")):
    """Retrieve the transcript for a sample."""
    _, out_dir, _, _ = _find_sample_dirs(sample_id)
    if out_dir is None or not out_dir.exists():
        raise HTTPException(status_code=404, detail=f"No outputs for '{sample_id}'")

    if version == "latest":
        transcripts = sorted(out_dir.glob("audio_transcript_v*.txt"), reverse=True)
        if not transcripts:
            raise HTTPException(status_code=404, detail="No transcripts found")
        t_path = transcripts[0]
    else:
        t_path = out_dir / f"audio_transcript_{version}.txt"

    if not t_path.exists():
        raise HTTPException(status_code=404, detail=f"Transcript {version} not found")

    return {"content": t_path.read_text(), "version": version}


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------
@router.post("/batch/upload", response_model=BatchUploadResponse)
async def batch_upload(
    files: list[UploadFile] = File(...),
    manifest: str = Form("[]"),
):
    """
    Upload a batch of encounter files.

    The manifest is a JSON array of BatchUploadItem objects describing each
    encounter. Files are matched to items by filename pattern:
    {sample_id}_audio.mp3 or {sample_id}_details.json.

    Skips files where local timestamp >= uploaded timestamp (overwrite_policy).
    """
    require_feature("batch_processing")

    data_root = _get_pipeline_data_dir()

    try:
        items = json.loads(manifest)
    except json.JSONDecodeError:
        items = []

    # Build lookup by sample_id
    item_map = {item["sample_id"]: item for item in items if "sample_id" in item}

    uploaded = 0
    skipped = 0
    errors = []

    for f in files:
        fname = f.filename or ""
        # Parse sample_id from filename: {sample_id}_audio.mp3 or {sample_id}_details.json
        parts = fname.rsplit("_", 1)
        if len(parts) < 2:
            errors.append(f"Cannot parse sample_id from filename: {fname}")
            continue

        sample_id = parts[0]
        item = item_map.get(sample_id, {})
        mode = item.get("mode", "dictation")
        provider_id = item.get("provider_id", "unknown")

        data_mode = "conversation" if mode == "ambient" else mode
        encounter_dir = data_root / data_mode / provider_id / sample_id
        encounter_dir.mkdir(parents=True, exist_ok=True)

        dest = encounter_dir / fname.split("/")[-1].replace(f"{sample_id}_", "")

        # Check overwrite policy
        if dest.exists() and cfg.sync.output_sync.overwrite_policy == "newer_only":
            skipped += 1
            continue

        content = await f.read()
        dest.write_bytes(content)
        uploaded += 1

    logger.info("batch_upload", uploaded=uploaded, skipped=skipped, errors=len(errors))
    return BatchUploadResponse(uploaded=uploaded, skipped=skipped, errors=errors)


@router.post("/batch/trigger")
async def batch_trigger(req: BatchTriggerRequest):
    """
    Trigger batch pipeline execution.

    If sample_ids is None, processes all samples that have audio but
    no generated output (or need re-processing).
    """
    require_feature("batch_processing")

    data_root = _get_pipeline_data_dir()
    output_root = _get_pipeline_output_dir()

    # Discover samples to process
    sample_ids = req.sample_ids
    if not sample_ids:
        sample_ids = []
        for mode_dir in ["dictation", "conversation"]:
            mode_path = data_root / mode_dir
            if not mode_path.exists():
                continue
            for physician_dir in mode_path.iterdir():
                if not physician_dir.is_dir():
                    continue
                for sample_dir in physician_dir.iterdir():
                    if sample_dir.is_dir():
                        sample_ids.append(sample_dir.name)

    batch_id = f"batch-{str(uuid.uuid4())[:8]}"
    jobs_created = []

    for sid in sample_ids:
        data_dir, out_dir, data_mode, physician = _find_sample_dirs(sid)
        if data_dir is None:
            continue

        # Find audio
        audio_path = None
        for candidate in ["dictation.mp3", "conversation_audio.mp3"]:
            p = data_dir / candidate
            if p.exists():
                audio_path = str(p)
                break
        if not audio_path:
            continue

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        job_id = f"job-{str(uuid.uuid4())[:8]}"
        mode = "ambient" if data_mode == "conversation" else "dictation"

        _jobs[job_id] = {
            "job_id": job_id,
            "sample_id": sid,
            "status": "pending",
            "stage": None,
            "pct": 0,
            "message": "Queued for batch processing",
            "mode": mode,
            "provider_id": physician,
            "data_dir": str(data_dir),
            "batch_id": batch_id,
        }
        jobs_created.append(job_id)

    # Start processing sequentially (to manage VRAM)
    if jobs_created:
        asyncio.create_task(_run_batch(jobs_created, req.version, req.two_pass))

    return {
        "batch_id": batch_id,
        "total_samples": len(jobs_created),
        "job_ids": jobs_created,
        "status": "processing",
    }


async def _run_batch(job_ids: list[str], version: str | None, two_pass: bool) -> None:
    """Process batch jobs sequentially to manage GPU VRAM."""
    for job_id in job_ids:
        job = _jobs.get(job_id)
        if not job:
            continue

        sample_id = job["sample_id"]
        data_dir = Path(job["data_dir"])
        mode = job["mode"]
        provider_id = job["provider_id"]

        # Find audio
        audio_path = None
        for candidate in ["dictation.mp3", "conversation_audio.mp3"]:
            p = data_dir / candidate
            if p.exists():
                audio_path = str(p)
                break
        if not audio_path:
            job["status"] = "error"
            job["message"] = "No audio file found"
            continue

        # Find output dir
        _, out_dir, data_mode, _ = _find_sample_dirs(sample_id)
        if out_dir is None:
            output_root = _get_pipeline_output_dir()
            data_mode = "conversation" if mode == "ambient" else "dictation"
            out_dir = output_root / data_mode / provider_id / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Detect version
        v = version
        if not v:
            existing = sorted(
                [int(f.stem.split("_v")[1]) for f in out_dir.glob("generated_note_v*.md")
                 if f.stem.split("_v")[1].isdigit()],
                reverse=True,
            )
            if existing:
                next_num = existing[0] + 1
            else:
                # Use global latest version + 1
                from api.data_loader import get_latest_version
                latest = get_latest_version()
                next_num = int(latest[1:]) + 1 if latest != "v1" else 1
            v = f"v{next_num}"

        note_audio_path = None
        if data_mode == "conversation":
            for candidate in ["notes.mp3", "note_audio.mp3"]:
                p = data_dir / candidate
                if p.exists():
                    note_audio_path = str(p)
                    break

        await _run_pipeline(
            job_id=job_id,
            sample_id=sample_id,
            audio_path=audio_path,
            note_audio_path=note_audio_path,
            mode=mode,
            provider_id=provider_id,
            visit_type="follow_up",
            output_dir=str(out_dir),
            version=v,
        )

    # After all samples complete, generate aggregate quality report
    final_version = version or v  # Use the version from last sample
    if final_version:
        try:
            from api.quality_runner import generate_aggregate_report
            logger.info("batch_quality_sweep_start", version=final_version)
            await asyncio.to_thread(generate_aggregate_report, final_version)
            logger.info("batch_quality_sweep_complete", version=final_version)
        except Exception as qe:
            logger.warning("batch_quality_sweep_failed", error=str(qe))


# ---------------------------------------------------------------------------
# Batch output retrieval
# ---------------------------------------------------------------------------
@router.get("/outputs/batch")
def batch_retrieve_outputs(
    sample_ids: str = Query(..., description="Comma-separated sample IDs"),
    since: str = Query("", description="ISO timestamp — only return files modified after this"),
):
    """
    Batch retrieval of output files for the provider-facing server.

    Returns a manifest of available files with their metadata. The provider
    server then fetches individual files via /pipeline/output/{sample_id}/note etc.
    """
    ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
    results = []

    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            pass

    for sid in ids:
        _, out_dir, _, _ = _find_sample_dirs(sid)
        if out_dir is None or not out_dir.exists():
            continue

        files = []
        for f in sorted(out_dir.iterdir()):
            if not f.is_file():
                continue
            stat = f.stat()
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            if since_dt and mod_time <= since_dt:
                continue
            files.append({
                "filename": f.name,
                "size": stat.st_size,
                "modified": mod_time.isoformat(),
            })

        if files:
            results.append({"sample_id": sid, "files": files})

    return {"samples": results}
