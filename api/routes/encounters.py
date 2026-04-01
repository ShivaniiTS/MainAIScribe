"""
api/routes/encounters.py

Encounter / sample routes. Reads from the output/ and data/ directories.
Supports creating new encounters with audio upload → pipeline trigger.

In provider-facing mode, pipeline operations (upload → trigger → poll → fetch)
are proxied to the remote processing-pipeline server. PHI stays local.
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import structlog
import yaml
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from api import data_loader as dl
from api.models import (
    SampleSummary,
    SampleDetail,
    NoteContent,
    ComparisonContent,
    GoldNoteContent,
    EncounterCreateRequest,
    EncounterResponse,
)
from api.proxy import needs_proxy

router = APIRouter(prefix="/encounters", tags=["encounters"])
logger = structlog.get_logger()

from config.paths import ROOT, DATA_DIR, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Sample listing (read-only, backed by output/ directory)
# ---------------------------------------------------------------------------

@router.get("", response_model=list[SampleSummary])
def list_encounters(
    mode: Optional[str] = Query(None, description="Filter by 'dictation' or 'ambient'"),
    version: Optional[str] = Query(None, description="Filter by version e.g. 'v4'"),
):
    """List all available samples / encounters."""
    samples = dl.list_samples()
    if mode:
        samples = [s for s in samples if s["mode"] == mode]
    if version:
        samples = [s for s in samples if version in s["versions"]]
    return [SampleSummary(**s) for s in samples]


@router.get("/{sample_id}", response_model=SampleDetail)
def get_encounter(sample_id: str):
    """Get metadata + patient context for a sample."""
    samples = dl.list_samples()
    match = next((s for s in samples if s["sample_id"] == sample_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
    return SampleDetail(
        **match,
        patient_context=dl.get_patient_context(sample_id),
    )


@router.get("/{sample_id}/note", response_model=NoteContent)
async def get_note(
    sample_id: str,
    version: str = Query(dl.LATEST_VERSION, description="Pipeline version"),
):
    """Get the generated clinical note (Markdown).

    Falls back to fetching from the pipeline server when the note isn't
    available locally (e.g. proxy fetch failed during processing).
    """
    content = dl.get_generated_note(sample_id, version)

    # Proxy fallback: if local note missing and we're provider-facing, try pipeline
    if content is None and needs_proxy():
        try:
            from api.proxy import proxy_get_note
            remote = await proxy_get_note(sample_id, version)
            content = remote.get("content")
            # Cache locally so subsequent requests don't need the proxy
            if content:
                out_dir = dl._output_dir_for(sample_id)
                if out_dir is None:
                    # Create the output dir based on encounter metadata
                    for enc in _encounters.values():
                        if enc.get("sample_id") == sample_id and enc.get("output_dir"):
                            out_dir = Path(enc["output_dir"])
                            out_dir.mkdir(parents=True, exist_ok=True)
                            break
                if out_dir:
                    resolved = dl.resolve_version(version)
                    (out_dir / f"generated_note_{resolved}.md").write_text(content)
                    logger.info("proxy_note_cached", sample_id=sample_id, version=version)
        except Exception as e:
            logger.warning("proxy_note_fallback_failed", sample_id=sample_id, error=str(e))

    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"Generated note not found for sample '{sample_id}' version '{version}'",
        )
    return NoteContent(sample_id=sample_id, version=version, content=content)


@router.get("/{sample_id}/comparison", response_model=ComparisonContent)
def get_comparison(
    sample_id: str,
    version: str = Query(dl.LATEST_VERSION, description="Pipeline version"),
):
    """Get the gold vs generated comparison (Markdown)."""
    content = dl.get_comparison(sample_id, version)
    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"Comparison not found for sample '{sample_id}' version '{version}'",
        )
    return ComparisonContent(sample_id=sample_id, version=version, content=content)


@router.get("/{sample_id}/versions")
def get_versions(sample_id: str):
    """Get all available versions for a sample."""
    out_dir = dl._output_dir_for(sample_id)
    if out_dir is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
    from api.data_loader import _discover_sample_versions
    versions = _discover_sample_versions(out_dir)
    return {"sample_id": sample_id, "versions": versions}


@router.get("/{sample_id}/gold", response_model=GoldNoteContent)
def get_gold_note(sample_id: str):
    """Get the gold-standard clinical note (Markdown)."""
    content = dl.get_gold_note(sample_id)
    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"Gold note not found for sample '{sample_id}'",
        )
    return GoldNoteContent(sample_id=sample_id, content=content)


@router.get("/{sample_id}/transcript")
async def get_transcript(
    sample_id: str,
    version: str = Query(dl.LATEST_VERSION),
):
    """Get the standalone transcript text for a sample.

    Falls back to fetching from the pipeline server when not available locally.
    """
    content = dl.get_transcript(sample_id, version)

    # Proxy fallback
    if content is None and needs_proxy():
        try:
            from api.proxy import proxy_get_transcript
            remote = await proxy_get_transcript(sample_id, version)
            content = remote.get("content")
            if content:
                out_dir = dl._output_dir_for(sample_id)
                if out_dir is None:
                    for enc in _encounters.values():
                        if enc.get("sample_id") == sample_id and enc.get("output_dir"):
                            out_dir = Path(enc["output_dir"])
                            out_dir.mkdir(parents=True, exist_ok=True)
                            break
                if out_dir:
                    resolved = dl.resolve_version(version)
                    (out_dir / f"audio_transcript_{resolved}.txt").write_text(content)
                    logger.info("proxy_transcript_cached", sample_id=sample_id, version=version)
        except Exception as e:
            logger.warning("proxy_transcript_fallback_failed", sample_id=sample_id, error=str(e))

    if content is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not found for sample '{sample_id}' version '{version}'",
        )
    transcript_versions = dl.list_transcript_versions(sample_id)
    return {
        "sample_id": sample_id,
        "version": version,
        "content": content,
        "versions": transcript_versions,
    }


@router.get("/{sample_id}/audio")
def get_audio(sample_id: str, type: Optional[str] = Query(None, description="Audio type: 'notes' for note dictation, omit for primary")):
    """Stream the raw audio file for a sample.

    Pass ``?type=notes`` to retrieve the note dictation audio (conversation mode).
    Omit or pass any other value for the primary audio.
    """
    from fastapi.responses import FileResponse

    if type == "notes":
        paths = dl.get_audio_paths(sample_id)
        note_path = paths.get("note_dictation")
        if note_path is None:
            raise HTTPException(status_code=404, detail=f"Notes audio not found for '{sample_id}'")
        return FileResponse(note_path, media_type="audio/mpeg", filename=f"{sample_id}_note_audio.mp3")

    audio_path = dl.get_audio_path(sample_id)
    if audio_path is None:
        raise HTTPException(status_code=404, detail=f"Audio not found for '{sample_id}'")
    return FileResponse(audio_path, media_type="audio/mpeg", filename=f"{sample_id}.mp3")


@router.get("/{sample_id}/quality")
def get_sample_quality(
    sample_id: str,
    version: str = Query(dl.LATEST_VERSION),
):
    """Get quality scores for a specific sample."""
    scores = dl._get_sample_scores(sample_id, version)
    if scores is None:
        raise HTTPException(
            status_code=404,
            detail=f"Quality scores not found for sample '{sample_id}' version '{version}'",
        )
    return {"sample_id": sample_id, "version": version, **scores}


# ---------------------------------------------------------------------------
# Create & upload (triggers pipeline run)
# ---------------------------------------------------------------------------

# In-memory store for live encounters — replace with DB in production
_encounters: dict[str, dict] = {}


def _load_patient_from_roster(patient_id: str) -> dict | None:
    """Look up a patient by ID from the stub EHR roster."""
    from config.paths import CONFIG_DIR
    roster_path = CONFIG_DIR / "ehr_stub" / "patient_roster.json"
    if not roster_path.exists():
        return None
    data = json.loads(roster_path.read_text())
    for p in data.get("patients", []):
        if p["id"] == patient_id:
            return p
    return None


@router.post("", response_model=EncounterResponse, status_code=201)
def create_encounter(req: EncounterCreateRequest):
    """Create a new encounter (returns encounter_id for polling)."""
    from config.provider_manager import get_provider_manager

    canonical_provider_id = get_provider_manager().resolve_provider_id(req.provider_id)
    encounter_id = req.encounter_id or str(uuid.uuid4())[:8]
    enc = {
        "encounter_id": encounter_id,
        "status": "pending",
        "provider_id": canonical_provider_id,
        "patient_id": req.patient_id,
        "patient_name": req.patient_name,
        "visit_type": req.visit_type,
        "mode": req.mode,
        "message": "Waiting for audio upload",
    }
    _encounters[encounter_id] = enc
    return EncounterResponse(**enc)


@router.post("/{encounter_id}/upload")
async def upload_audio(
    encounter_id: str,
    audio: UploadFile = File(...),
    note_audio: Optional[UploadFile] = File(None),
):
    """
    Upload audio and trigger the encounter pipeline.

    Creates the encounter folder in ai-scribe-data/, generates
    patient_context.yaml + encounter metadata, saves the audio file,
    then launches the pipeline asynchronously with WebSocket progress events.
    """
    enc = _encounters.get(encounter_id)
    if not enc:
        raise HTTPException(status_code=404, detail=f"Encounter '{encounter_id}' not found")

    provider_id = enc["provider_id"]
    patient_id = enc["patient_id"]
    visit_type = enc["visit_type"]
    mode = enc["mode"]
    today = date.today().isoformat()

    # Look up patient name: prefer name from create request, fall back to roster
    patient_name = "unknown"
    patient = None
    if enc.get("patient_name"):
        # Name provided by MDCO bridge (e.g. "BHALALA, NIRAV")
        raw = enc["patient_name"]
        if "," in raw:
            parts = [p.strip() for p in raw.split(",", 1)]
            patient_name = f"{parts[1]}_{parts[0]}".lower()
        else:
            patient_name = raw.replace(" ", "_").lower()
        # Sanitize for filesystem
        patient_name = re.sub(r"[^a-z0-9_]+", "_", patient_name).strip("_")
    else:
        patient = _load_patient_from_roster(patient_id)
        if patient:
            patient_name = f"{patient['first_name']}_{patient['last_name']}".lower()

    # Build folder name: {patient_name}_{encounter_id}_{date}
    folder_name = f"{patient_name}_{encounter_id}_{today}"
    data_mode = "conversation" if mode == "ambient" else "dictation"

    # Create data folder: ai-scribe-data/{mode}/{provider_id}/{folder_name}/
    encounter_dir = DATA_DIR / data_mode / provider_id / folder_name
    encounter_dir.mkdir(parents=True, exist_ok=True)

    # Create matching output folder
    output_encounter_dir = OUTPUT_DIR / data_mode / provider_id / folder_name
    output_encounter_dir.mkdir(parents=True, exist_ok=True)

    # Save audio file
    audio_filename = "dictation.mp3" if mode == "dictation" else "conversation_audio.mp3"
    audio_path = encounter_dir / audio_filename
    content = await audio.read()
    audio_path.write_bytes(content)

    # Save note dictation audio (conversation mode dual-audio)
    note_audio_bytes: bytes | None = None
    note_audio_filename: str | None = None
    note_audio_path_str: str | None = None
    if note_audio and note_audio.filename:
        note_audio_filename = "note_audio.mp3"
        note_audio_save_path = encounter_dir / note_audio_filename
        note_audio_bytes = await note_audio.read()
        note_audio_save_path.write_bytes(note_audio_bytes)
        note_audio_path_str = str(note_audio_save_path)

    # Generate patient_context.yaml (same format as batch data)
    # Derive patient display name from provided name or roster
    if enc.get("patient_name"):
        raw = enc["patient_name"]
        if "," in raw:
            parts = [p.strip() for p in raw.split(",", 1)]
            display_name = f"{parts[1]} {parts[0]}"  # "Firstname Lastname"
            first_name = parts[1] if len(parts) > 1 else ""
            last_name = parts[0]
        else:
            display_name = raw
            name_parts = raw.strip().split()
            first_name = name_parts[0] if name_parts else ""
            last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
    elif patient:
        display_name = f"{patient['first_name']} {patient['last_name']}"
        first_name = patient["first_name"]
        last_name = patient["last_name"]
    else:
        display_name = "Unknown"
        first_name = "Unknown"
        last_name = "Unknown"

    patient_context = {
        "patient": {
            "name": display_name,
            "date_of_birth": patient.get("date_of_birth", "") if patient else "",
            "sex": patient.get("sex", "") if patient else "",
            "mrn": patient.get("mrn", "") if patient else "",
        },
        "encounter": {
            "date_of_service": today,
            "visit_type": visit_type,
        },
        "provider": {
            "name": provider_id,
            "credentials": "",
            "specialty": "",
        },
        "facility": {
            "name": "Talisman Solutions",
            "location": "",
        },
    }

    # Enrich provider context from provider profile if available
    try:
        from config.provider_manager import get_provider_manager
        mgr = get_provider_manager()
        profile = mgr.load_or_default(provider_id)
        patient_context["provider"]["name"] = profile.name or provider_id
        patient_context["provider"]["credentials"] = profile.credentials or ""
        patient_context["provider"]["specialty"] = profile.specialty or ""
    except Exception:
        pass

    context_path = encounter_dir / "patient_context.yaml"
    context_path.write_text(yaml.dump(patient_context, default_flow_style=False))

    # Generate patient_demographics.json
    demographics = {
        "patient_id": patient_id,
        "first_name": first_name,
        "last_name": last_name,
        "date_of_birth": patient.get("date_of_birth", "") if patient else "",
        "sex": patient.get("sex", "") if patient else "",
        "mrn": patient.get("mrn", "") if patient else "",
    }
    (encounter_dir / "patient_demographics.json").write_text(
        json.dumps(demographics, indent=2)
    )

    # Generate encounter_details.json
    encounter_details = {
        "encounter_id": encounter_id,
        "provider_id": provider_id,
        "patient_id": patient_id,
        "visit_type": visit_type,
        "mode": mode,
        "date_of_service": today,
        "created_at": datetime.utcnow().isoformat(),
        "audio_file": audio_filename,
        "note_audio_file": note_audio_filename,
        "has_gold_standard": False,
    }
    (encounter_dir / "encounter_details.json").write_text(
        json.dumps(encounter_details, indent=2)
    )

    # Update in-memory state
    enc["status"] = "processing"
    enc["message"] = "Audio received — pipeline running"
    enc["sample_id"] = folder_name
    enc["data_dir"] = str(encounter_dir)
    enc["output_dir"] = str(output_encounter_dir)

    if needs_proxy():
        # Provider-facing mode: send audio + encounter_details to pipeline server
        # PHI (patient_demographics.json, patient_context.yaml) stays local
        asyncio.create_task(
            _proxy_pipeline_run(
                encounter_id=encounter_id,
                sample_id=folder_name,
                audio_bytes=content,
                audio_filename=audio_filename,
                note_audio_bytes=note_audio_bytes,
                note_audio_filename=note_audio_filename,
                mode=mode,
                provider_id=provider_id,
                visit_type=visit_type,
                encounter_details=encounter_details,
                output_dir=str(output_encounter_dir),
            )
        )
    else:
        # processing-pipeline: run pipeline directly
        asyncio.create_task(
            _run_pipeline_async(
                encounter_id=encounter_id,
                sample_id=folder_name,
                audio_path=str(audio_path),
                note_audio_path=note_audio_path_str,
                mode=mode,
                provider_id=provider_id,
                visit_type=visit_type,
                output_dir=str(output_encounter_dir),
                data_dir=str(encounter_dir),
            )
        )

    return {
        "encounter_id": encounter_id,
        "sample_id": folder_name,
        "status": "processing",
        "message": "Audio received — pipeline running",
    }


async def _proxy_pipeline_run(
    encounter_id: str,
    sample_id: str,
    audio_bytes: bytes,
    audio_filename: str,
    mode: str,
    provider_id: str,
    visit_type: str,
    encounter_details: dict,
    output_dir: str,
    note_audio_bytes: bytes | None = None,
    note_audio_filename: str | None = None,
) -> None:
    """
    Proxy pipeline execution to the remote processing-pipeline server.

    Flow: upload audio → trigger → poll until complete → fetch note+transcript.
    PHI stays on the provider-facing server; only audio + encounter_details are sent.
    """
    from api.proxy import proxy_upload, proxy_trigger, proxy_status
    from api.proxy import proxy_get_note, proxy_get_transcript
    from api.ws.session_events import manager

    enc = _encounters.get(encounter_id, {})

    try:
        await asyncio.sleep(0.5)
        await manager.send_progress(encounter_id, "init", 5, "Uploading to pipeline server...")

        # 1. Upload audio + de-identified metadata to pipeline server
        upload_result = await proxy_upload(
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
            note_audio_bytes=note_audio_bytes,
            note_audio_filename=note_audio_filename,
            sample_id=sample_id,
            mode=mode,
            provider_id=provider_id,
            encounter_details=encounter_details,
        )
        job_id = upload_result["job_id"]
        enc["pipeline_job_id"] = job_id

        await manager.send_progress(encounter_id, "init", 10, "Triggering pipeline...")

        # 2. Trigger pipeline execution
        await proxy_trigger(
            job_id=job_id,
            mode=mode,
            provider_id=provider_id,
            visit_type=visit_type,
        )

        await manager.send_progress(encounter_id, "transcribe", 20, "Pipeline running on GPU server...")

        # 3. Poll until complete (max ~10 minutes)
        max_polls = 120
        consecutive_errors = 0
        for i in range(max_polls):
            await asyncio.sleep(5)
            try:
                status = await proxy_status(job_id)
                consecutive_errors = 0  # reset on success
            except Exception as poll_exc:
                consecutive_errors += 1
                logger.warning(
                    "proxy_poll_error",
                    encounter_id=encounter_id,
                    job_id=job_id,
                    attempt=i,
                    consecutive_errors=consecutive_errors,
                    error=f"{type(poll_exc).__name__}: {poll_exc}",
                )
                if consecutive_errors >= 5:
                    raise RuntimeError(
                        f"Pipeline status polling failed {consecutive_errors} times in a row: {poll_exc}"
                    )
                continue

            remote_status = status.get("status", "")
            pct = status.get("pct", 0)
            stage = status.get("stage", "")
            message = status.get("message", "")

            await manager.send_progress(encounter_id, stage, pct, message)

            if remote_status == "complete":
                break
            elif remote_status == "error":
                raise RuntimeError(f"Pipeline error: {message}")
        else:
            raise RuntimeError("Pipeline timed out after 10 minutes")

        await manager.send_progress(encounter_id, "delivery", 90, "Fetching results...")

        # 4. Fetch generated note and transcript back to local output/
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        version = status.get("version", "latest")

        try:
            note_data = await proxy_get_note(sample_id, version)
            note_content = note_data.get("content", "")
            if note_content:
                (out_path / f"generated_note_{version}.md").write_text(note_content)
                logger.info("proxy_note_fetched", sample_id=sample_id, version=version)
        except Exception as e:
            logger.warning("proxy_note_fetch_failed", error=str(e))

        try:
            transcript_data = await proxy_get_transcript(sample_id, version)
            transcript_content = transcript_data.get("content", "")
            if transcript_content:
                (out_path / f"audio_transcript_{version}.txt").write_text(transcript_content)
                logger.info("proxy_transcript_fetched", sample_id=sample_id, version=version)
        except Exception as e:
            logger.warning("proxy_transcript_fetch_failed", error=str(e))

        enc["status"] = "complete"
        enc["message"] = f"Pipeline complete — {version} generated"
        enc["version"] = version

        await manager.send_complete(encounter_id, sample_id)
        logger.info("proxy_pipeline_complete", encounter_id=encounter_id, sample_id=sample_id)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__} (no message)"
        logger.error("proxy_pipeline_error", encounter_id=encounter_id, error=error_msg, exc_info=True)
        enc["status"] = "error"
        enc["message"] = f"Pipeline error: {error_msg}"
        await manager.send_error(encounter_id, error_msg)


async def _run_pipeline_async(
    encounter_id: str,
    sample_id: str,
    audio_path: str,
    mode: str,
    provider_id: str,
    visit_type: str,
    output_dir: str,
    data_dir: str,
    note_audio_path: str | None = None,
) -> None:
    """Run the encounter pipeline in the background, sending WebSocket events."""
    from api.ws.session_events import manager

    enc = _encounters.get(encounter_id, {})

    try:
        # Give the client a moment to connect its WebSocket before we send events.
        await asyncio.sleep(1.0)

        await manager.send_progress(encounter_id, "init", 5, "Loading provider profile...")

        # Load provider profile
        from config.provider_manager import get_provider_manager
        mgr = get_provider_manager()
        profile = mgr.load_or_default(provider_id)

        await manager.send_progress(encounter_id, "init", 10, "Building pipeline...")

        # Build encounter state
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

        await manager.send_progress(encounter_id, "transcribe", 20, "Starting ASR transcription...")

        # Run the pipeline synchronously in a thread to avoid blocking the event loop
        from orchestrator.graph import build_graph, run_encounter
        graph = build_graph()

        final = await asyncio.to_thread(run_encounter, graph, state)

        # Free ASR model from GPU memory before quality evaluation
        try:
            from mcp_servers.registry import get_registry
            get_registry().unload_engine("asr")
        except Exception:
            pass

        await manager.send_progress(encounter_id, "transcribe", 50, "Transcription complete")

        # Save outputs
        out_path = Path(output_dir)

        # Detect next version dynamically
        from api.data_loader import _discover_sample_versions, get_latest_version
        existing = _discover_sample_versions(out_path)
        if existing:
            next_num = max(int(v[1:]) for v in existing) + 1
        else:
            latest = get_latest_version()
            next_num = int(latest[1:]) + 1 if latest != "v1" else 1
        pipeline_version = f"v{next_num}"

        # Save generated note
        if final.final_note:
            from output.markdown_writer import write_clinical_note
            write_clinical_note(
                final,
                path=out_path / f"generated_note_{pipeline_version}.md",
                version=pipeline_version,
                sample_id=sample_id,
            )
            await manager.send_progress(encounter_id, "note", 80, "Clinical note generated")

        # Save standalone transcript
        if final.transcript and final.transcript.full_text.strip():
            transcript_path = out_path / f"audio_transcript_{pipeline_version}.txt"
            transcript_path.write_text(final.transcript.full_text.strip())
            await manager.send_progress(encounter_id, "note", 85, "Transcript saved")

        # Run quality evaluation if gold standard exists
        await manager.send_progress(encounter_id, "quality", 90, "Running quality evaluation...")
        try:
            from api.quality_runner import evaluate_sample
            gold = dl.get_gold_note(sample_id)
            gen_text = final.final_note if isinstance(final.final_note, str) else (out_path / f"generated_note_{pipeline_version}.md").read_text() if (out_path / f"generated_note_{pipeline_version}.md").exists() else ""
            tx_text = final.transcript.full_text.strip() if final.transcript else ""
            if gold:
                await asyncio.to_thread(
                    evaluate_sample,
                    sample_id=sample_id,
                    generated_note=gen_text,
                    gold_note=gold,
                    transcript=tx_text,
                    version=pipeline_version,
                    output_dir=out_path,
                )
                await manager.send_progress(encounter_id, "quality", 95, "Quality evaluation complete")
        except Exception as qe:
            logger.warning("quality_eval_skipped", encounter_id=encounter_id, error=str(qe))

        await manager.send_progress(encounter_id, "delivery", 98, "Finalizing...")

        # Update in-memory state
        enc["status"] = "complete"
        enc["message"] = f"Pipeline complete — {pipeline_version} generated"
        enc["version"] = pipeline_version

        await manager.send_complete(encounter_id, sample_id)

        logger.info(
            "pipeline_complete",
            encounter_id=encounter_id,
            sample_id=sample_id,
        )

    except Exception as e:
        logger.error(
            "pipeline_error",
            encounter_id=encounter_id,
            error=str(e),
        )
        enc["status"] = "error"
        enc["message"] = f"Pipeline error: {str(e)}"
        await manager.send_error(encounter_id, str(e))


@router.post("/{sample_id}/rerun")
async def rerun_pipeline(sample_id: str):
    """
    Re-run the pipeline on an existing sample, auto-incrementing the version.

    Finds the sample's audio file, detects the next version number,
    and launches the pipeline asynchronously.
    """
    # Find the sample in data directories
    result = dl._data_dir_for(sample_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found in data")

    data_mode, physician, data_dir = result

    # Find audio file
    audio_path = None
    for candidate in ["dictation.mp3", "conversation_audio.mp3", "note_audio.mp3", "conversation.mp3"]:
        p = data_dir / candidate
        if p.exists():
            audio_path = str(p)
            break
    if not audio_path:
        raise HTTPException(status_code=404, detail=f"No audio file found for sample '{sample_id}'")

    # Determine the output directory
    out_dir = dl._output_dir_for(sample_id)
    if out_dir is None:
        # Create it mirroring the data layout
        out_dir = OUTPUT_DIR / data_mode / physician / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)

    # Detect next version number
    existing_versions = sorted(
        [int(f.stem.split("_v")[1]) for f in out_dir.glob("generated_note_v*.md")
         if f.stem.split("_v")[1].isdigit()],
        reverse=True,
    )
    if existing_versions:
        next_version_num = existing_versions[0] + 1
    else:
        # No existing output — use global latest + 1
        latest = dl.get_latest_version()
        next_version_num = int(latest[1:]) + 1 if latest != "v1" else 1
    next_version = f"v{next_version_num}"

    # Determine mode
    mode = "ambient" if data_mode == "conversation" else "dictation"

    # Create a tracking entry
    encounter_id = f"rerun-{str(uuid.uuid4())[:8]}"
    _encounters[encounter_id] = {
        "encounter_id": encounter_id,
        "status": "processing",
        "provider_id": physician,
        "patient_id": f"patient-{sample_id}",
        "visit_type": "",
        "mode": mode,
        "message": f"Re-running pipeline → {next_version}",
        "sample_id": sample_id,
        "version": next_version,
    }

    if needs_proxy():
        # Provider-facing: send audio to pipeline server for re-processing
        audio_bytes = Path(audio_path).read_bytes()
        audio_filename = Path(audio_path).name
        encounter_details = {}
        details_path = data_dir / "encounter_details.json"
        if details_path.exists():
            encounter_details = json.loads(details_path.read_text())

        # Check for note audio to forward
        rerun_note_audio_bytes: bytes | None = None
        rerun_note_audio_filename: str | None = None
        if data_mode == "conversation":
            for candidate in ["note_audio.mp3"]:
                p = data_dir / candidate
                if p.exists():
                    rerun_note_audio_bytes = p.read_bytes()
                    rerun_note_audio_filename = candidate
                    break

        asyncio.create_task(
            _proxy_pipeline_run(
                encounter_id=encounter_id,
                sample_id=sample_id,
                audio_bytes=audio_bytes,
                audio_filename=audio_filename,
                note_audio_bytes=rerun_note_audio_bytes,
                note_audio_filename=rerun_note_audio_filename,
                mode=mode,
                provider_id=physician,
                visit_type=encounter_details.get("visit_type", "follow_up"),
                encounter_details=encounter_details,
                output_dir=str(out_dir),
            )
        )
    else:
        # Local mode: run pipeline directly
        note_audio_path = None
        if data_mode == "conversation":
            for candidate in ["note_audio.mp3"]:
                p = data_dir / candidate
                if p.exists():
                    note_audio_path = str(p)
                    break

        asyncio.create_task(
            _run_rerun_pipeline(
                encounter_id=encounter_id,
                sample_id=sample_id,
                audio_path=audio_path,
                note_audio_path=note_audio_path,
                mode=mode,
                provider_id=physician,
                output_dir=str(out_dir),
                version=next_version,
            )
        )

    return {
        "encounter_id": encounter_id,
        "sample_id": sample_id,
        "version": next_version,
        "status": "processing",
        "message": f"Pipeline re-run started → {next_version}",
    }


async def _run_rerun_pipeline(
    encounter_id: str,
    sample_id: str,
    audio_path: str,
    note_audio_path: str | None,
    mode: str,
    provider_id: str,
    output_dir: str,
    version: str,
) -> None:
    """Re-run pipeline on existing sample, saving output with the given version."""
    from api.ws.session_events import manager

    enc = _encounters.get(encounter_id, {})

    try:
        # Give the client a moment to connect its WebSocket before we send events.
        # The rerun POST returns the encounter_id; the client opens a WS with it.
        # Without this pause, early progress events are lost because no WS is connected yet.
        await asyncio.sleep(1.0)

        await manager.send_progress(encounter_id, "init", 5, "Loading provider profile...")

        from config.provider_manager import get_provider_manager
        mgr = get_provider_manager()
        profile = mgr.load_or_default(provider_id)

        await manager.send_progress(encounter_id, "init", 10, "Building pipeline...")

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

        await manager.send_progress(encounter_id, "transcribe", 20, "Starting ASR transcription...")

        from orchestrator.graph import build_graph, run_encounter
        graph = build_graph()

        final = await asyncio.to_thread(run_encounter, graph, state)

        # Free ASR model from GPU memory before quality evaluation
        try:
            from mcp_servers.registry import get_registry
            get_registry().unload_engine("asr")
        except Exception:
            pass

        await manager.send_progress(encounter_id, "transcribe", 50, "Transcription complete")

        out_path = Path(output_dir)

        if final.final_note:
            from output.markdown_writer import write_clinical_note
            write_clinical_note(
                final,
                path=out_path / f"generated_note_{version}.md",
                version=version,
                sample_id=sample_id,
            )
            await manager.send_progress(encounter_id, "note", 80, "Clinical note generated")

        if final.transcript and final.transcript.full_text.strip():
            transcript_path = out_path / f"audio_transcript_{version}.txt"
            transcript_path.write_text(final.transcript.full_text.strip())
            await manager.send_progress(encounter_id, "note", 85, "Transcript saved")

        # Run quality evaluation if gold standard exists
        await manager.send_progress(encounter_id, "quality", 90, "Running quality evaluation...")
        try:
            from api.quality_runner import evaluate_sample
            gold = dl.get_gold_note(sample_id)
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
                await manager.send_progress(encounter_id, "quality", 95, "Quality evaluation complete")
        except Exception as qe:
            logger.warning("quality_eval_skipped", encounter_id=encounter_id, error=str(qe))

        await manager.send_progress(encounter_id, "delivery", 98, "Finalizing...")

        enc["status"] = "complete"
        enc["message"] = f"Pipeline complete — {version} generated"

        await manager.send_complete(encounter_id, sample_id)

        logger.info("rerun_pipeline_complete", encounter_id=encounter_id,
                     sample_id=sample_id, version=version)

    except Exception as e:
        logger.error("rerun_pipeline_error", encounter_id=encounter_id, error=str(e))
        enc["status"] = "error"
        enc["message"] = f"Pipeline error: {str(e)}"
        await manager.send_error(encounter_id, str(e))


@router.get("/{encounter_id}/status")
async def get_encounter_status(encounter_id: str):
    """Poll pipeline status for a created encounter."""
    enc = _encounters.get(encounter_id)
    if not enc:
        raise HTTPException(status_code=404, detail=f"Encounter '{encounter_id}' not found")

    # If proxied, fetch live status from pipeline server
    if needs_proxy() and enc.get("pipeline_job_id"):
        try:
            from api.proxy import proxy_status
            remote = await proxy_status(enc["pipeline_job_id"])
            enc["status"] = remote.get("status", enc["status"])
            enc["message"] = remote.get("message", enc["message"])
        except Exception:
            pass  # Return cached local state on proxy failure

    return enc
