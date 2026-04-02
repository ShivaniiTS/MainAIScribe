"""
api/ws/audio_stream.py — WebSocket endpoint for real-time audio streaming + ASR.

Protocol:
  Client → Server: binary audio chunks (WebM/Opus from MediaRecorder, or raw PCM)
  Server → Client: JSON messages
    { "type": "partial",  "text": "...", "is_final": false, "confidence": 0.9 }
    { "type": "final",    "text": "...", "is_final": true, "speaker": "SPEAKER_00", "start_ms": 1200, "end_ms": 3400 }
    { "type": "complete", "transcript": "Full accumulated transcript..." }
    { "type": "error",    "message": "..." }

Audio conversion: if the client sends WebM/Opus (browser MediaRecorder default),
the server pipes it through FFmpeg to convert to 16kHz mono PCM in real-time.

This endpoint runs on the pipeline server only (requires GPU for NeMo).
The provider-facing server proxies WebSocket connections to the pipeline server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from config.deployment import get_deployment_config, require_feature

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Active streaming ASR sessions
_active_sessions: dict[str, dict] = {}

# Preload state
_preload_task: Optional[asyncio.Task] = None


@router.post("/asr/preload", tags=["asr"])
async def preload_streaming_model(mode: str = "dictation"):
    """
    Preload the streaming ASR model into GPU memory.

    Call this when the user navigates to the Capture page so the model
    is warm by the time they start recording (~8s load time avoided).

    Returns immediately — loading happens in background.
    """
    global _preload_task

    engine = _get_streaming_engine(mode)
    if engine is None:
        return {"status": "unavailable", "message": "No streaming ASR engine configured"}

    if hasattr(engine, '_loaded') and engine._loaded:
        return {"status": "ready", "message": "Model already loaded"}

    # Start background load if not already running
    if _preload_task and not _preload_task.done():
        return {"status": "loading", "message": "Model load already in progress"}

    async def _load():
        logger.info("asr_preload: loading streaming model for mode=%s", mode)
        await asyncio.to_thread(engine._ensure_model)
        logger.info("asr_preload: model ready")

    _preload_task = asyncio.create_task(_load())
    return {"status": "loading", "message": "Model loading in background (~8s)"}


@router.get("/asr/status", tags=["asr"])
async def streaming_model_status():
    """Check if the streaming ASR model is loaded and ready."""
    for mode in ("dictation", "ambient"):
        engine = _get_streaming_engine(mode)
        if engine and hasattr(engine, '_loaded') and engine._loaded:
            if engine._model is not None:
                return {"status": "ready", "engine": engine.name, "model": engine.model_name}
            # Loaded flag set but model is None → load failed
            error_detail = getattr(engine, '_load_error', 'unknown error')
            return {"status": "failed", "message": f"Model load failed: {error_detail}"}

    if _preload_task and not _preload_task.done():
        return {"status": "loading"}

    return {"status": "not_loaded"}


def _get_streaming_engine(mode: str = "dictation"):
    """Get the appropriate streaming ASR engine for the recording mode."""
    from mcp_servers.registry import get_registry
    registry = get_registry()

    if mode == "ambient":
        # Try multitalker first for multi-speaker
        try:
            engine = registry.get("asr", "nemo_multitalker")
            return engine
        except (KeyError, ImportError):
            pass

    # Default: single-speaker NeMo streaming
    try:
        engine = registry.get("asr", "nemo_streaming")
        return engine
    except (KeyError, ImportError):
        logger.warning("audio_stream: no streaming ASR engine available")
        return None


class AudioConverter:
    """
    Converts streaming WebM/Opus audio to 16kHz mono PCM via FFmpeg pipe.

    Receives WebM chunks on stdin, outputs raw PCM on stdout.
    Runs as a subprocess for zero-copy streaming conversion.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None

    async def start(self) -> None:
        """Start the FFmpeg conversion subprocess."""
        self._process = await asyncio.to_thread(self._start_ffmpeg)

    def _start_ffmpeg(self) -> subprocess.Popen:
        return subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",           # Read from stdin
                "-f", "s16le",            # Output raw PCM
                "-acodec", "pcm_s16le",   # 16-bit signed
                "-ar", "16000",           # 16 kHz
                "-ac", "1",               # Mono
                "pipe:1",                 # Write to stdout
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    async def write(self, chunk: bytes) -> None:
        """Write a WebM chunk to FFmpeg's stdin."""
        if self._process and self._process.stdin:
            try:
                await asyncio.to_thread(self._process.stdin.write, chunk)
                await asyncio.to_thread(self._process.stdin.flush)
            except (BrokenPipeError, OSError):
                logger.warning("audio_stream: FFmpeg pipe broken")

    async def read(self, num_bytes: int) -> bytes:
        """Read converted PCM bytes from FFmpeg's stdout."""
        if self._process and self._process.stdout:
            try:
                return await asyncio.to_thread(self._process.stdout.read, num_bytes)
            except (BrokenPipeError, OSError):
                return b""
        return b""

    async def close(self) -> None:
        """Terminate the FFmpeg process."""
        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.terminate()
                await asyncio.to_thread(self._process.wait, timeout=5)
            except Exception:
                self._process.kill()
            self._process = None


@router.websocket("/ws/asr/{encounter_id}")
async def asr_stream(
    encounter_id: str,
    websocket: WebSocket,
    mode: str = Query("dictation"),
    format: str = Query("webm"),
):
    """
    WebSocket for real-time audio streaming + ASR transcription.

    Query params:
        mode: "dictation" (single speaker) or "ambient" (multi-speaker)
        format: "webm" (browser default) or "pcm" (raw 16kHz mono 16-bit)

    Client sends binary audio frames. Server responds with JSON transcript events.
    """
    await websocket.accept()

    engine = _get_streaming_engine(mode)
    if engine is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "No streaming ASR engine available. Install NeMo or check engines.yaml.",
        }))
        await websocket.close()
        return

    # Set up audio format conversion if needed
    converter: Optional[AudioConverter] = None
    needs_conversion = (format != "pcm")

    if needs_conversion:
        converter = AudioConverter()
        try:
            await converter.start()
        except Exception as exc:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Failed to start audio converter: {exc}",
            }))
            await websocket.close()
            return

    # Track session
    session_id = encounter_id
    from mcp_servers.asr.base import ASRConfig
    asr_config = ASRConfig(
        language="en",
        diarize=(mode == "ambient"),
    )

    # Inject patient and provider names as hotwords so they are always spelled
    # correctly regardless of NeMo's casing output.
    # get_patient_context is best-effort — a missing encounter silently skips.
    try:
        from api.data_loader import get_patient_context
        ctx = get_patient_context(encounter_id)
        if ctx:
            patient_name: str = (ctx.get("patient") or {}).get("name") or ""
            provider_name: str = (ctx.get("provider") or {}).get("name") or ""
            for name in (patient_name, provider_name):
                name = name.strip()
                if not name:
                    continue
                asr_config.hotwords.append(name)
                # Also add just the last name (first space-delimited final token)
                # so "Pello" alone is corrected even without the full "Scott J. Pello"
                parts = name.split()
                if len(parts) > 1:
                    asr_config.hotwords.append(parts[-1])
    except Exception:
        pass  # name injection is best-effort; never fail the session

    _active_sessions[session_id] = {
        "encounter_id": encounter_id,
        "mode": mode,
        "started_at": time.time(),
        "chunks_received": 0,
    }

    await websocket.send_text(json.dumps({
        "type": "connected",
        "encounter_id": encounter_id,
        "mode": mode,
        "engine": engine.name,
    }))

    logger.info("asr_stream: session started — encounter=%s mode=%s engine=%s",
                encounter_id, mode, engine.name)

    # Producer/consumer pattern: decouple audio receiving from ASR processing
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=100)
    accumulated_transcript = ""
    client_done = asyncio.Event()

    async def receive_audio():
        """Producer: receive audio chunks from WebSocket, put into queue.

        Keeps receiving until the client sends a close frame or disconnects.
        Uses a long timeout so slow clients don't get dropped.
        """
        try:
            while True:
                # Use websocket.receive() to handle both bytes and close frames
                message = await asyncio.wait_for(websocket.receive(), timeout=120.0)

                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"]:
                    data = message["bytes"]
                    if needs_conversion and converter:
                        await converter.write(data)
                        pcm_data = await converter.read(engine.chunk_bytes * 2)
                        if pcm_data:
                            await audio_queue.put(pcm_data)
                    else:
                        await audio_queue.put(data)
                # Ignore text messages (pings from client)

        except WebSocketDisconnect:
            pass
        except asyncio.TimeoutError:
            logger.info("asr_stream: receive timeout (120s idle) — closing")
        except Exception as exc:
            logger.warning("asr_stream: receive error — %s", exc)
        finally:
            await audio_queue.put(None)  # Sentinel: no more audio
            client_done.set()

    async def process_audio():
        """Consumer: pull audio from queue, run ASR, send results back.

        Processes ALL queued audio even after the client stops sending.
        This ensures the model has time to transcribe buffered chunks.
        """
        nonlocal accumulated_transcript
        while True:
            try:
                pcm_data = await asyncio.wait_for(audio_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # No new audio for 5s — if client is done, stop processing
                if client_done.is_set() and audio_queue.empty():
                    break
                continue

            if pcm_data is None:
                # Client done — but process any remaining buffered audio in the engine
                # Feed a dummy empty chunk to flush the engine's internal buffer
                try:
                    async for partial in engine.transcribe_stream(b"", session_id, asr_config):
                        if partial.text.strip():
                            accumulated_transcript += partial.text + " "
                            try:
                                await websocket.send_text(json.dumps({
                                    "type": "final", "text": partial.text,
                                    "is_final": True, "speaker": partial.speaker,
                                    "start_ms": partial.start_ms, "end_ms": partial.end_ms,
                                    "confidence": partial.confidence,
                                }))
                            except Exception:
                                pass
                except Exception:
                    pass
                break

            try:
                async for partial in engine.transcribe_stream(pcm_data, session_id, asr_config):
                    event = {
                        "type": "final" if partial.is_final else "partial",
                        "text": partial.text,
                        "is_final": partial.is_final,
                        "speaker": partial.speaker,
                        "start_ms": partial.start_ms,
                        "end_ms": partial.end_ms,
                        "confidence": partial.confidence,
                    }
                    try:
                        await websocket.send_text(json.dumps(event))
                    except Exception:
                        return  # WebSocket closed

                    if partial.is_final:
                        accumulated_transcript += partial.text + " "
            except Exception as exc:
                logger.warning("asr_stream: ASR error — %s", exc)

    try:
        # Run producer and consumer concurrently
        await asyncio.gather(receive_audio(), process_audio())
    except Exception as exc:
        logger.error("asr_stream: error — %s", exc)
    finally:
        # Finalize session
        raw_transcript = engine.finalize_session(session_id)

        # Send complete transcript to client
        try:
            final_text = accumulated_transcript.strip()
            if raw_transcript and raw_transcript.segments:
                final_text = " ".join(s.text for s in raw_transcript.segments)

            await websocket.send_text(json.dumps({
                "type": "complete",
                "transcript": final_text,
                "segments": [
                    {
                        "text": s.text,
                        "speaker": s.speaker,
                        "start_ms": s.start_ms,
                        "end_ms": s.end_ms,
                        "confidence": s.confidence,
                    }
                    for s in (raw_transcript.segments if raw_transcript else [])
                ],
            }))
        except Exception:
            pass

        # Cleanup
        if converter:
            await converter.close()
        _active_sessions.pop(session_id, None)

        logger.info("asr_stream: session ended — encounter=%s transcript_len=%d",
                     encounter_id, len(accumulated_transcript))
