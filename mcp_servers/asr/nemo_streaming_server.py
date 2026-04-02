"""
NeMo Streaming ASR Server — real-time transcription via NVIDIA Nemotron-Speech-Streaming.

Implements the ASREngine interface for both streaming and batch modes.
Uses NeMo's cache-aware FastConformer-RNNT for low-latency chunk processing.

Model: nvidia/nemotron-speech-streaming-en-0.6b (~2-3 GB VRAM)
Input: 16 kHz, mono, 16-bit PCM audio chunks (configurable chunk size)
Output: PartialTranscript per chunk (text, is_final, confidence, timing)

Session management: each streaming session caches encoder hidden state.
Sessions auto-expire after idle_timeout_s (default 300s).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from mcp_servers.asr.base import (
    ASRCapabilities,
    ASRConfig,
    ASREngine,
    PartialTranscript,
    RawSegment,
    RawTranscript,
    WordAlignment,
)

logger = logging.getLogger(__name__)


@dataclass
class StreamingSession:
    """State for a single streaming ASR session."""
    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    accumulated_text: str = ""
    segments: list[RawSegment] = field(default_factory=list)
    elapsed_ms: int = 0
    chunk_count: int = 0
    # NeMo cache state (populated when model is loaded)
    cache_state: Any = None
    # Audio buffer for incomplete frames
    audio_buffer: bytes = b""
    # Full session PCM buffer (for NeMo chunked transcription)
    full_pcm: bytes = b""
    # Last transcription result (for diffing)
    last_transcription: str = ""
    # Bytes already transcribed (to know when new audio warrants re-transcription)
    last_transcribed_bytes: int = 0


class NemoStreamingServer(ASREngine):
    """
    Streaming ASR engine wrapping NVIDIA Nemotron-Speech-Streaming.

    Supports both streaming (transcribe_stream) and batch (transcribe_batch) modes.
    The model is lazy-loaded on first request and unloaded after idle_timeout_s.
    """

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        device: str = "cuda",
        chunk_size_ms: int = 160,
        idle_timeout_s: int = 300,
        hotwords: list[str] | None = None,
        hotwords_files: list[str] | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.chunk_size_ms = chunk_size_ms
        self.idle_timeout_s = idle_timeout_s

        # Derived constants
        self.sample_rate = 16000
        self.chunk_samples = int(self.sample_rate * self.chunk_size_ms / 1000)
        self.chunk_bytes = self.chunk_samples * 2  # 16-bit = 2 bytes per sample

        # Model state (lazy loaded)
        self._model = None
        self._model_lock = threading.Lock()
        self._loaded = False

        # Active streaming sessions
        self._sessions: dict[str, StreamingSession] = {}
        self._sessions_lock = threading.Lock()

        # Medical hotword correction map: lowercase phrase → correct form
        self._hotword_map: dict[str, str] = {}
        self._build_hotword_map(hotwords or [], hotwords_files or [])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NemoStreamingServer":
        """Instantiate from engines.yaml config dict."""
        return cls(
            model_name=config.get("model", "nvidia/nemotron-speech-streaming-en-0.6b"),
            device=config.get("device", "cuda"),
            chunk_size_ms=config.get("chunk_size_ms", 160),
            idle_timeout_s=config.get("idle_unload_seconds", 300),
            hotwords=config.get("hotwords", []),
            hotwords_files=config.get("hotwords_files", []),
        )

    # ── Hotword correction ───────────────────────────────────────────────

    def _build_hotword_map(self, hotwords: list[str], hotwords_files: list[str]) -> None:
        """Build the lowercase→correct-form lookup from inline terms and dictionary files.

        Inline hotwords are loaded with no length restriction (so 2-char medical
        abbreviations like DM, IV, BP work). Terms loaded from *files* skip anything
        ≤2 chars to prevent false-positive corrections from the 98K wordlist.
        """
        # Load inline hotwords first — no length restriction.
        for term in hotwords:
            key = term.lower().strip()
            if key:
                self._hotword_map[key] = term.strip()

        # Load file terms — skip very short (≤2 chars) to avoid false positives
        # from generic entries in the 98K OpenMedSpel wordlist ("aa", "ab", etc.).
        for path in hotwords_files:
            if not os.path.isabs(path):
                # Resolve relative to project root (two levels up from this file)
                base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                path = os.path.join(base, path)
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        term = line.strip()
                        if not term or term.startswith("#"):
                            continue
                        key = term.lower()
                        # Skip ≤2-char entries from files only
                        if len(key) > 2:
                            self._hotword_map[key] = term
            except OSError as exc:
                logger.warning("nemo_streaming: could not load hotwords file %s — %s", path, exc)

        if self._hotword_map:
            logger.info("nemo_streaming: loaded %d hotword entries", len(self._hotword_map))

    def reload_hotwords(self, hotwords: list[str], hotwords_files: list[str]) -> None:
        """Replace the hotword map at runtime (e.g. after provider switch)."""
        self._hotword_map.clear()
        self._build_hotword_map(hotwords, hotwords_files)

    def _apply_hotword_corrections(self, text: str, extra_hotwords: list[str] | None = None) -> str:
        """Post-process transcribed text to fix medical term casing/spelling.

        Longest-match-first scan: tries up to 5-word phrases, falls back
        to shorter spans so 'coronary artery disease' beats 'coronary' alone.
        Case-insensitive matching; original casing from the dictionary is restored.
        """
        if not text:
            return text

        # Merge in any per-request hotwords (e.g. from ASRConfig.hotwords)
        active_map = self._hotword_map
        if extra_hotwords:
            active_map = dict(self._hotword_map)
            for term in extra_hotwords:
                key = term.lower().strip()
                if key:
                    active_map[key] = term.strip()

        if not active_map:
            return text

        # Tokenise preserving punctuation attached to words
        tokens = re.split(r'(\s+)', text)  # odd indices = whitespace, even = words
        words = [t for t in tokens if not re.match(r'^\s+$', t)]
        spaces = [t for t in tokens if re.match(r'^\s+$', t)]

        max_phrase_len = 5
        result: list[str] = []
        i = 0
        while i < len(words):
            matched = False
            for length in range(min(max_phrase_len, len(words) - i), 0, -1):
                phrase_raw = " ".join(words[i:i + length])
                # Strip trailing punctuation for the lookup key only
                phrase_key = re.sub(r'[^\w\s]$', '', phrase_raw).lower()
                if phrase_key in active_map:
                    # Preserve any trailing punctuation that was on the last token
                    trailing = re.search(r'[^\w]$', words[i + length - 1])
                    corrected = active_map[phrase_key]
                    if trailing:
                        corrected += trailing.group()
                    result.append(corrected)
                    i += length
                    matched = True
                    break
            if not matched:
                result.append(words[i])
                i += 1

        # Re-interleave the original whitespace
        out_tokens: list[str] = []
        for idx, word in enumerate(result):
            out_tokens.append(word)
            if idx < len(spaces):
                out_tokens.append(spaces[idx])
        return "".join(out_tokens)

    @property
    def name(self) -> str:
        return "nemo_streaming"

    # ── Model lifecycle ──────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Lazy-load the NeMo model on first use."""
        if self._loaded:
            return
        with self._model_lock:
            if self._loaded:
                return
            try:
                import nemo.collections.asr as nemo_asr
                logger.info("nemo_streaming: loading model %s on %s", self.model_name, self.device)
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self.model_name,
                )
                if self.device == "cuda":
                    self._model = self._model.cuda()
                self._model.eval()
                self._loaded = True
                logger.info("nemo_streaming: model loaded successfully")
            except ImportError:
                logger.warning(
                    "nemo_streaming: NeMo not installed — streaming will use simulation mode. "
                    "Install with: pip install nemo_toolkit[asr]"
                )
                self._model = None
                self._loaded = True  # Mark as loaded so we don't retry
            except Exception as exc:
                logger.error("nemo_streaming: failed to load model — %s", exc)
                self._model = None
                self._loaded = True

    def unload_model(self) -> None:
        """Unload the model from GPU memory."""
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                self._loaded = False
                # Free GPU memory
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                logger.info("nemo_streaming: model unloaded")

    # ── Session management ───────────────────────────────────────────────

    def _get_or_create_session(self, session_id: str) -> StreamingSession:
        with self._sessions_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = StreamingSession(session_id=session_id)
                logger.info("nemo_streaming: new session %s", session_id)
            session = self._sessions[session_id]
            session.last_activity = time.monotonic()
            return session

    def _close_session(self, session_id: str) -> Optional[StreamingSession]:
        with self._sessions_lock:
            return self._sessions.pop(session_id, None)

    def get_session_transcript(self, session_id: str) -> Optional[str]:
        """Get the accumulated transcript for a session."""
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            return session.accumulated_text if session else None

    def finalize_session(self, session_id: str) -> Optional[RawTranscript]:
        """Close a session and return the accumulated transcript as a RawTranscript."""
        session = self._close_session(session_id)
        if session is None:
            return None

        return RawTranscript(
            segments=session.segments,
            engine="nemo_streaming",
            model=self.model_name,
            language="en",
            audio_duration_ms=session.elapsed_ms,
            diarization_applied=False,
        )

    def cleanup_expired_sessions(self) -> int:
        """Remove sessions that have been idle longer than idle_timeout_s."""
        now = time.monotonic()
        expired = []
        with self._sessions_lock:
            for sid, session in self._sessions.items():
                if now - session.last_activity >= self.idle_timeout_s:
                    expired.append(sid)
            for sid in expired:
                del self._sessions[sid]
        if expired:
            logger.info("nemo_streaming: expired %d idle sessions", len(expired))
        return len(expired)

    # ── Streaming transcription ──────────────────────────────────────────

    # Transcribe only the latest N seconds (not the full buffer)
    # Lower = more responsive but more GPU calls. 1s gives ~100ms inference per window.
    STREAM_WINDOW_S = 1.0

    async def transcribe_stream(
        self,
        audio_chunk: bytes,
        session_id: str,
        config: ASRConfig,
    ) -> AsyncIterator[PartialTranscript]:
        """
        Process a streaming audio chunk and yield partial transcripts.

        Strategy: sliding window — only transcribe the latest STREAM_WINDOW_S
        seconds of new audio (not the growing full buffer). Each window is
        independent, giving O(1) latency regardless of session length.

        If NeMo is not installed, uses a simulation mode for UI development.
        """
        self._ensure_model()
        session = self._get_or_create_session(session_id)

        # Buffer the incoming audio
        session.audio_buffer += audio_chunk

        # Move complete frame-aligned data to full_pcm
        while len(session.audio_buffer) >= self.chunk_bytes:
            chunk_data = session.audio_buffer[:self.chunk_bytes]
            session.audio_buffer = session.audio_buffer[self.chunk_bytes:]
            session.full_pcm += chunk_data
            session.chunk_count += 1
            session.elapsed_ms += self.chunk_size_ms

        # Check if we have enough new audio since last transcription
        new_bytes = len(session.full_pcm) - session.last_transcribed_bytes
        new_seconds = new_bytes / (self.sample_rate * 2)

        if new_seconds < self.STREAM_WINDOW_S:
            return

        start_ms = session.last_transcribed_bytes * 1000 // (self.sample_rate * 2)
        end_ms = session.elapsed_ms

        # Extract only the new audio window (not full buffer)
        window_pcm = session.full_pcm[session.last_transcribed_bytes:]
        session.last_transcribed_bytes = len(session.full_pcm)

        if self._model is not None:
            result = await asyncio.to_thread(
                self._transcribe_window_nemo, window_pcm
            )
        else:
            result = self._transcribe_window_simulated(session, start_ms, end_ms)

        if result:
            text = result.get("text", "").strip()
            confidence = result.get("confidence", 0.9)

            # Strip trailing sentence-final punctuation NeMo adds to every short
            # window — full punctuation is rebuilt by the LLM cleanup pass.
            text = re.sub(r'[.!?]+$', '', text).strip()

            # Apply medical hotword corrections
            extra_hw = (config.hotwords if config is not None else None) or None
            text = self._apply_hotword_corrections(text, extra_hotwords=extra_hw)

            if text:
                partial = PartialTranscript(
                    text=text,
                    is_final=True,
                    speaker=None,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=confidence,
                )

                session.accumulated_text += text + " "
                session.segments.append(RawSegment(
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=confidence,
                ))

                yield partial

    def _transcribe_window_nemo(self, window_pcm: bytes) -> dict:
        """Transcribe a single audio window via NeMo (runs in thread).

        Only transcribes the latest window — O(1) per call regardless of
        total session length. ~60-100ms per 3-second window on A10G.
        """
        try:
            import numpy as np
            import soundfile as sf
            import tempfile
            import os

            audio_array = np.frombuffer(window_pcm, dtype=np.int16).astype(np.float32) / 32768.0

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_array, self.sample_rate)
                temp_path = f.name

            results = self._model.transcribe([temp_path])
            text = results[0] if isinstance(results, list) else str(results)
            if hasattr(text, 'text'):
                text = text.text

            os.unlink(temp_path)

            return {"text": text, "confidence": 0.9}

        except Exception as exc:
            logger.warning("nemo_streaming: window transcription failed — %s", exc)
            return {}

    def _transcribe_window_simulated(
        self, session: StreamingSession,
        start_ms: int, end_ms: int,
    ) -> dict:
        """Simulation mode: return placeholder text for UI development."""
        segment_num = len(session.segments) + 1
        return {
            "text": f"[streaming segment {segment_num} at {start_ms}ms]",
            "confidence": 0.85,
        }

    # ── Batch transcription (compatibility) ──────────────────────────────

    async def transcribe_batch(
        self,
        audio_path: str,
        config: ASRConfig,
    ) -> RawTranscript:
        """
        Transcribe a full audio file in batch mode.

        Uses NeMo's standard (non-streaming) transcription if available,
        otherwise falls back to a stub result.
        """
        self._ensure_model()

        if self._model is not None:
            result = await asyncio.to_thread(self._batch_transcribe_nemo, audio_path, config)
            return result

        # Simulation mode
        logger.warning("nemo_streaming: batch mode — NeMo not installed, returning stub")
        return RawTranscript(
            segments=[RawSegment(
                text="[NeMo not installed — batch transcription unavailable]",
                start_ms=0,
                end_ms=0,
            )],
            engine="nemo_streaming",
            model=self.model_name,
            language="en",
            audio_duration_ms=0,
        )

    def _batch_transcribe_nemo(self, audio_path: str, config: ASRConfig) -> RawTranscript:
        """Run NeMo batch transcription (in thread)."""
        try:
            results = self._model.transcribe([audio_path])
            text = results[0] if results else ""

            return RawTranscript(
                segments=[RawSegment(
                    text=text,
                    start_ms=0,
                    end_ms=0,
                    confidence=0.9,
                )],
                engine="nemo_streaming",
                model=self.model_name,
                language="en",
                audio_duration_ms=0,
            )
        except Exception as exc:
            logger.error("nemo_streaming: batch transcription failed — %s", exc)
            return RawTranscript(
                segments=[RawSegment(
                    text=f"[ASR error: {exc}]",
                    start_ms=0,
                    end_ms=0,
                )],
                engine="nemo_streaming",
                model=self.model_name,
                language="en",
                audio_duration_ms=0,
            )

    # ── Sync wrapper (for LangGraph which calls sync) ────────────────────

    def transcribe_batch_sync(self, audio_path: str, config: ASRConfig) -> RawTranscript:
        """Synchronous wrapper for batch transcription."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.transcribe_batch(audio_path, config))
                return future.result()
        return asyncio.run(self.transcribe_batch(audio_path, config))

    # ── Capabilities ─────────────────────────────────────────────────────

    async def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            streaming=True,
            batch=True,
            diarization=False,
            word_alignment=False,
            medical_vocab=True,   # hotword correction is active
            max_speakers=1,
            supported_formats=["pcm", "wav"],
        )