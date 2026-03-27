"""
ASR MCP Server Base Interface.

All ASR engines (WhisperX, NeMo, Deepgram, …) implement ASREngine.
The Transcribe node calls only this interface — never the engine directly.

Implementations:
    whisperx_server.py    — DEFAULT (batch, diarization via pyannote)
    nemo_streaming_server.py — streaming, real-time
    deepgram_server.py    — cloud option (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class ASRConfig:
    """Per-request configuration passed to the ASR engine."""
    language: str = "en"
    sample_rate: int = 16000
    # Diarization
    diarize: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # Vocabulary boosting
    custom_vocabulary: list[str] = field(default_factory=list)
    # hotwords: direct logit boost for specific tokens during beam search.
    # More targeted than initial_prompt (which is a decoder prefix hint).
    # Use for rare specialty terms, drug names, provider-specific phrases.
    hotwords: list[str] = field(default_factory=list)
    # Vocabulary priming — passed as initial_prompt to faster-whisper decoder
    initial_prompt: Optional[str] = None
    # Decoding quality
    beam_size: int = 5
    vad_filter: bool = True
    # condition_on_previous_text: carry context across 30-second chunks.
    # True  → dictation (physician dictates a coherent document; prior chunk
    #          text primes the next chunk's decoder)
    # False → ambient (patient speech must NOT condition physician chunks;
    #          prevents cross-speaker hallucination)
    condition_on_previous_text: bool = True
    # compression_ratio_threshold: raise (e.g. 2.8) for physicians who use
    # repetitive clinical phrasing to avoid triggering temperature fallback.
    compression_ratio_threshold: float = 2.4
    # no_speech_threshold: lower (e.g. 0.4) for physicians with frequent
    # pauses mid-dictation to prevent segments being dropped as silence.
    no_speech_threshold: float = 0.6
    # vad_threshold: speech probability cutoff for Silero VAD (0.0–1.0).
    # Lower (e.g. 0.3) for soft-spoken providers or noisy environments.
    # Higher (e.g. 0.7) to aggressively suppress background noise.
    vad_threshold: float = 0.5


@dataclass
class WordAlignment:
    text: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0


@dataclass
class RawSegment:
    """Raw transcription segment from the ASR engine (before post-processing)."""
    text: str
    start_ms: int
    end_ms: int
    speaker: Optional[str] = None
    confidence: float = 1.0
    words: list[WordAlignment] = field(default_factory=list)


@dataclass
class RawTranscript:
    """Complete raw output from an ASR engine call."""
    segments: list[RawSegment]
    engine: str
    model: str
    language: str
    audio_duration_ms: int
    diarization_applied: bool = False


@dataclass
class PartialTranscript:
    """Streaming chunk from real-time ASR."""
    text: str
    is_final: bool
    speaker: Optional[str] = None
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 1.0


@dataclass
class ASRCapabilities:
    streaming: bool = False
    batch: bool = True
    diarization: bool = False
    word_alignment: bool = False
    medical_vocab: bool = False
    max_speakers: int = 1
    supported_formats: list[str] = field(default_factory=lambda: ["wav", "mp3", "m4a", "flac"])
    max_audio_duration_s: Optional[int] = None


class ASREngine(ABC):
    """
    Abstract base class for all ASR MCP servers.

    Every ASR engine must implement these three methods.
    The transcribe node calls only this interface.
    """

    @abstractmethod
    async def transcribe_batch(
        self,
        audio_path: str,
        config: ASRConfig,
    ) -> RawTranscript:
        """
        Transcribe an audio file and return the full transcript.

        Args:
            audio_path: Absolute path to the audio file.
            config:     Per-request ASR configuration.

        Returns:
            RawTranscript with segments, timing, and optional diarization.
        """
        ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunk: bytes,
        session_id: str,
        config: ASRConfig,
    ) -> AsyncIterator[PartialTranscript]:
        """
        Process a streaming audio chunk and yield partial transcripts.

        Args:
            audio_chunk: Raw PCM bytes (16kHz, 16-bit, mono).
            session_id:  Session identifier for stateful streaming.
            config:      Per-request ASR configuration.

        Yields:
            PartialTranscript chunks (is_final=True for sentence-complete chunks).
        """
        ...

    @abstractmethod
    async def get_capabilities(self) -> ASRCapabilities:
        """
        Return the capabilities of this ASR engine.

        Used by the ASR router to select appropriate engine for the task.
        """
        ...

    async def health_check(self) -> bool:
        """
        Verify the engine is reachable and ready.

        Default: try to call get_capabilities(). Override for custom health checks.
        """
        try:
            await self.get_capabilities()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        """Engine name (used in state logging and routing)."""
        return self.__class__.__name__.replace("Server", "").lower()
