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

# ── Spoken date normalization ─────────────────────────────────────────────────
# NeMo transcribes spoken dates as lowercase words.  These tables let us convert
# e.g. "october twenty fifth twenty twenty four" → "October 25, 2024".

_SPOKEN_MONTHS: dict[str, str] = {
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June",
    "july": "July", "august": "August", "september": "September",
    "october": "October", "november": "November", "december": "December",
}

# Simple one-word ordinals (e.g. "fifteenth" → 15)
_SPOKEN_ORDINALS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}

# Ordinal suffix words used in compound ordinals ("twenty FIRST", "thirty FIRST")
_ORDINAL_ONES: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
}

_SPOKEN_TENS: dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SPOKEN_ONES: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_SPOKEN_TEENS: dict[str, int] = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

# ── Cardinal number → digit conversion ───────────────────────────────────────
# Used by _convert_medical_numbers.  Ordinals ("first", "second"…) are kept
# separate in _SPOKEN_ORDINALS so they are NOT turned into standalone digits.

_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}

# ALL-CAPS words that are common English function words mistakenly capitalised
# by NeMo (e.g. "THE", "OF").  These are always safe to lowercase.
_ALLCAPS_STOPWORDS: frozenset[str] = frozenset([
    "THE", "A", "AN", "OF", "TO", "IN", "IS", "ARE", "WAS", "WERE",
    "BE", "BEEN", "BEING", "AND", "OR", "BUT", "FOR", "WITH", "AT",
    "BY", "FROM", "AS", "INTO", "ON", "OFF", "THAT", "THIS", "HER",
    "HIS", "ITS", "SHE", "HE", "IT", "WE", "THEY", "HIM", "THEM",
])

# ── Medical note section-header patterns ──────────────────────────────────────
# When a provider dictates a section header it should appear in ALL CAPS in the
# transcript (the frontend renders it as a heading).  Patterns are matched
# case-insensitively; the longest / most-specific are listed first.
_SECTION_HEADER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bhist(?:ory)?\s+of\s+present\s+illness\b', re.IGNORECASE),
     "HISTORY OF PRESENT ILLNESS"),
    (re.compile(r'\bpast\s+(?:medical\s+)?hist(?:ory)?\b', re.IGNORECASE),
     "PAST MEDICAL HISTORY"),
    (re.compile(r'\bphysical\s+exam(?:ination)?\b', re.IGNORECASE),
     "PHYSICAL EXAMINATION"),
    (re.compile(r'\breview\s+of\s+systems\b', re.IGNORECASE),
     "REVIEW OF SYSTEMS"),
    (re.compile(r'\bchief\s+complaint\b', re.IGNORECASE),
     "CHIEF COMPLAINT"),
    (re.compile(r'\bassessment\b', re.IGNORECASE),
     "ASSESSMENT"),
    (re.compile(r'\bplan\b', re.IGNORECASE),
     "PLAN"),
]

# ── Optional wordninja import ─────────────────────────────────────────────────
# wordninja splits merged English words using word-frequency statistics.
# e.g. "historyOf" → "history Of"  /  "neck paintand" → "neck paint and"
# Install with:  pip install wordninja
# If not installed the feature is silently disabled (no error).
try:
    import wordninja as _wordninja
    _WORDNINJA_AVAILABLE = True
except ImportError:
    _WORDNINJA_AVAILABLE = False

# ── Phonetic corrections ──────────────────────────────────────────────────────
# ASR phonetic misrecognitions → correct medical/clinical forms.
# These are seeded into the hotword map at startup so they use the same
# longest-match-first algorithm as hotword casing corrections.
# Key   = what NeMo actually outputs (lowercase, as-spoken)
# Value = what it should be (correct medical form)
_PHONETIC_CORRECTIONS: dict[str, str] = {
    # ── Reflexes ──────────────────────────────────────────────────────────
    "deep tendon duplexes": "deep tendon reflexes",
    "deep tendon duplicates": "deep tendon reflexes",
    "deep tendon complex": "deep tendon reflexes",
    "deep tendon reflex is": "deep tendon reflexes",
    "two plex": "2+",
    "to plex": "2+",
    # ── Spurling's sign ───────────────────────────────────────────────────
    "spiraling sign": "Spurling's sign",
    "spirling sign": "Spurling's sign",
    "sperlings sign": "Spurling's sign",
    "sparlings sign": "Spurling's sign",
    "spurlings sign": "Spurling's sign",
    "sporting sign": "Spurling's sign",
    "spearling sign": "Spurling's sign",
    "spelling sign": "Spurling's sign",
    # ── Radiculitis / radiculopathy ───────────────────────────────────────
    "radical itis": "radiculitis",
    "radicular itis": "radiculitis",
    "radical opathy": "radiculopathy",
    "radicular opathy": "radiculopathy",
    "radio culopathy": "radiculopathy",
    # NeMo mishears "radiculitis" as "reticulities" / "reticulo*" variants
    "reticulities": "radiculitis",
    "cervical reticulities": "cervical radiculitis",
    "lumbar reticulities": "lumbar radiculitis",
    "retic ulitis": "radiculitis",
    "reticulo pathy": "radiculopathy",
    "cervical reticulo pathy": "cervical radiculopathy",
    # ── Cervicothoracic / thoracolumbar ────────────────────────────────────
    "cervico thoracic": "cervicothoracic",
    "cervico thoracic junction": "cervicothoracic junction",
    "cervico thoracic spine": "cervicothoracic spine",
    "thoraco lumbar": "thoracolumbar",
    "thoraco lumbar junction": "thoracolumbar junction",
    "thoraco lumbar spine": "thoracolumbar spine",
    "thoraco lumbar fascia": "thoracolumbar fascia",
    # ── Cervical / lumbar levels ──────────────────────────────────────────
    "cdl seven": "C7",
    "cdl six": "C6",
    "cdl five": "C5",
    "cdl four": "C4",
    "cdl three": "C3",
    "the cdl seven": "C7",
    "c seven": "C7",
    "c six": "C6",
    "c five": "C5",
    "l five s one": "L5-S1",
    "l four l five": "L4-L5",
    "l five": "L5",
    "l four": "L4",
    # ── Posttraumatic / postconcussive ────────────────────────────────────
    "post concussive": "postconcussive",
    "post concussive syndrome": "postconcussive syndrome",
    "post traumatic": "posttraumatic",
    "post traumatic headaches": "posttraumatic headaches",
    # ── Neurofibromatosis ─────────────────────────────────────────────────
    "neuro fibromatosis": "neurofibromatosis",
    "neuro fibroma": "neurofibroma",
    "neuro fibromas": "neurofibromas",
    # ── Myofascial ────────────────────────────────────────────────────────
    "myo fascial": "myofascial",
    "mayo fascial": "myofascial",
    "my o facial": "myofascial",
    # ── Common medications ────────────────────────────────────────────────
    "pro prana lol": "propranolol",
    "propanol": "propranolol",
    "pro pran alol": "propranolol",
    "gap a pentin": "gabapentin",
    "gamma pentin": "gabapentin",
    "gaba pentin": "gabapentin",
    "cyclo benso preen": "cyclobenzaprine",
    "cyclo benz a preen": "cyclobenzaprine",
    "melo xicam": "meloxicam",
    "mel oxicam": "meloxicam",
    "uber levy": "Ubrelvy",
    "uber leave me": "Ubrelvy",
    "nur tech": "Nurtec",
    "nur tec": "Nurtec",
    "nor tec": "Nurtec",
    "zow fran": "Zofran",
    "zoe fran": "Zofran",
    "zon a flex": "Zanaflex",
    "zona flex": "Zanaflex",
    "kill epta": "Qulipta",
    "quill ipta": "Qulipta",
    "quill apta": "Qulipta",
    "riza triptan": "rizatriptan",
    "soma triptan": "sumatriptan",
    "suma triptan": "sumatriptan",
    "pheno barbital": "phenobarbital",
    "feno barbital": "phenobarbital",
    # ── Procedures / injections ───────────────────────────────────────────
    "trigger point injections": "trigger point injections",
    "epidural steroid injection": "epidural steroid injection",
    "inter laminar": "interlaminar",
    "inter laminar epidural": "interlaminar epidural",
    "para spinal": "paraspinal",
    "para spinal musculature": "paraspinal musculature",
    # ── Anatomy ───────────────────────────────────────────────────────────
    "sack ro iliac": "sacroiliac",
    "sacro iliac": "sacroiliac",
    "peri neural": "perineural",
    "trans itional": "transitional",
    # ── Conditions ────────────────────────────────────────────────────────
    "dis equilibrium": "disequilibrium",
    "this equilibrium": "disequilibrium",
    "paris thesias": "paresthesias",
    "pair asthesias": "paresthesias",
    "spondy lo listhesis": "spondylolisthesis",
    "spondylo listhesis": "spondylolisthesis",
    "spinal listen thesis": "spondylolisthesis",
    "neural fibromatosis": "neurofibromatosis",
    "occipital neural gia": "occipital neuralgia",
    # ── Cognitive evaluation ──────────────────────────────────────────────
    "serial seven": "Serial-7",
    "serial sevens": "Serial-7",
    "romberg sign": "Romberg sign",
    # ── Additional phonetic errors seen in live dictation ─────────────────
    # Zofran variants
    "zaffron": "Zofran",
    "zaf fran": "Zofran",
    "zaf ron": "Zofran",
    "zaff ron": "Zofran",
    # Myofascial
    "myofacial": "myofascial",
    "myo facial": "myofascial",
    "mayo facial": "myofascial",
    # Vestibular
    "tibular": "vestibular",
    "tibular cognitive therapy": "vestibular cognitive therapy",
    "tib ular": "vestibular",
    # "further details" misheard as "Firefox details"
    "firefox details": "further details",
    "fire fox details": "further details",
    # "pre-injury job" phrasing
    "pre job injury": "pre-injury job duties",
    # Chiropractic
    "chiro practic": "chiropractic",
    "chiro practic therapy": "chiropractic therapy",
    # "she is no complaining" → "she is not complaining"  (NeMo hears "no" for "not")
    "is no complaining": "is not complaining",
    "is no able to": "is not able to",
    "she is no able": "she is not able",
    # Visual disturbance (merged capitalised)
    "visualdisturbance": "visual disturbance",
    # "date of access" → avoid interpreting "access" as a stop word
    "date of access": "date of access",
}

# ── Optional deepmultilingualpunctuation import ───────────────────────────────
# deepmultilingualpunctuation restores sentence punctuation (. , ?) and
# capitalises sentence starts.  Loaded once as a singleton at first use.
# Model is ~300 MB; ~80 ms per segment on CPU.
# Install with:  pip install deepmultilingualpunctuation
# If not installed the feature is silently disabled (no error); in either case
# the punctuation pass is bounded to PUNCT_MAX_WORKERS concurrent CPU calls so
# it cannot overload the server under high concurrency (100+ providers).
_PUNCT_MAX_WORKERS = 4          # max simultaneous punctuation inference calls
_punct_semaphore = threading.BoundedSemaphore(_PUNCT_MAX_WORKERS)
_punct_model_lock = threading.Lock()
_punct_model: "object | None" = None   # holds PunctuationModel instance once loaded
_PUNCT_MODEL_AVAILABLE: bool = False

try:
    from deepmultilingualpunctuation import PunctuationModel as _PunctuationModel  # type: ignore[import]
    _PUNCT_MODEL_AVAILABLE = True
except ImportError:
    _PUNCT_MODEL_AVAILABLE = False


def _get_punct_model() -> "object | None":
    """Return the singleton PunctuationModel, loading on first call (thread-safe)."""
    global _punct_model
    if _punct_model is not None:
        return _punct_model
    if not _PUNCT_MODEL_AVAILABLE:
        return None
    with _punct_model_lock:
        if _punct_model is None:
            try:
                _punct_model = _PunctuationModel()
                logger.info("nemo_streaming: punctuation model loaded")
            except Exception as exc:
                logger.warning("nemo_streaming: punctuation model failed to load — %s", exc)
                _punct_model = None
    return _punct_model


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
        stream_window_s: float = 3.0,
    ):
        self.model_name = model_name
        self.device = device
        self.chunk_size_ms = chunk_size_ms
        self.idle_timeout_s = idle_timeout_s
        # How many seconds of NEW audio to accumulate before triggering a decode.
        # Higher = better accuracy and spacing (model has more context) but longer
        # delay before text appears in the UI.
        # 1s  — ~1s latency, poor word boundaries, period-per-utterance
        # 3s  — ~3s latency, good quality for clear speech  (DEFAULT)
        # 5s  — ~5s latency, best accuracy for accented/fast speech
        self.STREAM_WINDOW_S: float = stream_window_s

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
            stream_window_s=config.get("stream_window_s", 3.0),
        )

    # ── Hotword correction ───────────────────────────────────────────────

    def _build_hotword_map(self, hotwords: list[str], hotwords_files: list[str]) -> None:
        """Build the lowercase→correct-form lookup from inline terms and dictionary files.

        Inline hotwords are loaded with no length restriction (so 2-char medical
        abbreviations like DM, IV, BP work). Terms loaded from *files* skip anything
        ≤2 chars to prevent false-positive corrections from the 98K wordlist.

        Priority order (highest wins):
          1. Inline per-server hotwords (passed to __init__)
          2. Phonetic corrections (_PHONETIC_CORRECTIONS) — always override wordlist
          3. Dictionary / wordlist files

        _PHONETIC_CORRECTIONS are applied LAST so that a generic wordlist entry like
        "propanol" in the 98K OpenMedSpel list cannot silently cancel a known
        misrecognition fix ("propanol" → "propranolol").
        """
        # Load file terms first (lowest priority).
        # Skip very short (≤2 chars) to avoid false positives from generic entries
        # in the 98K OpenMedSpel wordlist ("aa", "ab", etc.).
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

        # Phonetic corrections override file-loaded terms (they fix known misrecognitions
        # that must not be cancelled by a generic wordlist casing entry).
        self._hotword_map.update(_PHONETIC_CORRECTIONS)

        # Inline hotwords override everything — provider-specific exact spellings.
        for term in hotwords:
            key = term.lower().strip()
            if key:
                self._hotword_map[key] = term.strip()

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

        # Re-interleave the original whitespace.
        # When multi-word phrases collapse to fewer words the space count can
        # become mismatched, producing a trailing space — strip it.
        out_tokens: list[str] = []
        for idx, word in enumerate(result):
            out_tokens.append(word)
            if idx < len(spaces):
                out_tokens.append(spaces[idx])
        return "".join(out_tokens).rstrip()

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

    # ── Spoken date & text normalization ────────────────────────────────

    # ── wordninja: split camelCase / merged-word tokens ──────────────────

    def _split_merged_words(self, text: str) -> str:
        """Split merged or camelCase tokens using wordninja.

        Two cases handled:
          1. CamelCase merges  — internal uppercase boundary, e.g. "historyOf",
             "Visualdisturbance", "AccESSOctober", "THEpatient".
          2. All-lowercase merges — no uppercase at all, e.g. "todaypostconcussive",
             "headacheand", "wellvestibular".  These are caught by a minimum length
             threshold (> 14 chars) combined with a split-quality check: wordninja
             must return at least 2 parts AND each part must be ≥ 3 chars.

        Tokens that are:
          • short (≤ 6 chars for camelCase; ≤ 14 chars for pure-lowercase)
          • pure ALL-CAPS known abbreviations (e.g. ASR, HTN, IV)
          • present in the medical hotword map (single-word entries)
        are left untouched to protect medical terminology.

        Silently returns original text when wordninja is not installed.
        """
        if not _WORDNINJA_AVAILABLE:
            return text

        # Boundary indicating two merged words: lowercase→Uppercase ("historyOf")
        # or ALL-CAPS run followed by lowercase ("THEpatient", "AccESSOctober").
        _camel_re = re.compile(r'[a-z][A-Z]|[A-Z]{2,}[a-z]')

        # Minimum token length to attempt all-lowercase splitting.
        # "postconcussive" is 14 chars; shorter single words should not be split.
        _LOWERCASE_SPLIT_THRESHOLD = 14

        tokens = re.split(r'(\s+)', text)
        result: list[str] = []
        for token in tokens:
            if re.match(r'^\s+$', token):
                result.append(token)
                continue

            # Separate leading / trailing punctuation from the core word
            m_lead = re.match(r'^([^\w]+)', token)
            m_trail = re.search(r'([^\w]+)$', token)
            lead  = m_lead.group(1)  if m_lead  else ""
            trail = m_trail.group(1) if m_trail else ""
            core  = token[len(lead): len(token) - len(trail) if trail else len(token)]

            # Guard: pure ALL-CAPS abbreviation or single-word hotword entry — never split
            if core.isupper() or core.lower() in self._hotword_map:
                result.append(token)
                continue

            has_camel = bool(_camel_re.search(core))
            is_long_lower = (not has_camel
                             and len(core) > _LOWERCASE_SPLIT_THRESHOLD
                             and core.replace("-", "").replace("'", "").isalpha())

            # Short token with no camelCase → leave it alone
            if len(core) <= 6 and not has_camel:
                result.append(token)
                continue

            if not has_camel and not is_long_lower:
                result.append(token)
                continue

            parts = _wordninja.split(core)

            # Quality gate for all-lowercase splits: each part must be ≥ 3 chars.
            # This rejects bad splits like ["a", "nd"] for "and".
            if is_long_lower and not has_camel:
                if len(parts) < 2 or any(len(p) < 3 for p in parts):
                    result.append(token)
                    continue

            if len(parts) > 1:
                result.append(lead + " ".join(parts) + trail)
            else:
                result.append(token)

        return "".join(result)

    # ── ALL-CAPS function-word normalizer ────────────────────────────────

    def _normalize_allcaps_stopwords(self, text: str) -> str:
        """Lowercase ALL-CAPS common English function words that NeMo misraises.

        NeMo sometimes capitalises entire tokens when its confidence is low
        (e.g. "THE patient" → "THE patient" where "THE" should be "the").
        This only lowercases tokens listed in _ALLCAPS_STOPWORDS; all other
        ALL-CAPS tokens (abbreviations like HTN, IV, MRI) are preserved.
        First token of the text is never lowercased so sentence start is kept.
        """
        words = text.split()
        for idx, w in enumerate(words):
            bare = w.rstrip(".,;:!?")  # strip trailing punctuation for comparison
            if bare in _ALLCAPS_STOPWORDS:
                if idx > 0:  # never lowercase very first word
                    words[idx] = w.lower()
        return " ".join(words)

    # ── Cardinal number → digit conversion ──────────────────────────────

    @staticmethod
    def _parse_cardinal(words: list[str]) -> "tuple[int, int] | None":
        """Parse a spoken cardinal number from the start of a word list.

        Returns (integer_value, words_consumed) or None.
        Handles 0–999.  Does NOT handle ordinals (first/second/…) — those
        are kept in _SPOKEN_ORDINALS for date parsing.

        Examples:
            ["sixty", "five"] → (65, 2)
            ["one", "hundred", "twenty", "three"] → (123, 4)
            ["five", "by", "five"] → (5, 1)  # stops before "by"
        """
        if not words:
            return None
        w = words[0].lower().rstrip(".,;:!?-")
        if w not in _NUMBER_WORDS:
            return None

        val = _NUMBER_WORDS[w]
        consumed = 1

        # "N hundred [rest]"
        if (consumed < len(words)
                and words[consumed].lower().rstrip(".,;:!?-") == "hundred"
                and 1 <= val <= 9):
            hundreds = val * 100
            consumed += 1
            # Optional "and" after hundred
            if consumed < len(words) and words[consumed].lower() == "and":
                consumed += 1
            # Try to parse tens+ones (0–99) after "hundred"
            if consumed < len(words):
                w2 = words[consumed].lower().rstrip(".,;:!?-")
                if w2 in _NUMBER_WORDS:
                    v2 = _NUMBER_WORDS[w2]
                    sub = 1
                    if v2 >= 20 and consumed + 1 < len(words):
                        w3 = words[consumed + 1].lower().rstrip(".,;:!?-")
                        if w3 in _NUMBER_WORDS and 1 <= _NUMBER_WORDS[w3] <= 9:
                            return (hundreds + v2 + _NUMBER_WORDS[w3], consumed + 2)
                    return (hundreds + v2, consumed + 1)
            return (hundreds, consumed)

        # Century-year pattern: "twenty twenty six" → 2026, "nineteen ninety five" → 1995
        # Only fires when val is 19 or 20 and the remainder forms a 2-digit (10–99) number
        # yielding a plausible year (1900–2100). This avoids converting "twenty five" → 2005.
        if val in (19, 20):
            sub = NemoStreamingServer._parse_cardinal(words[consumed:])
            if sub is not None and 10 <= sub[0] <= 99:
                year_cand = val * 100 + sub[0]
                if 1900 <= year_cand <= 2100:
                    return (year_cand, consumed + sub[1])

        # Tens + optional ones: "twenty five" → 25
        if val >= 20 and consumed < len(words):
            w2 = words[consumed].lower().rstrip(".,;:!?-")
            if w2 in _NUMBER_WORDS and 1 <= _NUMBER_WORDS[w2] <= 9:
                return (val + _NUMBER_WORDS[w2], 2)

        return (val, consumed)

    def _convert_medical_numbers(self, text: str) -> str:
        """Convert spoken number words to digits with medical-specific patterns.

        Priority patterns (checked before generic digit replacement):
          1. "five out of five" / "five by five" / "five slash five" → "5/5"
             (covers strength grading, recall scoring)
          2. "two plus" / "one plus" → "2+" / "1+"
             (deep-tendon reflex grading)
          3. "twenty seven year old" → "27-year-old"
             (demographic age)
          4. All remaining number words → digits
             "sixty five" → "65",  "three hundred mg" → "300 mg"
        """
        words = text.split()
        result: list[str] = []
        i = 0
        while i < len(words):
            num_result = self._parse_cardinal(words[i:])
            if num_result is None:
                result.append(words[i])
                i += 1
                continue

            num_val, num_consumed = num_result
            j = i + num_consumed          # index of first word AFTER the number
            remaining = words[j:]

            def _trail(word: str) -> str:
                m = re.search(r'[^\w]+$', word)
                return m.group() if m else ""

            # ── Pattern 1: X out of Y / X by Y / X slash Y → "X/Y" ──────
            if remaining:
                r0 = remaining[0].lower().rstrip(".,;:")
                if r0 in ("out", "by", "/", "slash"):
                    skip = 2 if (r0 == "out"
                                 and len(remaining) > 1
                                 and remaining[1].lower() == "of") else 1
                    y_words = remaining[skip:]
                    y_result = self._parse_cardinal(y_words)
                    if y_result:
                        y_val, y_consumed = y_result
                        trail_str = _trail(y_words[y_consumed - 1])
                        result.append(f"{num_val}/{y_val}{trail_str}")
                        i = j + skip + y_consumed
                        continue

            # ── Pattern 2: X plus → "X+" ─────────────────────────────────
            if remaining and remaining[0].lower().rstrip(".,;:") == "plus":
                result.append(f"{num_val}+{_trail(remaining[0])}")
                i = j + 1
                continue

            # ── Pattern 3: X year old → "X-year-old" ────────────────────
            if len(remaining) >= 2:
                r0 = remaining[0].lower()
                r1 = remaining[1].lower().rstrip(".,;:")
                if r0 == "year" and r1 == "old":
                    result.append(f"{num_val}-year-old{_trail(remaining[1])}")
                    i = j + 2
                    continue

            # ── General: number word(s) → digit ──────────────────────────
            trail_str = _trail(words[j - 1])
            result.append(f"{num_val}{trail_str}")
            i = j

        return " ".join(result)

    def _normalize_numeric_date(self, text: str) -> str:
        """Convert numeric digit date sequences to formatted dates.

        After _convert_medical_numbers runs, handles two patterns:

          4-digit year (D/M/YYYY or M/D/YYYY):
            "24 2 2026" → "24/2/2026"   (d1≤31, d2≤12, year 1900-2100)

          2-digit year, American M/D/YY (d1 must be a valid month 1-12):
            "2 14 23"  → "2/14/2023"   (d1≤12, d2≤31, yr≤30 → 2000+yr)
            "2 8 23"   → "2/8/2023"
            "10 5 24"  → "10/5/2024"

          For 2-digit years: 00-30 → 2000-2030, 31-99 → 1931-1999.
          The 2-digit path requires d1≤12 so that bare 3-number medical
          expressions (e.g. "24 mg 10") are not mistaken for dates.
        """
        def _replace_4(m: re.Match) -> str:
            d1, d2, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 1 <= d1 <= 31 and 1 <= d2 <= 12 and 1900 <= yr <= 2100:
                return f"{d1}/{d2}/{yr}"
            return m.group(0)

        def _replace_2(m: re.Match) -> str:
            d1, d2, yr_short = int(m.group(1)), int(m.group(2)), int(m.group(3))
            yr = 2000 + yr_short if yr_short <= 30 else 1900 + yr_short
            # Only fire when d1 is a valid month (1-12) to reduce false positives.
            if 1 <= d1 <= 12 and 1 <= d2 <= 31:
                return f"{d1}/{d2}/{yr}"
            return m.group(0)

        # 4-digit year first (more specific, can't be confused with 2-digit)
        text = re.sub(r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b', _replace_4, text)
        # 2-digit year: M/D/YY only when d1 is a valid month
        text = re.sub(r'\b(\d{1,2})\s+(\d{1,2})\s+(\d{2})\b', _replace_2, text)
        return text

    # ── Ordinal-day parser (for spoken-date normalizer) ──────────────────

    @staticmethod
    def _parse_ordinal_day(words: list[str]) -> "tuple[int, int] | None":
        """Parse spoken ordinal day from a word list.

        Returns (day_number, words_consumed) or None.
        Examples: ["fifteenth"] → (15, 1)
                  ["twenty", "fifth"] → (25, 2)
        """
        if not words:
            return None
        w = words[0]
        # Simple one-word ordinals: "first", "fifteenth", "twentieth", etc.
        if w in _SPOKEN_ORDINALS:
            return (_SPOKEN_ORDINALS[w], 1)
        # Compound two-word ordinals: "twenty first", "thirty first", etc.
        if w in _SPOKEN_TENS and len(words) >= 2 and words[1] in _ORDINAL_ONES:
            return (_SPOKEN_TENS[w] + _ORDINAL_ONES[words[1]], 2)
        return None

    @staticmethod
    def _parse_spoken_year(words: list[str]) -> "tuple[int, int] | None":
        """Parse spoken year from a word list.

        Returns (year, words_consumed) or None.
        Examples: ["twenty", "twenty", "four"] → (2024, 3)
                  ["nineteen", "eighty", "five"] → (1985, 3)
                  ["twenty", "twenty"] → (2020, 2)
        """
        if not words:
            return None
        w = words[0]
        # Century word must be teens (nineteen) or tens (twenty, thirty, …)
        if w in _SPOKEN_TEENS:
            century = _SPOKEN_TEENS[w] * 100
        elif w in _SPOKEN_TENS:
            century = _SPOKEN_TENS[w] * 100
        else:
            return None

        remaining = words[1:]
        if not remaining:
            return None  # bare "twenty" alone is not a year

        r0 = remaining[0]
        # Teens: "twenty sixteen" → 2016
        if r0 in _SPOKEN_TEENS:
            return (century + _SPOKEN_TEENS[r0], 2)
        # Tens (+optional ones): "twenty twenty" → 2020, "twenty twenty four" → 2024
        if r0 in _SPOKEN_TENS:
            yy = _SPOKEN_TENS[r0]
            if len(remaining) >= 2 and remaining[1] in _SPOKEN_ONES:
                return (century + yy + _SPOKEN_ONES[remaining[1]], 3)
            return (century + yy, 2)
        # Single ones: "twenty four" → 2004 (unusual but valid)
        if r0 in _SPOKEN_ONES:
            return (century + _SPOKEN_ONES[r0], 2)
        return None

    def _normalize_spoken_dates(self, text: str) -> str:
        """Convert spoken date patterns in text to written date format.

        Examples:
            "october twenty fifth twenty twenty four"  → "October 25, 2024"
            "january fifteenth twenty twenty six"      → "January 15, 2026"
            "february second twenty twenty six"        → "February 2, 2026"
        Month-only or month+day (no year) are also handled gracefully.
        """
        words = text.split()
        result: list[str] = []
        i = 0
        while i < len(words):
            # Strip trailing punctuation for month detection; preserve it
            raw_word = words[i]
            w_lower = raw_word.lower().rstrip(",.")
            trailing = raw_word[len(w_lower):]  # the stripped punctuation, if any

            if w_lower not in _SPOKEN_MONTHS:
                result.append(raw_word)
                i += 1
                continue

            month_str = _SPOKEN_MONTHS[w_lower]
            remaining = [ww.lower().rstrip(",.") for ww in words[i + 1:]]

            day_result = self._parse_ordinal_day(remaining)
            if day_result is None:
                # Not followed by an ordinal day — keep as-is
                result.append(raw_word)
                i += 1
                continue

            day, day_consumed = day_result
            year_result = self._parse_spoken_year(remaining[day_consumed:])

            if year_result:
                year, year_consumed = year_result
                result.append(f"{month_str} {day}, {year}{trailing}")
                i += 1 + day_consumed + year_consumed
            else:
                result.append(f"{month_str} {day}{trailing}")
                i += 1 + day_consumed

        return " ".join(result)

    async def _punctuate_text(self, text: str) -> str:
        """Restore punctuation and capitalisation using deepmultilingualpunctuation.

        Production safety guarantees (100+ concurrent providers):
          • Model is a singleton: loaded once, shared across all sessions.
          • BoundedSemaphore(_PUNCT_MAX_WORKERS=4): at most 4 concurrent CPU calls.
          • If the semaphore is saturated, this returns the raw text immediately
            (0 ms) — graceful degradation, never blocks the live stream.
          • asyncio.to_thread keeps the event loop free during inference.
          • Any unexpected exception also degrades gracefully.

        Capacity: 4 workers × ~12 calls/s ≈ 48 punctuated segments/s.
        At 100 providers × 1 chunk/3 s ≈ 33 chunks/s — comfortable headroom.
        """
        if not _PUNCT_MODEL_AVAILABLE or not text:
            return text

        def _run_in_thread() -> str:
            # Non-blocking semaphore acquire — skip if all workers are busy.
            if not _punct_semaphore.acquire(blocking=False):
                return text
            try:
                model = _get_punct_model()
                if model is None:
                    return text
                punctuated: str = model.restore_punctuation(text)  # type: ignore[attr-defined]
                return punctuated
            except Exception:  # pragma: no cover — degradation path
                return text
            finally:
                _punct_semaphore.release()

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_in_thread),
                timeout=0.35,   # 350 ms hard cap; never delays live stream > 1 frame
            )
        except (asyncio.TimeoutError, Exception):
            return text  # always degrade gracefully

    @staticmethod
    def _capitalize_sentences(text: str) -> str:
        """Capitalize the first letter after sentence-ending punctuation (.!?).

        Runs in O(n) with zero external dependencies and zero latency.
        Called after _punctuate_text so that punctuation is already in place.

        Rules:
          - The very first non-space character is always capitalized.
          - After a sentence-ending character (.!?) followed by one or more
            spaces, the next letter is capitalized.
          - Digits, non-letter characters, and existing uppercase are unchanged.

        Examples:
            "the patient presents today. she reports pain." →
            "The patient presents today. She reports pain."
        """
        if not text:
            return text

        chars = list(text)
        capitalize_next = True
        for i, ch in enumerate(chars):
            if ch in ".!?":
                capitalize_next = True
            elif capitalize_next and ch.isalpha():
                chars[i] = ch.upper()
                capitalize_next = False
            elif not ch.isspace():
                if capitalize_next and ch.isalpha():
                    chars[i] = ch.upper()
                capitalize_next = False
        return "".join(chars)

    @staticmethod
    def _normalize_section_headers(text: str) -> str:
        """Uppercase recognized medical note section headers.

        When a provider dictates a section name (e.g. "history of present illness")
        it should appear ALL CAPS in the transcript so the frontend can render it
        as a heading.  Matching is case-insensitive.

        Examples:
            "history of present illness"  → "HISTORY OF PRESENT ILLNESS"
            "Physical Examination"        → "PHYSICAL EXAMINATION"
            "past medical history"        → "PAST MEDICAL HISTORY"
            "assessment"                  → "ASSESSMENT"
        """
        for pattern, replacement in _SECTION_HEADER_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _normalize_segment_text(self, text: str) -> str:
        """Final normalization pass on a transcribed segment.

        Pipeline (in order):
          1. wordninja: split merged/camelCase/all-lowercase tokens
             ("historyOf" → "history Of", "todaypostconcussive" → "today postconcussive")
          2. ALL-CAPS stopwords: lowercase NeMo-raised function words ("THE" → "the")
          3. Spoken-date conversion: "october twenty fifth twenty twenty four" → "October 25, 2024"
          4. Number words → digits: "twenty seven year old" → "27-year-old"
          5. Numeric-date formatting: "24 2 2026" → "24/2/2026"
          6. Collapse double-spaces.

        Punctuation restoration and sentence capitalisation are performed
        separately (in _punctuate_text and _capitalize_sentences) so they
        can run async with bounded concurrency.
        """
        if not text:
            return text
        text = self._split_merged_words(text)
        text = self._normalize_allcaps_stopwords(text)
        text = self._normalize_spoken_dates(text)       # dates first (before numbers corrupt them)
        text = self._convert_medical_numbers(text)
        text = self._normalize_numeric_date(text)
        text = re.sub(r" {2,}", " ", text).strip()
        return text

    # ── Streaming transcription ──────────────────────────────────────────

    # STREAM_WINDOW_S is now set as an instance variable in __init__.
    # Default is 3.0s — see __init__ docstring for latency/quality tradeoff.

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

            # Apply medical hotword corrections (includes phonetic alias corrections)
            extra_hw = (config.hotwords if config is not None else None) or None
            text = self._apply_hotword_corrections(text, extra_hotwords=extra_hw)

            # Normalize spacing and convert spoken dates / numbers to written form
            text = self._normalize_segment_text(text)

            # Restore punctuation & capitalisation (async, bounded, graceful degradation)
            text = await self._punctuate_text(text)

            # Capitalize sentence starts after punctuation is restored (free, 0 ms)
            text = self._capitalize_sentences(text)

            # Uppercase medical note section headers (free, 0 ms)
            text = self._normalize_section_headers(text)

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