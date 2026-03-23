"""
mcp_servers/asr/whisperx_lora_server.py

Provider-specific Whisper ASR server with LoRA adapter.

Extends WhisperXServer to load a PEFT LoRA adapter trained by
scripts/finetune_whisper_lora.py.  The base model (whisper-large-v3) is
shared across all providers in GPU memory; only the lightweight adapter
(<50 MB) is swapped per provider.

The engine registry in mcp_servers/registry.py automatically selects this
server when models/whisper_lora/{provider_id}/ exists — otherwise it falls
back to the base WhisperXServer.  Zero pipeline code changes required.

Usage (direct, for testing):
    from mcp_servers.asr.whisperx_lora_server import WhisperXLoRAServer
    server = WhisperXLoRAServer.for_provider("dr_faraz_rahman")
    result = server.transcribe_batch_sync(audio_path, config)

Requires:
    pip install peft transformers torch
    (Same environment as finetune_whisper_lora.py)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from mcp_servers.asr.base import ASRCapabilities, ASRConfig, RawTranscript
from mcp_servers.asr.whisperx_server import WhisperXServer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR     = ROOT / "models" / "whisper_lora"
CT2_MODELS_DIR = ROOT / "models" / "whisper_ct2"


def adapter_exists(provider_id: str) -> bool:
    """Return True if a trained LoRA adapter exists for this provider."""
    adapter_path = MODELS_DIR / provider_id / "adapter_model.safetensors"
    return adapter_path.exists()


def ct2_export_exists(provider_id: str) -> bool:
    """Return True if a CTranslate2 export of the merged model exists.

    Prefer this over the HF PEFT path — it uses the full faster-whisper
    pipeline (beam search, temperature fallback, VAD, proper punctuation).
    Export via: python scripts/export_lora_ct2.py --provider {provider_id}
    """
    ct2_path = CT2_MODELS_DIR / provider_id / "model.bin"
    return ct2_path.exists()


class WhisperXLoRAServer(WhisperXServer):
    """
    WhisperX ASR engine with a provider-specific LoRA adapter loaded on top
    of the frozen whisper-large-v3 base model.

    The base model weights are loaded once (shared); the LoRA adapter adds
    ~40-60 MB of delta weights that bias the decoder toward the provider's
    specific vocabulary, pronunciation, and dictation style.

    Initialization flow:
        1. Parent __init__ sets model_size, device, compute_type etc.
        2. _load_lora_model() replaces self._model with a PEFT-wrapped model.
        3. transcribe_batch() calls the PEFT model via the standard faster-whisper
           pipeline — no code changes needed in WhisperXServer.transcribe_batch.

    Fallback:
        If the LoRA adapter fails to load (PEFT not installed, corrupt weights,
        wrong base model), logs a warning and falls back to the base WhisperXServer
        behaviour automatically.
    """

    def __init__(
        self,
        provider_id: str,
        adapter_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.provider_id = provider_id
        self._adapter_path = adapter_path or str(MODELS_DIR / provider_id)
        self._lora_loaded = False
        self._lora_failed = False

    @classmethod
    def for_provider(
        cls,
        provider_id: str,
        device: str = "cuda",
        compute_type: str = "float16",
        **kwargs: Any,
    ) -> "WhisperXLoRAServer":
        """Convenience constructor that resolves adapter path automatically."""
        return cls(
            provider_id=provider_id,
            device=device,
            compute_type=compute_type,
            **kwargs,
        )

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Override: load LoRA-enhanced model, preferring CTranslate2 export.

        Loading priority:
          1. CTranslate2 export (models/whisper_ct2/{provider_id}/) — uses the
             full faster-whisper pipeline with beam search, temperature fallback,
             VAD segmentation, and proper capitalization/punctuation.
             Generate with: python scripts/export_lora_ct2.py --provider {id}
          2. HuggingFace PEFT model — fallback when CT2 export doesn't exist.
             Inference quality is slightly lower (simpler decoding path).
          3. Base WhisperX — final fallback if both LoRA paths fail.
        """
        if self._model is not None:
            return

        if not self._lora_failed:
            try:
                # Prefer CTranslate2 export for full faster-whisper quality
                if ct2_export_exists(self.provider_id):
                    self._load_ct2_model()
                else:
                    logger.info(
                        "whisperx_lora: no CT2 export found for '%s' — "
                        "using HF PEFT path (run export_lora_ct2.py for better quality)",
                        self.provider_id,
                    )
                    self._load_lora_model()
                return
            except Exception as exc:
                logger.warning(
                    "whisperx_lora: LoRA adapter load failed (%s) — "
                    "falling back to base WhisperX",
                    exc,
                )
                self._lora_failed = True

        # Fallback: standard WhisperX
        super()._load_model()

    def _load_ct2_model(self) -> None:
        """Load the CTranslate2-exported merged model via whisperx.load_model().

        Passing the local CT2 directory as the model name gives the full
        faster-whisper pipeline (VAD, beam search, temperature fallback,
        capitalization) — identical to the base WhisperX setup.
        """
        import whisperx

        ct2_path = CT2_MODELS_DIR / self.provider_id

        logger.info(
            "whisperx_lora: loading CTranslate2 model for provider '%s' from %s",
            self.provider_id, ct2_path,
        )

        # whisperx.load_model() accepts a local directory path as model_name
        # when it contains a CTranslate2 model.bin — same as passing "large-v3"
        # except it loads our LoRA-merged weights instead of the vanilla base.
        self._model = whisperx.load_model(
            str(ct2_path),
            self.device,
            compute_type=self.compute_type,
            language=self.language,
        )
        self._lora_loaded = True

        logger.info(
            "whisperx_lora: CTranslate2 model loaded (provider=%s, device=%s)",
            self.provider_id, self.device,
        )

    def _load_lora_model(self) -> None:
        """Load whisper-large-v3 via HuggingFace transformers + PEFT adapter."""
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from peft import PeftModel

        adapter_path = Path(self._adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_path}. "
                f"Run scripts/finetune_whisper_lora.py --provider {self.provider_id}"
            )

        logger.info(
            "whisperx_lora: loading LoRA adapter for provider '%s' from %s",
            self.provider_id, adapter_path,
        )

        device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Load base model
        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3",
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        # Load LoRA adapter
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        peft_model.eval()

        # Load processor (tokenizer + feature extractor)
        processor = WhisperProcessor.from_pretrained(
            str(adapter_path), language="en", task="transcribe"
        )

        # Wrap in a thin adapter that mimics the faster-whisper API
        # expected by the parent's transcribe_batch
        self._model = _HFWhisperAdapter(peft_model, processor, device, dtype)
        self._lora_loaded = True

        logger.info(
            "whisperx_lora: LoRA adapter loaded (provider=%s, device=%s)",
            self.provider_id, device,
        )

    # ── Capabilities ─────────────────────────────────────────────────────────

    async def get_capabilities(self) -> ASRCapabilities:
        caps = await super().get_capabilities()
        # Expose LoRA as a capability tag via the model name
        caps.medical_vocab = True  # LoRA adapter encodes medical vocabulary
        return caps

    @property
    def name(self) -> str:
        return f"whisperx_lora/{self.provider_id}"


class _HFWhisperAdapter:
    """
    Thin wrapper around HuggingFace WhisperForConditionalGeneration that
    exposes the `transcribe(audio, **kwargs)` API expected by the parent
    WhisperXServer._model.transcribe() call.

    This allows WhisperXServer.transcribe_batch to work unchanged with either
    the CTranslate2 faster-whisper model or the HuggingFace PEFT model.
    """

    def __init__(self, model: Any, processor: Any, device: str, dtype: Any) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype

    def transcribe(self, audio: Any, batch_size: int = 8, language: str = "en",
                   initial_prompt: Optional[str] = None, **kwargs: Any) -> dict:
        """
        Transcribe a numpy audio array using the PEFT-LoRA Whisper model.

        Returns a dict compatible with faster-whisper output:
            {"segments": [...], "language": "en"}

        Each segment: {"text": str, "start": float, "end": float, "words": [...]}
        """
        import torch
        import numpy as np

        # WhisperX loads audio as float32 numpy at 16kHz
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        # Process in chunks of ~30s (Whisper's native context window)
        chunk_samples = 30 * 16000
        segments_out: list[dict] = []
        offset = 0.0

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            inputs = self._processor.feature_extractor(
                chunk, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(
                self._device, dtype=self._dtype
            )

            with torch.no_grad():
                forced_ids = self._processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
                generated_ids = self._model.generate(
                    input_features=input_features,
                    forced_decoder_ids=forced_ids,
                    max_new_tokens=225,
                    num_beams=5,            # match faster-whisper default beam size
                    no_repeat_ngram_size=3, # suppress repetition common in greedy decode
                )

            transcription = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            chunk_duration = len(chunk) / 16000
            if transcription:
                segments_out.append({
                    "text": transcription,
                    "start": offset,
                    "end": offset + chunk_duration,
                    "words": [],  # word-level alignment done by WhisperX align step
                })

            offset += chunk_duration

        return {"segments": segments_out, "language": language}
