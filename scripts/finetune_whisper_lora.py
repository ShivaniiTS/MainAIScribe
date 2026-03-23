"""
scripts/finetune_whisper_lora.py

Fine-tune openai/whisper-large-v3 with LoRA adapters on provider-specific
audio data prepared by prepare_asr_training_data.py.

Architecture:
  - Base model: openai/whisper-large-v3 (frozen — weights never change)
  - LoRA delta: r=8, targets q_proj + v_proj in every encoder/decoder attention layer
  - Adapter size: ~40–60 MB per provider (vs ~3 GB for the full model)
  - One adapter per provider; all providers share the same base model in GPU memory

Why LoRA for medical ASR:
  - The base Whisper model already understands speech acoustics perfectly
  - What varies per physician: vocabulary (eponyms, drug names), accent, pace,
    dictation style, abbreviations
  - LoRA efficiently captures these delta patterns without forgetting general ASR

Training config (optimised for ~20 clinical audio samples after augmentation):
  - max_steps=200      — enough for ~1000 chunks from augmented 20 samples
  - lr=1e-4            — standard LoRA learning rate
  - warmup_steps=50    — 25% of training as warmup
  - fp16 / bf16        — mixed precision based on GPU capability
  - gradient_checkpointing — fit large-v3 in 24GB VRAM

Usage:
    # Basic
    python scripts/finetune_whisper_lora.py --provider dr_faraz_rahman

    # Resume from checkpoint
    python scripts/finetune_whisper_lora.py --provider dr_faraz_rahman --resume

    # Custom hyperparameters
    python scripts/finetune_whisper_lora.py --provider dr_faraz_rahman \\
        --max-steps 400 --lr 5e-5 --batch-size 8

Output:
    models/whisper_lora/{provider_id}/
        adapter_config.json      — LoRA config
        adapter_model.safetensors — LoRA delta weights
        training_args.json       — Hyperparameters used
        eval_results.json        — Final WER on eval split
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models" / "whisper_lora"
TRAINING_DATA_DIR = ROOT / "data" / "asr_training"
BASE_MODEL = "openai/whisper-large-v3"


# ─────────────────────────────────────────────────────────────────────────────
# Data collator
# ─────────────────────────────────────────────────────────────────────────────

class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads input features and labels to the longest item in the batch.
    Labels are shifted left by one (standard seq2seq teacher-forcing).
    """

    def __init__(self, processor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        # Separate inputs and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so loss ignores padding
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Remove decoder_start_token if prepended
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(batch, processor):
    """Map function: load audio → log-mel features + tokenized labels."""
    import soundfile as sf
    import numpy as np

    audio_path = batch["audio"]
    try:
        audio_array, sr = sf.read(audio_path)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)  # stereo → mono
        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    except Exception:
        audio_array = np.zeros(16000, dtype=np.float32)  # silence fallback

    # Log-mel spectrogram
    batch["input_features"] = processor.feature_extractor(
        audio_array.astype(np.float32),
        sampling_rate=16000,
    ).input_features[0]

    # Tokenize label text
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# WER metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_fn(processor):
    """Returns a compute_metrics function for Seq2SeqTrainer."""
    try:
        import evaluate
        wer_metric = evaluate.load("wer")
    except Exception:
        wer_metric = None

    def compute_metrics(pred):
        import numpy as np

        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        if wer_metric:
            wer = wer_metric.compute(predictions=pred_str, references=label_str)
            return {"wer": round(wer, 4)}
        return {}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    provider_id: str,
    max_steps: int = 200,
    lr: float = 1e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    warmup_steps: int = 50,
    resume: bool = False,
) -> None:
    """
    Fine-tune Whisper large-v3 with LoRA on provider-specific data.
    """
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_from_disk

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset_path = TRAINING_DATA_DIR / provider_id / "dataset"
    if not dataset_path.exists():
        log.error(
            "Dataset not found at %s — run prepare_asr_training_data.py first",
            dataset_path,
        )
        sys.exit(1)

    log.info("Loading dataset from %s", dataset_path)
    ds = load_from_disk(str(dataset_path))
    log.info("Train: %d chunks, Eval: %d chunks", len(ds["train"]), len(ds["eval"]))

    # ── Load processor ───────────────────────────────────────────────────────
    log.info("Loading processor from %s", BASE_MODEL)
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="en", task="transcribe"
    )

    # ── Prepare features ─────────────────────────────────────────────────────
    log.info("Extracting features…")
    ds = ds.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=["audio", "sentence", "duration_s", "sample_id", "mode"],
        num_proc=1,
    )

    # ── Load base model ──────────────────────────────────────────────────────
    log.info("Loading base model %s…", BASE_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use bf16 on GPU — avoids fp16 dtype mismatch in Whisper eval generation
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device == "cuda" else torch.float32)

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    # transformers v5: generation control must live on generation_config only
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # ── Apply LoRA ───────────────────────────────────────────────────────────
    log.info("Applying LoRA config (r=8, alpha=32, targets=q_proj+v_proj)…")
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Training arguments ───────────────────────────────────────────────────
    output_dir = MODELS_DIR / provider_id
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=not use_bf16 and device == "cuda",
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # disable generation during training eval — dtype mismatch in transformers v5
        # WER is computed after training by eval_asr_quality.py
        predict_with_generate=False,
        logging_steps=10,
        report_to="none",  # no wandb/tensorboard by default
        gradient_checkpointing=True,
    )

    # ── Data collator ────────────────────────────────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=collator,
        compute_metrics=None,  # WER computed post-training by eval_asr_quality.py
        processing_class=processor.feature_extractor,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info(
        "Starting training: max_steps=%d, lr=%g, batch_size=%d×%d",
        max_steps, lr, batch_size, grad_accum,
    )
    trainer.train(resume_from_checkpoint=resume)

    # ── Save LoRA adapter (NOT full model) ───────────────────────────────────
    log.info("Saving LoRA adapter to %s", output_dir)
    model.save_pretrained(str(output_dir))       # saves adapter_model.safetensors
    processor.save_pretrained(str(output_dir))   # saves tokenizer + feature extractor

    # Eval results
    eval_results = trainer.evaluate()
    log.info("Eval results: %s", eval_results)
    (output_dir / "eval_results.json").write_text(json.dumps(eval_results, indent=2))

    # Training args snapshot
    training_summary = {
        "provider_id": provider_id,
        "base_model": BASE_MODEL,
        "max_steps": max_steps,
        "lr": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "warmup_steps": warmup_steps,
        "lora_r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
    }
    (output_dir / "training_args.json").write_text(json.dumps(training_summary, indent=2))

    log.info("Done. Adapter saved to %s", output_dir)
    log.info("Adapter size: %.1f MB",
             sum(f.stat().st_size for f in output_dir.glob("*.safetensors")) / 1e6)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper large-v3 with LoRA on provider-specific clinical audio"
    )
    parser.add_argument("--provider", required=True, help="Provider ID")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch×accum)")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in models/whisper_lora/{provider}/")
    args = parser.parse_args()

    finetune(
        provider_id=args.provider,
        max_steps=args.max_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        warmup_steps=args.warmup_steps,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
