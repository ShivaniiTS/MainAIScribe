"""
MedASR ML Post-Processor: Model Training
==========================================
Fine-tunes a seq2seq model (T5 or BART) to correct CTC decoder
artifacts in medical transcripts.

Architecture choices:
  - T5-small (60M params): Fast training, good for CTC pattern learning
  - T5-base (220M params): Better accuracy, still trainable on single GPU
  - BioBART (base): Pre-trained on biomedical text, best for medical terms
  - Custom: ByT5 (byte-level) — directly models character-level patterns

The model learns the mapping: garbled_text → clean_text
using (source, target) pairs from generate_training_data.py.

Usage:
  # Initial training from rule-based pipeline output
  python train_model.py \
    --training-data training_pairs.jsonl \
    --model-name google/t5-small \
    --output-dir ./medasr_model/ \
    --epochs 10 --batch-size 16

  # Continuous learning with new human-corrected data
  python train_model.py \
    --training-data new_corrections.jsonl \
    --model-name ./medasr_model/ \
    --output-dir ./medasr_model_v2/ \
    --epochs 3 --learning-rate 1e-5

Requirements:
  pip install torch transformers datasets accelerate
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ============================================================
# CONFIGURATION
# ============================================================

# Recommended model configs for different hardware profiles
MODEL_CONFIGS = {
    'tiny': {
        'name': 'google/t5-small',      # 60M params
        'max_source_len': 256,
        'max_target_len': 256,
        'batch_size': 32,
        'epochs': 15,
        'learning_rate': 3e-4,
        'description': 'Fast training, single GPU, good baseline',
    },
    'base': {
        'name': 'google/t5-v1_1-base',  # 250M params
        'max_source_len': 512,
        'max_target_len': 512,
        'batch_size': 8,
        'epochs': 10,
        'learning_rate': 1e-4,
        'description': 'Best balance of speed and accuracy',
    },
    'medical': {
        'name': 'GanjinZero/biobart-v2-base',  # 140M params, biomedical pre-training
        'max_source_len': 512,
        'max_target_len': 512,
        'batch_size': 8,
        'epochs': 10,
        'learning_rate': 5e-5,
        'description': 'Best for medical terminology correction',
    },
    'byte': {
        'name': 'google/byt5-small',     # 300M params, byte-level
        'max_source_len': 1024,           # byte-level needs more tokens
        'max_target_len': 1024,
        'batch_size': 4,
        'epochs': 8,
        'learning_rate': 1e-4,
        'description': 'Directly models character patterns — ideal for CTC stutters',
    },
}


# ============================================================
# DATA LOADING
# ============================================================

def load_training_data(path: str) -> List[Dict]:
    """Load JSONL training pairs."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    log.info(f"Loaded {len(pairs)} training pairs from {path}")
    return pairs


def prepare_datasets(pairs: List[Dict], val_fraction: float = 0.1,
                     task_prefix: str = "correct: "):
    """
    Prepare HuggingFace datasets from training pairs.
    
    The task_prefix tells T5 what operation to perform.
    For CTC correction we use "correct: " as the prefix.
    """
    from datasets import Dataset
    import random
    
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - val_fraction))
    
    train_data = pairs[:split_idx]
    val_data = pairs[split_idx:]
    
    def format_examples(examples):
        return {
            'source': [task_prefix + ex['source'] for ex in examples],
            'target': [ex['target'] for ex in examples],
        }
    
    train_formatted = format_examples(train_data)
    val_formatted = format_examples(val_data)
    
    train_ds = Dataset.from_dict(train_formatted)
    val_ds = Dataset.from_dict(val_formatted)
    
    log.info(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")
    return train_ds, val_ds


# ============================================================
# MODEL TRAINING
# ============================================================

def train(args):
    """Main training loop."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments, Seq2SeqTrainer,
        DataCollatorForSeq2Seq, EarlyStoppingCallback,
    )
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Device: {device}")
    if device == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load config
    config = MODEL_CONFIGS.get(args.config)
    model_name = args.model_name or config['name']
    max_source_len = args.max_source_len or config['max_source_len']
    max_target_len = args.max_target_len or config['max_target_len']
    batch_size = args.batch_size or config['batch_size']
    epochs = args.epochs or config['epochs']
    lr = args.learning_rate or config['learning_rate']
    
    log.info(f"Model: {model_name}")
    log.info(f"Config: {args.config} — {config['description']}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load and prepare data
    pairs = load_training_data(args.training_data)
    train_ds, val_ds = prepare_datasets(pairs, val_fraction=0.1)
    
    # Tokenize
    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples['source'],
            max_length=max_source_len,
            truncation=True,
            padding='max_length',
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'],
                max_length=max_target_len,
                truncation=True,
                padding='max_length',
            )
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['source', 'target'])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=['source', 'target'])
    
    # Data collator (handles dynamic padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    
    # Training arguments
    output_dir = args.output_dir
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        
        # Evaluation
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        
        # Generation config for eval
        predict_with_generate=True,
        generation_max_length=max_target_len,
        
        # Performance
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        
        # Logging
        logging_steps=50,
        report_to='none',  # Change to 'wandb' for experiment tracking
    )
    
    # Custom metrics
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Character-level accuracy
        total_chars = 0
        correct_chars = 0
        exact_match = 0
        
        for pred, label in zip(decoded_preds, decoded_labels):
            pred = pred.strip()
            label = label.strip()
            
            if pred == label:
                exact_match += 1
            
            for p, l in zip(pred, label):
                total_chars += 1
                if p == l:
                    correct_chars += 1
            total_chars += abs(len(pred) - len(label))
        
        char_accuracy = correct_chars / max(total_chars, 1)
        exact_match_rate = exact_match / max(len(decoded_preds), 1)
        
        return {
            'char_accuracy': round(char_accuracy, 4),
            'exact_match': round(exact_match_rate, 4),
        }
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    log.info("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save model metadata
    metadata = {
        'model_name': model_name,
        'config': args.config,
        'training_pairs': len(pairs),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'max_source_len': max_source_len,
        'max_target_len': max_target_len,
    }
    with open(os.path.join(output_dir, 'medasr_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log.info(f"\nModel saved to {output_dir}")
    log.info("Done!")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train MedASR ML post-processor")
    parser.add_argument('--training-data', required=True, help='JSONL training pairs file')
    parser.add_argument('--output-dir', default='./medasr_model/', help='Output model directory')
    parser.add_argument('--config', choices=list(MODEL_CONFIGS.keys()), default='tiny',
                        help='Model configuration preset')
    parser.add_argument('--model-name', help='Override model name (for continuing training)')
    parser.add_argument('--max-source-len', type=int, help='Override max source length')
    parser.add_argument('--max-target-len', type=int, help='Override max target length')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
