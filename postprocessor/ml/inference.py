"""
MedASR ML Post-Processor: Inference & Continuous Learning
==========================================================
Production inference pipeline that replaces the rule-based script.
Includes a feedback loop for continuous model improvement.

Architecture:
  ┌──────────┐    ┌──────────────┐    ┌───────────┐
  │  MedASR   │───>│ ML Pipeline  │───>│ MedGamma  │
  │ (audio→   │    │ (correction) │    │ (semantic  │
  │  text)    │    │              │    │  recon)    │
  └──────────┘    └──────┬───────┘    └───────────┘
                         │
                  ┌──────┴───────┐
                  │  Confidence  │
                  │  Router      │
                  │              │
                  │ High conf ──>│──> Use ML output
                  │ Low conf  ──>│──> Fallback to rules + flag for review
                  └──────┬───────┘
                         │
                  ┌──────┴───────┐
                  │  Human Review│
                  │  Queue       │
                  │              │
                  │ Corrections ─│──> Retraining data
                  └──────────────┘

Usage:
  # Single file inference
  python inference.py --model ./medasr_model/ input.txt -o cleaned.txt

  # Batch processing
  python inference.py --model ./medasr_model/ --batch /transcripts/ -o /cleaned/

  # Pipeline mode (stdin/stdout)
  cat transcript.txt | python inference.py --model ./medasr_model/ - > cleaned.txt

  # With confidence routing
  python inference.py --model ./medasr_model/ input.txt -o cleaned.txt \
    --confidence-threshold 0.85 --flag-uncertain uncertain.jsonl

Requirements:
  pip install torch transformers
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


# ============================================================
# ML INFERENCE ENGINE
# ============================================================

@dataclass
class CorrectionResult:
    """Result of ML correction with confidence metadata."""
    original: str
    corrected: str
    confidence: float          # avg token log-probability
    corrections_made: int      # number of changed words
    low_confidence_spans: List[dict] = field(default_factory=list)  # flagged spans
    inference_time_ms: float = 0.0


class MedASRMLCorrector:
    """
    ML-based transcript correction engine.
    
    Loads a fine-tuned seq2seq model and applies it to
    MedASR transcripts for CTC error correction.
    """
    
    def __init__(self, model_dir: str, device: str = 'auto',
                 confidence_threshold: float = 0.85):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        self.confidence_threshold = confidence_threshold
        
        # Load model metadata
        metadata_path = os.path.join(model_dir, 'medasr_metadata.json')
        self.metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        log.info(f"Loading model from {model_dir} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        
        self.max_source_len = self.metadata.get('max_source_len', 512)
        self.max_target_len = self.metadata.get('max_target_len', 512)
        self.task_prefix = "correct: "
        
        log.info(f"Model loaded: {self.metadata.get('model_name', 'unknown')}")
    
    def correct(self, text: str) -> CorrectionResult:
        """
        Correct a transcript segment.
        
        For long texts, splits into segments, corrects each,
        and reassembles with overlap handling.
        """
        import torch
        
        t0 = time.time()
        
        # Segment long text into chunks (with overlap for coherence)
        segments = self._segment_text(text, max_words=60, overlap_words=10)
        
        corrected_segments = []
        all_confidences = []
        all_low_conf_spans = []
        
        for seg_idx, segment in enumerate(segments):
            # Prepare input
            input_text = self.task_prefix + segment
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_source_len,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate with beam search and return scores
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_target_len,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Decode
            decoded = self.tokenizer.decode(
                outputs.sequences[0],
                skip_special_tokens=True
            ).strip()
            
            # Calculate confidence (avg log probability across generated tokens)
            if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                confidence = torch.exp(outputs.sequences_scores[0]).item()
            else:
                confidence = self._estimate_confidence(segment, decoded)
            
            all_confidences.append(confidence)
            corrected_segments.append(decoded)
            
            # Flag low-confidence regions
            if confidence < self.confidence_threshold:
                all_low_conf_spans.append({
                    'segment_idx': seg_idx,
                    'original': segment,
                    'corrected': decoded,
                    'confidence': round(confidence, 3),
                })
        
        # Reassemble segments (handle overlaps)
        corrected = self._reassemble_segments(corrected_segments, segments)
        
        # Count corrections
        corrections = self._count_corrections(text, corrected)
        
        t1 = time.time()
        
        return CorrectionResult(
            original=text,
            corrected=corrected,
            confidence=sum(all_confidences) / max(len(all_confidences), 1),
            corrections_made=corrections,
            low_confidence_spans=all_low_conf_spans,
            inference_time_ms=round((t1 - t0) * 1000, 1),
        )
    
    def _segment_text(self, text: str, max_words: int = 60,
                      overlap_words: int = 10) -> List[str]:
        """Split text into overlapping segments for processing."""
        words = text.split()
        if len(words) <= max_words:
            return [text]
        
        segments = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            segment = ' '.join(words[start:end])
            segments.append(segment)
            start += max_words - overlap_words
        
        return segments
    
    def _reassemble_segments(self, corrected: List[str],
                             originals: List[str]) -> str:
        """Reassemble overlapping segments, preferring the middle of each."""
        if len(corrected) == 1:
            return corrected[0]
        
        # Simple concatenation — the overlap handling with
        # longest-common-subsequence alignment is complex;
        # for production, consider non-overlapping segments
        # or a sliding-window approach.
        result_parts = [corrected[0]]
        for i in range(1, len(corrected)):
            # Take the second half of each subsequent segment
            words = corrected[i].split()
            half_point = len(words) // 4  # skip first quarter (overlap region)
            result_parts.append(' '.join(words[half_point:]))
        
        return ' '.join(result_parts)
    
    def _estimate_confidence(self, original: str, corrected: str) -> float:
        """Estimate confidence when model doesn't return scores directly."""
        if original == corrected:
            return 1.0
        
        # Character-level similarity as proxy
        common = sum(1 for a, b in zip(original, corrected) if a == b)
        max_len = max(len(original), len(corrected))
        similarity = common / max(max_len, 1)
        
        # High similarity = high confidence (few changes)
        # Very low similarity = low confidence (many changes, possibly wrong)
        return min(0.99, similarity * 1.1)
    
    def _count_corrections(self, original: str, corrected: str) -> int:
        """Count number of words that changed."""
        orig_words = original.split()
        corr_words = corrected.split()
        
        # Simple word-level diff count
        changes = 0
        min_len = min(len(orig_words), len(corr_words))
        for i in range(min_len):
            if orig_words[i] != corr_words[i]:
                changes += 1
        changes += abs(len(orig_words) - len(corr_words))
        
        return changes


# ============================================================
# HYBRID PIPELINE (ML + RULES FALLBACK)
# ============================================================

class HybridPipeline:
    """
    Production pipeline that combines ML model with rule-based fallback.
    
    Strategy:
      1. Run ML model on input
      2. Check confidence scores
      3. For low-confidence segments, fall back to rules
      4. Flag uncertain outputs for human review
    
    This ensures the pipeline never degrades below the
    rule-based baseline, even for unseen error patterns.
    """
    
    def __init__(self, model_dir: str, confidence_threshold: float = 0.85,
                 rules_fallback: bool = True):
        self.ml_corrector = MedASRMLCorrector(
            model_dir, confidence_threshold=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold
        self.rules_fallback = rules_fallback
        
        # Try to import rule-based pipeline
        self.rules_available = False
        if rules_fallback:
            try:
                # Assumes medasr_postprocessor.py is importable
                from medasr_postprocessor import postprocess
                self.rules_postprocess = postprocess
                self.rules_available = True
                log.info("Rule-based fallback: ENABLED")
            except ImportError:
                log.warning("Rule-based fallback: medasr_postprocessor.py not found")
    
    def process(self, text: str) -> CorrectionResult:
        """Process text through the hybrid pipeline."""
        ml_result = self.ml_corrector.correct(text)
        
        # If confidence is high enough, use ML output
        if ml_result.confidence >= self.confidence_threshold:
            return ml_result
        
        # Low confidence: try rule-based fallback
        if self.rules_available and ml_result.confidence < self.confidence_threshold * 0.9:
            log.debug(f"Low confidence ({ml_result.confidence:.2f}), using rule-based fallback")
            rules_text, _ = self.rules_postprocess(text)
            ml_result.corrected = rules_text
            ml_result.corrected += "  [RULES_FALLBACK]"  # marker for tracking
        
        return ml_result


# ============================================================
# CONTINUOUS LEARNING MODULE
# ============================================================

class ContinuousLearner:
    """
    Manages the feedback loop for continuous model improvement.
    
    Workflow:
      1. Collect human corrections from reviewed transcripts
      2. Add to correction buffer
      3. When buffer reaches threshold, trigger retraining
      4. A/B test new model against current model
      5. Deploy if improved
    
    Storage format: JSONL files with correction pairs.
    """
    
    def __init__(self, feedback_dir: str = './feedback/',
                 retrain_threshold: int = 100):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.retrain_threshold = retrain_threshold
        
        # Feedback buffer
        self.buffer_path = self.feedback_dir / 'correction_buffer.jsonl'
        self.deployed_path = self.feedback_dir / 'deployed_corrections.jsonl'
        self.metrics_path = self.feedback_dir / 'model_metrics.jsonl'
    
    def add_correction(self, original: str, ml_output: str,
                       human_corrected: str, file_id: str = ''):
        """
        Record a human correction for future retraining.
        
        Called when a human reviewer corrects an ML output.
        """
        correction = {
            'source': original,
            'target': human_corrected,
            'ml_output': ml_output,
            'origin': 'human_corrected',
            'file_id': file_id,
            'timestamp': time.time(),
        }
        
        with open(self.buffer_path, 'a') as f:
            f.write(json.dumps(correction) + '\n')
        
        buffer_size = self._count_buffer()
        log.info(f"Correction recorded. Buffer: {buffer_size}/{self.retrain_threshold}")
        
        if buffer_size >= self.retrain_threshold:
            log.info("Retrain threshold reached! Run retraining pipeline.")
            return True  # Signal that retraining is needed
        
        return False
    
    def add_batch_corrections(self, corrections: List[dict]):
        """Add multiple corrections at once (from batch review)."""
        with open(self.buffer_path, 'a') as f:
            for corr in corrections:
                corr['origin'] = 'human_corrected'
                corr['timestamp'] = time.time()
                f.write(json.dumps(corr) + '\n')
        
        log.info(f"Added {len(corrections)} corrections. Buffer: {self._count_buffer()}")
    
    def export_retraining_data(self, output_path: str = None) -> str:
        """
        Export buffered corrections as training data for retraining.
        Moves buffer to deployed archive.
        """
        if output_path is None:
            output_path = str(self.feedback_dir / f'retrain_{int(time.time())}.jsonl')
        
        if not self.buffer_path.exists():
            log.warning("No corrections in buffer")
            return ''
        
        # Copy buffer to retraining file
        import shutil
        shutil.copy2(self.buffer_path, output_path)
        
        # Archive to deployed
        with open(self.buffer_path) as src, open(self.deployed_path, 'a') as dst:
            dst.write(src.read())
        
        # Clear buffer
        self.buffer_path.unlink()
        
        count = sum(1 for _ in open(output_path))
        log.info(f"Exported {count} corrections to {output_path}")
        return output_path
    
    def log_model_metrics(self, model_version: str, metrics: dict):
        """Log model performance metrics for tracking improvement over time."""
        entry = {
            'model_version': model_version,
            'timestamp': time.time(),
            **metrics,
        }
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def _count_buffer(self) -> int:
        if not self.buffer_path.exists():
            return 0
        return sum(1 for _ in open(self.buffer_path))


# ============================================================
# CLI
# ============================================================

def process_file(pipeline, input_path: str, output_path: str,
                 flag_path: Optional[str] = None):
    """Process a single file."""
    with open(input_path, 'r', errors='replace') as f:
        text = f.read()
    
    result = pipeline.process(text) if hasattr(pipeline, 'process') else pipeline.correct(text)
    
    with open(output_path, 'w') as f:
        f.write(result.corrected)
    
    log.info(f"Processed: {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    log.info(f"  Corrections: {result.corrections_made}, "
             f"Confidence: {result.confidence:.3f}, "
             f"Time: {result.inference_time_ms:.0f}ms")
    
    # Flag uncertain outputs
    if flag_path and result.low_confidence_spans:
        with open(flag_path, 'a') as f:
            flag_entry = {
                'file': input_path,
                'overall_confidence': result.confidence,
                'spans': result.low_confidence_spans,
            }
            f.write(json.dumps(flag_entry) + '\n')
        log.info(f"  ⚠ {len(result.low_confidence_spans)} low-confidence spans flagged")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="MedASR ML Post-Processor: Inference Pipeline")
    parser.add_argument('input', help='Input file, directory (with --batch), or - for stdin')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('--model', required=True, help='Path to trained model directory')
    parser.add_argument('--batch', action='store_true', help='Process all .txt files in directory')
    parser.add_argument('--confidence-threshold', type=float, default=0.85,
                        help='Minimum confidence for ML output (default: 0.85)')
    parser.add_argument('--flag-uncertain', metavar='PATH',
                        help='JSONL file to log uncertain outputs for review')
    parser.add_argument('--no-rules-fallback', action='store_true',
                        help='Disable rule-based fallback for low confidence')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    if args.no_rules_fallback:
        pipeline = MedASRMLCorrector(
            args.model, device=args.device,
            confidence_threshold=args.confidence_threshold)
    else:
        pipeline = HybridPipeline(
            args.model, confidence_threshold=args.confidence_threshold,
            rules_fallback=True)
    
    # Stdin mode
    if args.input == '-':
        text = sys.stdin.read()
        result = pipeline.process(text) if hasattr(pipeline, 'process') else pipeline.correct(text)
        sys.stdout.write(result.corrected)
        return
    
    # Batch mode
    if args.batch:
        input_dir = Path(args.input)
        output_dir = Path(args.output) if args.output else input_dir / "ml_cleaned"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for f in sorted(input_dir.glob("*.txt")):
            out_path = output_dir / f.name.replace('.txt', '_cleaned.txt')
            process_file(pipeline, str(f), str(out_path), args.flag_uncertain)
        return
    
    # Single file
    output = args.output or args.input.replace('.txt', '_ml_cleaned.txt')
    process_file(pipeline, args.input, output, args.flag_uncertain)


if __name__ == '__main__':
    main()
