"""
MedASR ML Post-Processor: Evaluation Framework
================================================
Compares ML model output against rule-based baseline using
medical transcription quality metrics.

Metrics:
  1. Word Error Rate (WER) — standard ASR metric
  2. Medical Term Accuracy — correct spelling of drug/anatomy terms
  3. Stutter Residual Rate — CTC artifacts remaining after cleanup
  4. Hallucination Rate — words introduced that weren't in the source
  5. Section Header Preservation — structural markers kept intact

Usage:
  python evaluate.py \
    --raw-dir /path/to/raw/ \
    --ml-dir /path/to/ml_cleaned/ \
    --rules-dir /path/to/rules_cleaned/ \
    --gold-dir /path/to/human_corrected/ \   # optional gold standard
    --report evaluation_report.txt
"""

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ============================================================
# METRICS
# ============================================================

@dataclass
class TranscriptMetrics:
    """Quality metrics for a single transcript."""
    file_id: str = ''
    word_count: int = 0
    
    # Stutter artifacts remaining
    stutter_pairs: int = 0
    char_stutters: int = 0
    stutter_rate_per_100w: float = 0.0
    
    # Vocabulary quality
    non_dict_words: int = 0
    non_dict_rate: float = 0.0
    
    # System artifacts remaining
    unintelligible_markers: int = 0
    format_commands: int = 0
    filler_markers: int = 0
    section_headers_preserved: int = 0
    
    # Hallucination check (words in output not traceable to input)
    hallucinated_words: int = 0
    
    # WER vs gold (if available)
    wer: float = -1.0


def load_dictionary(medical_wordlist_path: Optional[str] = None) -> Set[str]:
    """Load combined English + medical dictionary."""
    words = set()
    
    dict_paths = ['/usr/share/dict/words', '/usr/share/dict/american-english']
    for p in dict_paths:
        if os.path.exists(p):
            with open(p) as f:
                words.update(w.strip().lower() for w in f if len(w.strip()) >= 2)
            break
    
    # Medical wordlist
    med_paths = [
        medical_wordlist_path,
        'medical_wordlist.txt',
        'wordlist-medicalterms-en/wordlist.txt',
    ]
    for p in med_paths:
        if p and os.path.exists(p):
            with open(p) as f:
                words.update(w.strip().lower() for w in f if w.strip())
            break
    
    return words


def count_stutter_pairs(text: str) -> int:
    """Count word-level stutter pairs (fragment + full word)."""
    words = text.split()
    count = 0
    for i in range(len(words) - 1):
        w1 = re.sub(r'[^a-z]', '', words[i].lower())
        w2 = re.sub(r'[^a-z]', '', words[i + 1].lower())
        if len(w1) >= 2 and len(w2) >= 4 and w2.startswith(w1) and len(w2) > len(w1):
            count += 1
    return count


def count_char_stutters(text: str) -> int:
    """Count words with internal character-level stuttering."""
    count = 0
    for word in text.split():
        w = re.sub(r'[^a-z]', '', word.lower())
        if len(w) < 5:
            continue
        for i in range(len(w) - 3):
            if w[i:i + 2] == w[i + 2:i + 4] and w[i:i + 2].isalpha():
                count += 1
                break
    return count


def count_non_dict_words(text: str, dictionary: Set[str]) -> int:
    """Count words not found in the combined dictionary."""
    count = 0
    for w in text.split():
        clean = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', w).lower()
        if len(clean) >= 4 and clean not in dictionary and not clean.startswith('['):
            count += 1
    return count


def count_hallucinations(source: str, output: str) -> int:
    """
    Count words in output that can't be traced to the source input.
    
    A "hallucination" is a word in the output that:
      - Isn't in the source (even approximately)
      - Isn't a common English word (the/a/is/etc.)
      - Isn't a section header or formatting artifact
    
    This catches the dangerous case where an ML model invents content.
    """
    COMMON_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
        'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
        'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'it', 'its', 'he', 'she', 'his', 'her', 'they', 'them', 'their',
        'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your',
    }
    
    source_words = set(re.sub(r'[^a-z\s]', '', source.lower()).split())
    output_words = re.sub(r'[^a-z\s]', '', output.lower()).split()
    
    # Build character n-gram index of source for fuzzy matching
    source_trigrams = set()
    for w in source_words:
        for i in range(len(w) - 2):
            source_trigrams.add(w[i:i + 3])
    
    hallucinations = 0
    for w in output_words:
        if len(w) < 4:
            continue
        if w in source_words or w in COMMON_WORDS:
            continue
        # Check if word shares enough trigrams with source
        word_trigrams = set(w[i:i + 3] for i in range(len(w) - 2))
        overlap = len(word_trigrams & source_trigrams) / max(len(word_trigrams), 1)
        if overlap < 0.3:
            hallucinations += 1
    
    return hallucinations


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (standard ASR metric).
    WER = (substitutions + insertions + deletions) / reference_length
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Dynamic programming edit distance
    n = len(ref_words)
    m = len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    
    return d[n][m] / max(n, 1)


def evaluate_transcript(text: str, source: str, dictionary: Set[str],
                        gold: Optional[str] = None, file_id: str = '') -> TranscriptMetrics:
    """Compute all quality metrics for a single transcript."""
    words = text.split()
    word_count = len(words)
    
    stutters = count_stutter_pairs(text)
    char_stuts = count_char_stutters(text)
    non_dict = count_non_dict_words(text, dictionary)
    
    m = TranscriptMetrics(
        file_id=file_id,
        word_count=word_count,
        stutter_pairs=stutters,
        char_stutters=char_stuts,
        stutter_rate_per_100w=round(stutters / max(word_count, 1) * 100, 2),
        non_dict_words=non_dict,
        non_dict_rate=round(non_dict / max(word_count, 1) * 100, 2),
        unintelligible_markers=len(re.findall(
            r'(?:unintel|untel|intellig)', text, re.IGNORECASE)),
        format_commands=len(re.findall(r'\{[^}]*\}', text)),
        filler_markers=text.lower().count('[uh]') + text.lower().count('[um]')
                       + text.lower().count('[[uh]]'),
        section_headers_preserved=len(re.findall(r'\[[A-Z][A-Z\s]+\]', text)),
        hallucinated_words=count_hallucinations(source, text),
    )
    
    if gold:
        m.wer = round(compute_wer(gold, text), 4)
    
    return m


# ============================================================
# COMPARATIVE EVALUATION
# ============================================================

def compare_pipelines(raw_dir: str, ml_dir: str, rules_dir: str,
                      gold_dir: Optional[str] = None,
                      medical_wordlist: Optional[str] = None) -> str:
    """
    Run full comparative evaluation between ML and rule-based pipelines.
    Returns formatted report string.
    """
    dictionary = load_dictionary(medical_wordlist)
    
    raw_files = sorted(Path(raw_dir).glob("*.txt"))
    
    # Aggregate metrics
    raw_metrics = []
    ml_metrics = []
    rules_metrics = []
    
    for raw_file in raw_files:
        base = raw_file.stem
        raw_text = raw_file.read_text(errors='replace')
        
        # Find corresponding ML output
        ml_candidates = list(Path(ml_dir).glob(f"{base}*"))
        if not ml_candidates:
            continue
        ml_text = ml_candidates[0].read_text(errors='replace')
        
        # Find corresponding rules output
        rules_candidates = list(Path(rules_dir).glob(f"{base}*"))
        if not rules_candidates:
            continue
        rules_text = rules_candidates[0].read_text(errors='replace')
        
        # Gold standard (optional)
        gold_text = None
        if gold_dir:
            gold_candidates = list(Path(gold_dir).glob(f"{base}*"))
            if gold_candidates:
                gold_text = gold_candidates[0].read_text(errors='replace')
        
        # Evaluate all three
        rm = evaluate_transcript(raw_text, raw_text, dictionary,
                                 gold=gold_text, file_id=base)
        mm = evaluate_transcript(ml_text, raw_text, dictionary,
                                 gold=gold_text, file_id=base)
        rrm = evaluate_transcript(rules_text, raw_text, dictionary,
                                  gold=gold_text, file_id=base)
        
        raw_metrics.append(rm)
        ml_metrics.append(mm)
        rules_metrics.append(rrm)
    
    # Generate report
    return _format_comparison_report(raw_metrics, ml_metrics, rules_metrics)


def _format_comparison_report(raw: List[TranscriptMetrics],
                              ml: List[TranscriptMetrics],
                              rules: List[TranscriptMetrics]) -> str:
    """Format a comparison report between pipelines."""
    
    def avg(metrics, attr):
        vals = [getattr(m, attr) for m in metrics]
        return sum(vals) / max(len(vals), 1)
    
    def total(metrics, attr):
        return sum(getattr(m, attr) for m in metrics)
    
    n = len(raw)
    
    lines = [
        "=" * 74,
        "  MedASR POST-PROCESSOR: ML vs RULES COMPARISON",
        f"  {n} transcripts evaluated",
        "=" * 74,
        "",
        f"{'METRIC':<44s}  {'RAW':>8s}  {'RULES':>8s}  {'ML':>8s}  {'WINNER':>8s}",
        "-" * 74,
    ]
    
    comparisons = [
        ('Total words', 'word_count', total, 'neither'),
        ('Stutter pairs', 'stutter_pairs', total, 'lower'),
        ('Stutter rate (/100w)', 'stutter_rate_per_100w', avg, 'lower'),
        ('Char-level stutters', 'char_stutters', total, 'lower'),
        ('Non-dictionary words', 'non_dict_words', total, 'lower'),
        ('Non-dict rate (%)', 'non_dict_rate', avg, 'lower'),
        ('[unintelligible] markers', 'unintelligible_markers', total, 'lower'),
        ('{format commands}', 'format_commands', total, 'lower'),
        ('Filler markers [uh]/[um]', 'filler_markers', total, 'lower'),
        ('Section headers preserved', 'section_headers_preserved', total, 'higher'),
        ('Hallucinated words', 'hallucinated_words', total, 'lower'),
    ]
    
    ml_wins = 0
    rules_wins = 0
    
    for label, attr, agg_fn, direction in comparisons:
        raw_val = agg_fn(raw, attr)
        rules_val = agg_fn(rules, attr)
        ml_val = agg_fn(ml, attr)
        
        if direction == 'lower':
            winner = 'ML' if ml_val < rules_val else ('RULES' if rules_val < ml_val else 'TIE')
        elif direction == 'higher':
            winner = 'ML' if ml_val > rules_val else ('RULES' if rules_val > ml_val else 'TIE')
        else:
            winner = '—'
        
        if winner == 'ML':
            ml_wins += 1
        elif winner == 'RULES':
            rules_wins += 1
        
        # Format values
        if isinstance(raw_val, float):
            lines.append(f"{label:<44s}  {raw_val:>8.1f}  {rules_val:>8.1f}  {ml_val:>8.1f}  {winner:>8s}")
        else:
            lines.append(f"{label:<44s}  {raw_val:>8}  {rules_val:>8}  {ml_val:>8}  {winner:>8s}")
    
    # WER (only if gold standard available)
    if any(m.wer >= 0 for m in ml):
        ml_wer = avg([m for m in ml if m.wer >= 0], 'wer')
        rules_wer = avg([m for m in rules if m.wer >= 0], 'wer')
        raw_wer = avg([m for m in raw if m.wer >= 0], 'wer')
        winner = 'ML' if ml_wer < rules_wer else 'RULES'
        lines.append(f"{'WER vs gold standard':<44s}  {raw_wer:>8.3f}  {rules_wer:>8.3f}  {ml_wer:>8.3f}  {winner:>8s}")
    
    lines.extend([
        "",
        "-" * 74,
        f"  SCORECARD: ML wins {ml_wins}, Rules wins {rules_wins}, "
        f"Ties {len(comparisons) - ml_wins - rules_wins}",
        "",
    ])
    
    # Safety check
    ml_hallucinations = total(ml, 'hallucinated_words')
    if ml_hallucinations > 0:
        lines.extend([
            "  ⚠️  WARNING: ML model hallucinated words in output!",
            f"  Total hallucinated words: {ml_hallucinations}",
            "  This must reach 0 before deploying ML as primary pipeline.",
            "  Consider: stricter beam search, constrained decoding,",
            "  or hybrid mode with rules fallback.",
            "",
        ])
    
    # Recommendation
    lines.extend([
        "=" * 74,
        "  RECOMMENDATION",
        "=" * 74,
    ])
    
    if ml_hallucinations > 0:
        lines.append("  Use HYBRID mode (ML + rules fallback) until hallucinations reach 0.")
    elif ml_wins > rules_wins:
        lines.append("  ML model outperforms rules. Safe to deploy as primary with monitoring.")
    elif rules_wins > ml_wins:
        lines.append("  Rules still outperform ML. Continue collecting training data.")
        lines.append("  Consider: more augmentation, larger model, or domain-specific pre-training.")
    else:
        lines.append("  Performance is comparable. Use hybrid mode for safety.")
    
    lines.append("")
    return '\n'.join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ML vs rule-based pipeline")
    parser.add_argument('--raw-dir', required=True, help='Directory of raw transcripts')
    parser.add_argument('--ml-dir', required=True, help='Directory of ML-cleaned output')
    parser.add_argument('--rules-dir', required=True, help='Directory of rules-cleaned output')
    parser.add_argument('--gold-dir', help='Directory of human-corrected gold standard (optional)')
    parser.add_argument('--medical-wordlist', help='Path to medical wordlist')
    parser.add_argument('--report', default='comparison_report.txt', help='Output report path')
    args = parser.parse_args()
    
    report = compare_pipelines(
        raw_dir=args.raw_dir,
        ml_dir=args.ml_dir,
        rules_dir=args.rules_dir,
        gold_dir=args.gold_dir,
        medical_wordlist=args.medical_wordlist,
    )
    
    print(report)
    
    with open(args.report, 'w') as f:
        f.write(report)
    log.info(f"Report saved to {args.report}")


if __name__ == '__main__':
    main()
