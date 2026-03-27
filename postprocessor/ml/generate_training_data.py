"""
MedASR ML Post-Processor: Training Data Generator
====================================================
Generates aligned (noisy, clean) training pairs from:
  1. Real transcripts processed through the rule-based pipeline
  2. Synthetic CTC error injection into clean medical text
  3. Hard-example mining from residual errors

This is the BOOTSTRAP phase — the rule-based pipeline acts as
the "teacher" to train the initial ML model. Once the ML model
is live, human corrections feed back as gold-standard data.

Usage:
  python generate_training_data.py \
    --raw-dir /path/to/raw_transcripts/ \
    --clean-dir /path/to/cleaned_transcripts/ \
    --medical-corpus /path/to/medical_text.txt \
    --output training_pairs.jsonl \
    --augmentation-factor 10
"""

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ============================================================
# CTC ERROR SIMULATOR
# ============================================================
# Simulates the error patterns we've observed from MedASR's
# CTC decoder. These are NOT random noise — they follow
# specific mechanical patterns from beam search failures.

@dataclass
class TrainingPair:
    """A single (noisy, clean) training example."""
    source: str       # noisy / garbled text
    target: str       # clean / corrected text
    origin: str       # 'real', 'synthetic', 'hard_example', 'human_corrected'
    file_id: str = '' # source file identifier


class CTCErrorSimulator:
    """
    Simulates MedASR CTC decoder errors based on observed patterns.
    
    Error types (with observed frequencies from 10 transcripts):
      1. Stutter pairs:     10.1/100 words — fragment + full word
      2. Char doubling:      16.3/100 words — internal character repeat
      3. Format cmd doubles:  0.3/100 words — {periodperiod}
      4. Filler insertion:    1.5/100 words — "u h", "uuhhm"
      5. Punctuation doubles: 2.5/100 words — ".." ",,"
      6. Word splitting:      1.0/100 words — "fol lowup"
      7. Section header garbling: per header
    """
    
    # Observed stutter prefixes (most common fragment lengths)
    STUTTER_PREFIX_LENGTHS = [1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6]
    
    # Filler variants with their relative frequencies
    FILLER_VARIANTS = [
        'u h', 'u h', 'u h',        # most common
        'u hh', 'u hh',
        'U h', 'Uhh',
        'uuhhm', 'uhm',
        'U hh m',
    ]
    
    # Format commands that get doubled
    FORMAT_COMMANDS = [
        '{period}', '{comma}', '{new line}', '{new paragraph}',
        '{colon}', '{slash}', '{question mark}',
    ]
    
    # Character doubling patterns (consonants more often than vowels)
    DOUBLE_CHARS = 'bcdfghjklmnpqrstvwxyz'  # consonants
    DOUBLE_VOWELS = 'aeiouy'                  # vowels (less frequent)
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def inject_errors(self, clean_text: str,
                      stutter_rate: float = 0.10,
                      char_double_rate: float = 0.16,
                      filler_rate: float = 0.015,
                      punct_double_rate: float = 0.025,
                      word_split_rate: float = 0.01,
                      format_cmd_rate: float = 0.003) -> str:
        """
        Inject realistic CTC errors into clean text.
        
        Rates are per-word probabilities matching observed corpus statistics.
        Returns the garbled text.
        """
        words = clean_text.split()
        result = []
        
        for i, word in enumerate(words):
            # Skip very short words and markers
            if len(word) < 3 or word.startswith('[') or word.startswith('{'):
                result.append(word)
                continue
            
            modified = word
            
            # 1. Stutter pair: emit fragment then full word
            if self.rng.random() < stutter_rate:
                modified = self._add_stutter_pair(word)
            
            # 2. Character doubling within word
            if self.rng.random() < char_double_rate:
                modified = self._double_characters(modified)
            
            # 3. Insert filler before word
            if self.rng.random() < filler_rate:
                filler = self.rng.choice(self.FILLER_VARIANTS)
                modified = filler + ' ' + modified
            
            # 4. Double punctuation after word
            if modified.endswith(('.', ',', ';', ':')) and self.rng.random() < punct_double_rate:
                modified = modified + modified[-1]
            
            # 5. Split word
            if len(word) > 5 and self.rng.random() < word_split_rate:
                modified = self._split_word(modified)
            
            result.append(modified)
        
        text = ' '.join(result)
        
        # 6. Inject format command doubles (rare, per-text)
        if self.rng.random() < 0.3:
            text = self._inject_format_commands(text)
        
        return text
    
    def _add_stutter_pair(self, word: str) -> str:
        """Add a CTC stutter prefix: 'patient' → 'pa patient'"""
        prefix_len = self.rng.choice(self.STUTTER_PREFIX_LENGTHS)
        prefix_len = min(prefix_len, len(word) - 1)
        if prefix_len < 1:
            return word
        
        # Sometimes the stutter includes char doubling too
        fragment = word[:prefix_len].lower()
        if self.rng.random() < 0.3:
            # Double a char in the full word
            garbled_word = self._double_characters(word)
            return fragment + ' ' + garbled_word
        
        return fragment + ' ' + word
    
    def _double_characters(self, word: str) -> str:
        """Double 1-3 characters within a word: 'therapy' → 'therappy'"""
        if len(word) < 4:
            return word
        
        chars = list(word)
        num_doubles = self.rng.randint(1, min(3, len(word) // 3))
        
        positions = list(range(1, len(chars) - 1))
        self.rng.shuffle(positions)
        
        for pos in positions[:num_doubles]:
            ch = chars[pos].lower()
            if ch in self.DOUBLE_CHARS:
                weight = 0.6  # consonants doubled more often
            elif ch in self.DOUBLE_VOWELS:
                weight = 0.3
            else:
                continue
            
            if self.rng.random() < weight:
                chars.insert(pos, chars[pos])
        
        return ''.join(chars)
    
    def _split_word(self, word: str) -> str:
        """Split a word: 'followup' → 'fol lowup'"""
        # Strip punctuation for splitting, reattach after
        punct = ''
        core = word
        if core and core[-1] in '.,;:!?':
            punct = core[-1]
            core = core[:-1]
        
        if len(core) < 4:
            return word
        
        split_point = self.rng.randint(2, len(core) - 2)
        return core[:split_point] + ' ' + core[split_point:] + punct
    
    def _inject_format_commands(self, text: str) -> str:
        """Replace some punctuation with format commands, sometimes doubled."""
        # Replace a few periods with {period}
        replacements = [
            ('. ', '{period} ', 0.1),
            ('. ', '{periodperiod} ', 0.05),
            (', ', '{comma} ', 0.05),
            (', ', '{commacomma} ', 0.02),
        ]
        for old, new, rate in replacements:
            if old in text and self.rng.random() < rate:
                # Replace first occurrence only
                text = text.replace(old, new, 1)
        return text
    
    def add_section_header_noise(self, header: str) -> str:
        """Garble a section header: 'PHYSICAL EXAM' → 'PHHYSICAAL EXAMM'"""
        chars = list(header)
        result = []
        for ch in chars:
            result.append(ch)
            if ch.isalpha() and self.rng.random() < 0.3:
                result.append(ch)
        
        # Add bracket noise
        garbled = ''.join(result)
        variants = [
            f'[ [{garbled} ]]',
            f'[{garbled}]',
            f'[ [{garbled}]',
            f'[{garbled}]]',
        ]
        return self.rng.choice(variants)
    
    def add_unintelligible_markers(self, text: str, rate: float = 0.02) -> str:
        """Randomly insert [unintelligible] markers (with garbled spellings)."""
        variants = [
            '[unintelligible]', '[ [unintelligigible]]',
            '[ununintelligigible]', '[ [unintellelligible]]',
            '[untelligible]', '[ intell intelligible]]',
        ]
        words = text.split()
        result = []
        for w in words:
            if self.rng.random() < rate:
                result.append(self.rng.choice(variants))
            result.append(w)
        return ' '.join(result)


# ============================================================
# TRAINING DATA EXTRACTION
# ============================================================

def extract_aligned_segments(raw_text: str, clean_text: str,
                             max_segment_words: int = 64,
                             min_segment_words: int = 8) -> List[Tuple[str, str]]:
    """
    Extract aligned (noisy, clean) segments from paired texts.
    
    Uses sentence-level alignment by splitting on periods and
    matching segments by position. Segments are capped at
    max_segment_words for training efficiency.
    """
    # Split on sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n\n+', raw_text)
    clean_sentences = re.split(r'(?<=[.!?])\s+|\n\n+', clean_text)
    
    pairs = []
    
    # Positional alignment (works because cleaning doesn't reorder)
    min_len = min(len(raw_sentences), len(clean_sentences))
    
    for i in range(min_len):
        raw_seg = raw_sentences[i].strip()
        clean_seg = clean_sentences[i].strip()
        
        # Skip very short segments
        raw_words = raw_seg.split()
        clean_words = clean_seg.split()
        if len(clean_words) < min_segment_words:
            continue
        
        # Skip if segments are identical (no errors to learn from)
        if raw_seg == clean_seg:
            continue
        
        # Chunk long segments
        if len(raw_words) > max_segment_words:
            # Split into chunks
            for j in range(0, len(raw_words), max_segment_words):
                raw_chunk = ' '.join(raw_words[j:j + max_segment_words])
                clean_chunk = ' '.join(clean_words[j:j + max_segment_words])
                if len(clean_chunk.split()) >= min_segment_words:
                    pairs.append((raw_chunk, clean_chunk))
        else:
            pairs.append((raw_seg, clean_seg))
    
    return pairs


def generate_from_real_transcripts(raw_dir: str, clean_dir: str) -> List[TrainingPair]:
    """Generate training pairs from real paired transcripts."""
    pairs = []
    
    raw_files = sorted(Path(raw_dir).glob("*.txt"))
    clean_files = sorted(Path(clean_dir).glob("*_cleaned.txt"))
    
    # Match files by common prefix
    clean_map = {}
    for cf in clean_files:
        # Extract base name without _cleaned suffix
        base = cf.stem.replace('_cleaned', '')
        clean_map[base] = cf
    
    for rf in raw_files:
        base = rf.stem
        if base in clean_map:
            raw_text = rf.read_text(errors='replace')
            clean_text = clean_map[base].read_text(errors='replace')
            
            segments = extract_aligned_segments(raw_text, clean_text)
            for raw_seg, clean_seg in segments:
                pairs.append(TrainingPair(
                    source=raw_seg,
                    target=clean_seg,
                    origin='real',
                    file_id=base
                ))
            
            log.info(f"  {base}: {len(segments)} segments")
    
    return pairs


def generate_synthetic_pairs(clean_texts: List[str],
                             simulator: CTCErrorSimulator,
                             augmentation_factor: int = 10) -> List[TrainingPair]:
    """
    Generate synthetic training pairs by injecting CTC errors
    into clean medical text.
    
    Each clean text is augmented `augmentation_factor` times
    with different random error patterns.
    """
    pairs = []
    
    for text in clean_texts:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 5]
        
        for sent in sentences:
            for aug_idx in range(augmentation_factor):
                # Vary error rates slightly for diversity
                rate_multiplier = 0.5 + random.random()  # 0.5x to 1.5x
                
                noisy = simulator.inject_errors(
                    sent,
                    stutter_rate=0.10 * rate_multiplier,
                    char_double_rate=0.16 * rate_multiplier,
                    filler_rate=0.015 * rate_multiplier,
                )
                
                # Occasionally add unintelligible markers
                if random.random() < 0.1:
                    noisy = simulator.add_unintelligible_markers(noisy, rate=0.03)
                
                # Skip if no errors were actually injected
                if noisy == sent:
                    continue
                
                pairs.append(TrainingPair(
                    source=noisy,
                    target=sent,
                    origin='synthetic',
                    file_id=f'synth_{aug_idx}'
                ))
    
    return pairs


def generate_hard_examples(pairs: List[TrainingPair],
                           simulator: CTCErrorSimulator) -> List[TrainingPair]:
    """
    Generate hard examples by taking real error patterns that
    the rule-based pipeline struggled with and creating more
    training data around those patterns.
    
    "Hard" patterns from our analysis:
      - "cervivical" (nested syllable repetition)
      - "inspectspection" (overlapping word repetition)
      - "phyysoscopic" (multi-char garbling)
      - Broken fragments: "rength", "ction", "sible"
    """
    hard_patterns = [
        # (garbled, correct) pairs from residual analysis
        ('cervivical', 'cervical'),
        ('indicicated', 'indicated'),
        ('achching', 'aching'),
        ('Patatient', 'Patient'),
        ('negatative', 'negative'),
        ('inspectspection', 'inspection'),
        ('complaintplaint', 'complaint'),
        ('phyysoscopic', 'physioscopic'),
        ('mottorveicle', 'motor vehicle'),
    ]
    
    hard_pairs = []
    
    # Create sentence-level examples around hard patterns
    templates = [
        "The patient presents with {} symptoms.",
        "Examination reveals {} findings bilaterally.",
        "She was referred for {} evaluation.",
        "The {} report was reviewed.",
        "There is {} tenderness to palpation.",
    ]
    
    for garbled, correct in hard_patterns:
        for template in templates:
            noisy = template.format(garbled)
            clean = template.format(correct)
            hard_pairs.append(TrainingPair(
                source=noisy, target=clean,
                origin='hard_example', file_id='hard'
            ))
    
    return hard_pairs


# ============================================================
# MAIN
# ============================================================

def save_training_data(pairs: List[TrainingPair], output_path: str):
    """Save training pairs as JSONL (one JSON object per line)."""
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair)) + '\n')
    log.info(f"Saved {len(pairs)} training pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for MedASR ML post-processor")
    parser.add_argument('--raw-dir', required=True, help='Directory of raw MedASR transcripts')
    parser.add_argument('--clean-dir', required=True, help='Directory of rule-based cleaned transcripts')
    parser.add_argument('--medical-corpus', help='Optional clean medical text for synthetic augmentation')
    parser.add_argument('--output', default='training_pairs.jsonl', help='Output JSONL file')
    parser.add_argument('--augmentation-factor', type=int, default=10,
                        help='How many synthetic variants per clean sentence')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    simulator = CTCErrorSimulator(seed=args.seed)
    all_pairs = []
    
    # Phase 1: Real transcript pairs
    log.info("Phase 1: Extracting real transcript pairs...")
    real_pairs = generate_from_real_transcripts(args.raw_dir, args.clean_dir)
    all_pairs.extend(real_pairs)
    log.info(f"  → {len(real_pairs)} real pairs")
    
    # Phase 2: Synthetic augmentation from cleaned transcripts
    log.info("Phase 2: Generating synthetic training data...")
    clean_texts = []
    for f in sorted(Path(args.clean_dir).glob("*_cleaned.txt")):
        clean_texts.append(f.read_text(errors='replace'))
    
    if args.medical_corpus and os.path.exists(args.medical_corpus):
        with open(args.medical_corpus) as f:
            clean_texts.append(f.read())
    
    synth_pairs = generate_synthetic_pairs(clean_texts, simulator, args.augmentation_factor)
    all_pairs.extend(synth_pairs)
    log.info(f"  → {len(synth_pairs)} synthetic pairs")
    
    # Phase 3: Hard examples
    log.info("Phase 3: Generating hard examples...")
    hard_pairs = generate_hard_examples(all_pairs, simulator)
    all_pairs.extend(hard_pairs)
    log.info(f"  → {len(hard_pairs)} hard examples")
    
    # Shuffle and save
    random.shuffle(all_pairs)
    save_training_data(all_pairs, args.output)
    
    log.info(f"\nTotal training pairs: {len(all_pairs)}")
    log.info(f"  Real:      {sum(1 for p in all_pairs if p.origin == 'real')}")
    log.info(f"  Synthetic: {sum(1 for p in all_pairs if p.origin == 'synthetic')}")
    log.info(f"  Hard:      {sum(1 for p in all_pairs if p.origin == 'hard_example')}")


if __name__ == '__main__':
    main()
