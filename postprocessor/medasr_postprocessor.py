"""
MedASR Transcript Post-Processor
=================================
Cleans mechanical CTC decoder artifacts from MedASR transcripts
before sending to MedGamma LLM for semantic interpretation.

Design principle: Fix MECHANICAL artifacts that confuse LLMs.
Leave SEMANTIC interpretation (medical terms, sentence structure,
clinical meaning) to MedGamma — it has domain knowledge we don't.

Error patterns addressed (in processing order):
  1. Format command doubles: {periodperiod} → {period}
  2. Double/triple punctuation: ,, → ,  .. → .
  3. CTC stutter pairs: "pa painin" → "painin", "re reportport" → "reportport"
  4. Internal character stuttering: "todaayy" → "today", "neckck" → "neck"
  5. Filler word normalization: "u h", "u hh", "uuhhm" → [uh]
  6. Broken word merging: "mo motiontion" → "motiontion"  
  7. Section header cleanup: "[ [PHHYSICAAL EXAMM ]]" → "[PHYSICAL EXAM]"
  8. Whitespace normalization: double spaces, trailing spaces
  9. Trailing artifacts: "-e" at end of transcripts
  10. Dictionary matching: char dedup + fuzzy match vs English + medical terms
  11. Medical spell check: 98K OpenMedSpel terms via SymSpell (optional)
  12. MedASR artifact removal: [unintelligible], {format commands}, [uh]/[um]

Error patterns NOT addressed (left for MedGamma):
  - Missing words / dropped content
  - Sentence structure reconstruction
  - Clinical meaning interpretation

Usage:
  # Single file
  python medasr_postprocessor.py input_transcript.txt -o cleaned.txt

  # Batch directory
  python medasr_postprocessor.py /path/to/transcripts/ -o /path/to/cleaned/ --batch

  # With before/after diff report
  python medasr_postprocessor.py input.txt -o cleaned.txt --report

  # Pipeline mode (stdin/stdout)
  cat transcript.txt | python medasr_postprocessor.py - > cleaned.txt

Requirements:
  Python 3.8+ (stdlib only — no external dependencies)
"""

import argparse
import difflib
import itertools
import logging
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("postprocessor")


# ============================================================
# METRICS TRACKING
# ============================================================

@dataclass
class CleanupMetrics:
    """Track what was fixed for reporting."""
    format_cmd_doubles: int = 0
    punctuation_doubles: int = 0
    stutter_pairs_merged: int = 0
    char_stutters_fixed: int = 0
    fillers_normalized: int = 0
    broken_words_merged: int = 0
    headers_cleaned: int = 0
    whitespace_fixes: int = 0
    trailing_artifacts: int = 0
    dictionary_corrections: int = 0
    medical_corrections: int = 0
    medasr_artifacts_removed: int = 0
    format_commands_resolved: int = 0
    offensive_misrecognitions_removed: int = 0
    words_before: int = 0
    words_after: int = 0

    def summary(self) -> str:
        lines = [
            "=== POST-PROCESSING METRICS ===",
            f"  Format command doubles fixed:   {self.format_cmd_doubles}",
            f"  Punctuation doubles fixed:      {self.punctuation_doubles}",
            f"  CTC stutter pairs merged:       {self.stutter_pairs_merged}",
            f"  Character stutters fixed:       {self.char_stutters_fixed}",
            f"  Filler words normalized:        {self.fillers_normalized}",
            f"  Broken words merged:            {self.broken_words_merged}",
            f"  Section headers cleaned:        {self.headers_cleaned}",
            f"  Whitespace fixes:               {self.whitespace_fixes}",
            f"  Trailing artifacts removed:     {self.trailing_artifacts}",
            f"  Dictionary corrections:         {self.dictionary_corrections}",
            f"  Medical spell corrections:      {self.medical_corrections}",
            f"  MedASR artifacts removed:       {self.medasr_artifacts_removed}",
            f"  Format commands resolved:       {self.format_commands_resolved}",
            f"  Offensive misrecognitions removed: {self.offensive_misrecognitions_removed}",
            f"  Words: {self.words_before} → {self.words_after} "
            f"({self.words_before - self.words_after} removed, "
            f"{(self.words_before - self.words_after) / max(self.words_before, 1) * 100:.1f}% reduction)",
        ]
        return "\n".join(lines)


# ============================================================
# STAGE 0: OFFENSIVE ASR MISRECOGNITION FILTER
# ============================================================
#
# ASR models (especially CTC-based) occasionally decode audio as phonetically
# similar but semantically offensive words that would NEVER appear in a
# legitimate clinical transcript.  Leaving these in the transcript can:
#   1. Trigger LLM safety refusals, causing the note generation to fail.
#   2. Introduce hallucinated content into a clinical note.
#
# This list is intentionally narrow — only terms that are:
#   a) Offensive / inappropriate, AND
#   b) Have no plausible clinical meaning in any medical specialty.
#
# Words with legitimate clinical uses (e.g. "rape" in OB/GYN documentation,
# "incarceration" in hernia surgery) are intentionally EXCLUDED.
#
# Matched words are removed entirely (replaced with a single space).
# Word-boundary anchors (\b) prevent false matches inside longer tokens.

_OFFENSIVE_MISRECOGNITION_RE = re.compile(
    r"\b(?:"
    # Sexual exploitation / abuse terms with no clinical synonym
    r"pedophil(?:e|ia|ic|es|iac|iacs|ias)?"
    r"|pederast(?:s|y)?"
    r"|ephebophil(?:e|ia|ic)?"
    # Slurs and hate-speech terms that ASR commonly confuses with medical words
    r"|racist(?:s)?"
    r"|racism"
    r"|rapist(?:s)?"
    r"|neo-?nazi(?:s)?"
    r"|misogynist(?:s|ic)?"
    # Explicit sexual terms that are pure ASR artifacts in a clinical context
    r"|pornograph(?:y|ic|er|ers)?"
    r"|masturbat(?:ion|ing|e|es|ed)?"
    r")\b",
    re.IGNORECASE,
)


def remove_offensive_misrecognitions(text: str, metrics: CleanupMetrics) -> str:
    """
    Stage 0 — remove words that are almost certainly ASR misrecognitions of
    clinical speech.  Offensive terms listed in _OFFENSIVE_MISRECOGNITION_RE
    are stripped to prevent LLM safety refusals and hallucinated content.
    """
    def _replace(m: re.Match) -> str:
        metrics.offensive_misrecognitions_removed += 1
        log.debug(f"Stage 0: removed offensive misrecognition '{m.group()}'")
        return " "

    return _OFFENSIVE_MISRECOGNITION_RE.sub(_replace, text)


# ============================================================
# STAGE 1: FORMAT COMMAND DOUBLES
# ============================================================

def fix_format_command_doubles(text: str, metrics: CleanupMetrics) -> str:
    """
    Fix doubled format commands from CTC decoder.
    {periodperiod} → {period}
    {commacomma} → {comma}
    {newnew line} → {new line}
    {newnew paragraph} → {new paragraph}
    {newnewparagraph} → {new paragraph}
    {coloncolon} → {colon}
    {slashslash} → {slash}
    Also handles triple+ repeats.
    """
    patterns = [
        # Doubled command words inside braces
        (r'\{period\s*period\}', '{period}'),
        (r'\{comma\s*comma\}', '{comma}'),
        (r'\{colon\s*colon\}', '{colon}'),
        (r'\{slash\s*slash\}', '{slash}'),
        # New line/paragraph variants
        (r'\{new\s*new\s*line\}', '{new line}'),
        (r'\{new\s*new\s*paragraph\s*paragraph\}', '{new paragraph}'),
        (r'\{new\s*new\s*paragraph\}', '{new paragraph}'),
        (r'\{newnew\s*paragraph\s*paragraph\}', '{new paragraph}'),
        (r'\{newnewparagraph\}', '{new paragraph}'),
        (r'\{newnewparagraphparagraph\}', '{new paragraph}'),
        # Handle doubled braces: {{ → {
        (r'\{\s*\{', '{'),
        (r'\}\s*\}', '}'),
        # end of dictation doubles
        (r'\{end\s*end\b', '{end'),
        # next doubles
        (r'\{ne\s*next\}', '{next}'),
        (r'\{nenext\}', '{next}'),
    ]
    for pattern, replacement in patterns:
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count > 0:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            metrics.format_cmd_doubles += count
    return text


# ============================================================
# STAGE 2: PUNCTUATION CLEANUP
# ============================================================

def fix_punctuation_doubles(text: str, metrics: CleanupMetrics) -> str:
    """
    Fix doubled/tripled punctuation from CTC decoder.
    ,, → ,    .. → .    :: → :    ;; → ;
    Also fix space-separated doubles: , , → ,
    """
    original = text

    # Space-separated doubled punctuation: ". ." → "."
    for p in ['.', ',', ':', ';', '?', '!']:
        escaped = re.escape(p)
        text = re.sub(rf'{escaped}\s+{escaped}', p, text)

    # Adjacent doubled/tripled punctuation: ".." → "."
    for p in ['.', ',', ':', ';', '?', '!']:
        escaped = re.escape(p)
        text = re.sub(rf'{escaped}{{2,}}', p, text)

    # Count changes
    if text != original:
        # Rough count based on length reduction
        metrics.punctuation_doubles += (len(original) - len(text))

    return text


# ============================================================
# STAGE 3: CTC STUTTER PAIR MERGING
# ============================================================

# Common short fragments that are real words — don't merge these
REAL_SHORT_WORDS = {
    'a', 'an', 'as', 'at', 'be', 'by', 'do', 'go', 'he', 'if', 'in',
    'is', 'it', 'me', 'my', 'no', 'of', 'on', 'or', 'so', 'to', 'up',
    'us', 'we', 'am', 'an', 'are', 'and', 'but', 'can', 'did', 'for',
    'had', 'has', 'her', 'him', 'his', 'how', 'its', 'let', 'may',
    'new', 'not', 'now', 'old', 'one', 'our', 'out', 'own', 'per',
    'put', 'ran', 'say', 'she', 'the', 'too', 'two', 'use', 'was',
    'who', 'why', 'yet', 'you', 'all', 'any', 'day', 'get', 'got',
    'see', 'set', 'sit', 'six', 'ten', 'try', 'via', 'also', 'back',
    'been', 'both', 'come', 'does', 'down', 'each', 'even', 'from',
    'give', 'good', 'have', 'here', 'high', 'into', 'just', 'keep',
    'last', 'left', 'long', 'look', 'made', 'make', 'many', 'more',
    'much', 'must', 'next', 'only', 'over', 'part', 'past', 'same',
    'seem', 'show', 'side', 'some', 'such', 'take', 'tell', 'than',
    'that', 'them', 'then', 'they', 'this', 'time', 'turn', 'very',
    'want', 'well', 'went', 'were', 'what', 'when', 'will', 'with',
    'work', 'year', 'your', 'four', 'five',
    # Medical context
    'pain', 'hand', 'head', 'knee', 'back', 'neck', 'left', 'right',
    'full', 'bone', 'drug', 'dose', 'test', 'mild', 'skin', 'oral',
}


def _has_internal_stutter(word: str) -> bool:
    """Check if a word contains internal character-level stutter patterns.
    Words like 'todaayy', 'neckck', 'headadaches' clearly aren't real English."""
    w = word.lower()
    if len(w) < 5:
        return False
    # Check for repeated 2-char sequences
    for i in range(len(w) - 3):
        if w[i:i+2] == w[i+2:i+4] and w[i:i+2].isalpha():
            return True
    # Check for repeated 3-char sequences
    for i in range(len(w) - 5):
        if w[i:i+3] == w[i+3:i+6] and w[i:i+3].isalpha():
            return True
    return False


def merge_stutter_pairs(text: str, metrics: CleanupMetrics) -> str:
    """
    Merge CTC stutter pairs where a fragment precedes the full word.

    Patterns:
      "pa painin" → "painin"     (fragment is prefix of next word)
      "re reportport" → "reportport"
      "follow followup" → "followup"
      "to todaayy" → "todaayy"

    Rules:
      - Fragment must be 1-6 chars
      - Next word must start with the fragment
      - Next word must be longer than fragment
      - Fragment must NOT be a common real word UNLESS the next word
        has internal stutter patterns (e.g., "he headadaches" — "he"
        is a real word but "headadaches" is clearly stuttered)
      - Preserve punctuation attached to the kept word
    """
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        if i < len(words) - 1:
            w1_raw = words[i]
            w2_raw = words[i + 1]

            # Strip punctuation for comparison
            w1 = re.sub(r'^[\[\({]*', '', w1_raw)
            w1 = re.sub(r'[\]\)},.:;!?]*$', '', w1).lower()
            w2 = re.sub(r'^[\[\({]*', '', w2_raw)
            w2_clean = re.sub(r'[\]\)},.:;!?]*$', '', w2).lower()

            # Basic stutter conditions
            basic_match = (
                1 <= len(w1) <= 6
                and len(w2_clean) >= 3
                and w2_clean.startswith(w1)
                and len(w2_clean) > len(w1)
            )

            if basic_match:
                # If fragment is NOT a real word, always merge
                if w1 not in REAL_SHORT_WORDS:
                    result.append(w2_raw)
                    metrics.stutter_pairs_merged += 1
                    i += 2
                    continue

                # If fragment IS a real word, only merge if next word
                # has internal stutter patterns (strong signal it's garbled)
                if _has_internal_stutter(w2_clean):
                    result.append(w2_raw)
                    metrics.stutter_pairs_merged += 1
                    i += 2
                    continue

                # Also merge if w2 is just w1 with doubled ending
                # "to todday" → "todday", "he heaad" → "heaad"
                remainder = w2_clean[len(w1):]
                if len(remainder) >= 1 and remainder[0] == w1[-1]:
                    result.append(w2_raw)
                    metrics.stutter_pairs_merged += 1
                    i += 2
                    continue

        result.append(words[i])
        i += 1

    return ' '.join(result)


# ============================================================
# STAGE 4: INTERNAL CHARACTER STUTTER FIX
# ============================================================

def fix_char_stutters(text: str, metrics: CleanupMetrics) -> str:
    """
    Fix character-level stuttering within words caused by CTC decoder.
    Runs multiple passes since fixing one stutter may reveal another.

    Patterns:
      "todaayy" → "today"        (repeated trailing chars)
      "neckck" → "neck"          (repeated syllable ending)
      "motiontion" → "motion"    (repeated suffix)
      "Patiientent" → "Patient"  (repeated internal syllable)
      "cervivical" → "cervical"  (repeated internal syllable)
      "headadaches" → "headaches"
      "indicicated" → "indicated"
    """
    def fix_word_once(word: str) -> str:
        """Try to fix internal stuttering in a single word — one pass."""
        match = re.match(r'^([^a-zA-Z]*)(.*?)([^a-zA-Z]*)$', word)
        if not match:
            return word
        prefix, core, suffix = match.groups()
        if len(core) < 4:
            return word

        original_core = core
        core_lower = core.lower()

        # Pattern A: Repeated suffix (2-5 chars at end appear twice)
        # "neckck" → "neck", "motiontion" → "motion"
        for suf_len in range(5, 1, -1):
            if len(core_lower) >= suf_len * 2:
                ending = core_lower[-suf_len:]
                before = core_lower[-suf_len*2:-suf_len]
                if ending == before:
                    core = core[:-suf_len]
                    return prefix + core + suffix

        # Pattern B: Repeated internal syllable (2-4 chars appear twice consecutively)
        # "cervivical" → "cervical", "headadaches" → "headaches"
        # "indicicated" → "indicated"
        for chunk_len in range(4, 1, -1):
            for pos in range(0, len(core_lower) - chunk_len * 2 + 1):
                chunk = core_lower[pos:pos + chunk_len]
                next_chunk = core_lower[pos + chunk_len:pos + chunk_len * 2]
                if chunk == next_chunk and chunk.isalpha():
                    # Remove the first occurrence of the duplicate
                    core = core[:pos] + core[pos + chunk_len:]
                    return prefix + core + suffix

        # Pattern C: Doubled single characters at word end
        # "todaayy" → "today"
        end_match = re.match(r'^(.*[a-z])(\w)\2+$', core_lower)
        if end_match and len(core) > 4:
            core = core[:len(end_match.group(1)) + 1]
            return prefix + core + suffix

        return word

    # Process word by word, with multiple passes
    words = text.split()
    result = []
    for w in words:
        prev = w
        # Run up to 3 passes per word to collapse nested stutters
        for _ in range(3):
            fixed = fix_word_once(prev)
            if fixed == prev:
                break
            prev = fixed
        if prev != w:
            metrics.char_stutters_fixed += 1
            log.debug(f"  char_stutter: '{w}' → '{prev}'")
        result.append(prev)

    return ' '.join(result)


# ============================================================
# STAGE 5: FILLER WORD NORMALIZATION
# ============================================================

def normalize_fillers(text: str, metrics: CleanupMetrics) -> str:
    """
    Normalize ASR-decoded filler words to consistent markers.
    
    "u h" / "u hh" / "U h" / "Uh" → [uh]
    "uuhhm" / "uhm" / "UUhhm" / "Uhm" → [um]
    
    These become clean markers that MedGamma can handle or strip.
    """
    # UHM variants first (before UH, since uhm contains uh)
    uhm_pattern = r'\b[Uu]\s*[Uu]?\s*[Hh]+\s*[Mm]+\b'
    count_uhm = len(re.findall(uhm_pattern, text))
    text = re.sub(uhm_pattern, '[um]', text)

    # UH variants (with and without space between u and h)
    # Be careful not to match "uh" inside words
    uh_patterns = [
        r'\b[Uu]\s+[Hh]+\b',       # "u h", "u hh", "U h"
        r'\b[Uu][Uu]+\s*[Hh]+\b',  # "UUhh", "uuh" 
        r'\b[Uu][Hh][Hh]+\b',      # "Uhh", "uhhh"
    ]
    count_uh = 0
    for p in uh_patterns:
        matches = len(re.findall(p, text))
        count_uh += matches
        text = re.sub(p, '[uh]', text)

    # Simple "Uh" at word boundary (but not inside words)
    # Only match standalone "Uh" preceded by space/start and followed by space/end/punct
    text_new = re.sub(r'(?<![a-zA-Z])[Uu][Hh](?![a-zA-Z])', '[uh]', text)
    count_uh += (len(text) - len(text_new)) // 2  # rough
    text = text_new

    # Clean up double markers
    text = re.sub(r'\[uh\]\s*\[uh\]', '[uh]', text)
    text = re.sub(r'\[um\]\s*\[um\]', '[um]', text)

    metrics.fillers_normalized += count_uhm + count_uh
    return text


# ============================================================
# STAGE 6: BROKEN WORD MERGING
# ============================================================

def merge_broken_words(text: str, metrics: CleanupMetrics) -> str:
    """
    Merge words that were broken by spaces in the CTC decoder.
    
    "mo motiontion" was already partially handled by stutter merge,
    but there are cases like:
      "in spect ionion" → needs multiple merges
      "p palpation" → "palpation"
      "f flexion" → "flexion"
      "b bony" → "bony"
      "ss symptoms" → "symptoms" 
    
    Strategy: If a single letter or 2-letter fragment precedes a word
    that starts with that letter/fragment, merge them by keeping the
    longer word.
    
    This is actually a subset of the stutter merging (stage 3) but
    for very short fragments (1 char) that were excluded from the
    general stutter rules to avoid false positives. Here we're more
    aggressive because single-letter "words" are almost never intentional
    in medical dictation.
    """
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        if i < len(words) - 1:
            w1 = words[i].lower().strip('.,;:{}[]()-')
            w2 = words[i + 1].lower().strip('.,;:{}[]()-')

            # Single consonant followed by word starting with that consonant
            # "b bony" → "bony", "f flexion" → "flexion", "p palpation" → "palpation"
            if (len(w1) == 1
                and w1.isalpha()
                and w1 not in {'a', 'i'}  # legitimate single-letter words
                and len(w2) >= 3
                and w2.startswith(w1)):
                result.append(words[i + 1])  # keep the full word with its punctuation
                metrics.broken_words_merged += 1
                i += 2
                continue

            # Double consonant "ss" followed by word starting with "s"
            # "ss symptoms" → "symptoms"
            if (len(w1) == 2
                and w1[0] == w1[1]
                and w1[0].isalpha()
                and len(w2) >= 3
                and w2.startswith(w1[0])):
                result.append(words[i + 1])
                metrics.broken_words_merged += 1
                i += 2
                continue

        result.append(words[i])
        i += 1

    return ' '.join(result)


# ============================================================
# STAGE 7: SECTION HEADER CLEANUP
# ============================================================

# Known section headers in medical dictation
KNOWN_HEADERS = {
    'PAST MEDICAL HISTORY',
    'PAST SURGICAL HISTORY',
    'FAMILY HISTORY',
    'SOCIAL HISTORY',
    'ALLERGIES',
    'MEDICATIONS',
    'REVIEW OF SYSTEMS',
    'PHYSICAL EXAM',
    'PHYSICAL EXAMINATION',
    'ASSESSMENT',
    'PLAN',
    'IMPRESSION',
    'HPI',
    'HISTORY OF PRESENT ILLNESS',
    'CHIEF COMPLAINT',
    'LOCATION',
    'SUBJECTIVE',
    'OBJECTIVE',
}


def clean_section_headers(text: str, metrics: CleanupMetrics) -> str:
    """
    Clean garbled section headers.
    
    "[ [PHHYSICAAL EXAMM ]]" → "[PHYSICAL EXAM]"
    "[AS SSES SMENTT]" → "[ASSESSMENT]"
    "[PPLAN]" → "[PLAN]"
    
    Strategy: Find bracketed sections, extract their text content,
    fuzzy-match against known headers, replace if match found.
    """
    def clean_header_text(header_text: str) -> str:
        """Clean the text inside brackets and try to match known headers."""
        # Remove extra spaces and duplicate chars
        cleaned = header_text.strip()
        # Remove doubled chars aggressively since this is a header
        cleaned = re.sub(r'([A-Z])\1+', r'\1', cleaned.upper())
        # Remove internal spaces within what should be single words
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Try fuzzy matching against known headers
        best_match = None
        best_score = 0

        for known in KNOWN_HEADERS:
            # Simple character-level similarity
            # Remove spaces from both and compare
            c1 = cleaned.replace(' ', '')
            c2 = known.replace(' ', '')
            # Count matching chars
            matches = sum(1 for a, b in zip(c1, c2) if a == b)
            score = matches / max(len(c1), len(c2)) if max(len(c1), len(c2)) > 0 else 0
            if score > best_score and score > 0.6:
                best_score = score
                best_match = known

        return best_match if best_match else cleaned

    # Find all bracketed sections
    def replace_header(match):
        inner = match.group(1)
        cleaned = clean_header_text(inner)
        if cleaned != inner.strip():
            metrics.headers_cleaned += 1
        return f'[{cleaned}]'

    # Match various bracket patterns including doubled brackets
    # [[ TEXT ]] or [ TEXT ] or [TEXT]
    text = re.sub(r'\[\s*\[?\s*([A-Z][A-Z\s.,]+?)\s*\]?\s*\]', replace_header, text)

    return text


# ============================================================
# STAGE 8: WHITESPACE NORMALIZATION
# ============================================================

def normalize_whitespace(text: str, metrics: CleanupMetrics) -> str:
    """Clean up whitespace artifacts."""
    original_len = len(text)

    # Multiple spaces → single space
    text = re.sub(r'  +', ' ', text)

    # Space before punctuation: "word ." → "word."
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Space after opening bracket: "[ word" → "[word"
    text = re.sub(r'\[\s+', '[', text)

    # Space before closing bracket: "word ]" → "word]"
    text = re.sub(r'\s+\]', ']', text)

    # Multiple newlines → double newline
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Trim lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    metrics.whitespace_fixes += abs(len(text) - original_len)
    return text


# ============================================================
# STAGE 9: TRAILING ARTIFACTS
# ============================================================

def remove_trailing_artifacts(text: str, metrics: CleanupMetrics) -> str:
    """Remove known trailing artifacts from MedASR output."""
    # "-e" at end of transcript (common MedASR artifact)
    if text.rstrip().endswith('-e'):
        text = text.rstrip()[:-2].rstrip()
        metrics.trailing_artifacts += 1

    # Trailing isolated punctuation
    text = re.sub(r'\s+[.\-,]+\s*$', '', text)

    return text


# ============================================================
# STAGE 10: DICTIONARY MATCHING
# ============================================================

# Medical terms that aren't in standard English dictionaries.
# This list covers orthopedic, pain management, and general
# medical dictation vocabulary seen across our transcript corpus.
MEDICAL_DICTIONARY = {
    # Anatomy
    'cervical', 'thoracic', 'lumbar', 'sacral', 'paracervical',
    'parathoracic', 'paralumbar', 'musculature', 'paraspinal',
    'maxillofacial', 'intracranial', 'anteroposterior', 'posterolateral',
    'anterolateral', 'posteroanterior',
    # Joints / movements
    'flexion', 'extension', 'abduction', 'adduction', 'pronation',
    'supination', 'dorsiflexion', 'plantarflexion', 'circumduction',
    'eversion', 'inversion', 'subluxation', 'dislocation',
    # Orthopedic
    'meniscus', 'meniscal', 'ligament', 'ligamentous', 'tendon',
    'tendinitis', 'tendinosis', 'bursitis', 'arthritis', 'arthroplasty',
    'arthroscopy', 'orthopedic', 'orthopedist', 'chiropractic',
    'chiropractor', 'deformities', 'deformity',
    # Exam terms
    'palpation', 'palpable', 'tenderness', 'radicular', 'paresthesias',
    'paresthesia', 'numbness', 'tingling', 'bilaterally', 'bilateral',
    'unilateral', 'contralateral', 'ipsilateral',
    # Specific tests
    'spurling', "spurling's", 'tinel', "tinel's", 'phalen', "phalen's",
    'lachman', "lachman's", 'mcmurray', "mcmurray's",
    # Medications
    'naproxen', 'meloxicam', 'cyclobenzaprine', 'medrol', 'dosepak',
    'ibuprofen', 'acetaminophen', 'gabapentin', 'pregabalin',
    'prednisone', 'methylprednisolone', 'tramadol', 'hydrocodone',
    'oxycodone', 'celecoxib', 'diclofenac', 'ketorolac',
    # Procedures / imaging
    'cholecystectomy', 'appendectomy', 'laminectomy', 'discectomy',
    'arthroplasty', 'radiography', 'radiology', 'radiologist',
    'fluoroscopy', 'electromyography', 'electrodiagnostic',
    # Clinical terms
    'sprain', 'strain', 'contusion', 'laceration', 'abrasion',
    'avulsion', 'fracture', 'hematoma', 'edema', 'effusion',
    'crepitus', 'ecchymosis', 'erythema', 'induration',
    'posttraumatic', 'periarticular', 'intraarticular',
    # Body regions / descriptors
    'subacute', 'acute', 'chronic', 'proximal', 'distal',
    'medial', 'lateral', 'anterior', 'posterior', 'superior',
    'inferior', 'superficial', 'subcutaneous',
    # ROS / history terms
    'nausea', 'vomiting', 'diarrhea', 'constipation', 'dyspnea',
    'tachycardia', 'bradycardia', 'hypertension', 'hypotension',
    'asthma', 'diabetes', 'hypothyroidism', 'hyperthyroidism',
    'anxiety', 'depression',
    # Dictation / report structure
    'noncontributory', 'unremarkable', 'followup', 'workup',
    'sedentary', 'occupational', 'preexisting', 'reexamination',
    'reevaluation', 'symptomatology',
    # Common abbreviation expansions
    'milligrams', 'milligram', 'kilograms', 'kilogram',
    # Words commonly confused by standard dictionaries
    'carpal', 'metacarpal', 'phalanges', 'phalangeal',
    'acromioclavicular', 'sternoclavicular', 'glenohumeral',
    'patellofemoral', 'tibiofemoral', 'sacroiliac',
}

# Common format/dictation words used in medical transcription
DICTATION_VOCABULARY = {
    'evaluation', 'reevaluation', 'followup', 'recheck',
    'derangement', 'impingement', 'dysfunction',
    'nonsteroidal', 'antiinflammatory', 'corticosteroid',
    'rehabilitation', 'physiotherapy', 'kinesiology',
}


def _load_dictionary(extra_dict_path: Optional[str] = None) -> set:
    """Load combined English + medical dictionary."""
    words = set()

    # Try system dictionary
    dict_paths = [
        '/usr/share/dict/words',
        '/usr/share/dict/american-english',
        '/usr/share/dict/british-english',
    ]
    for path in dict_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    w = line.strip().lower()
                    if len(w) >= 2:
                        words.add(w)
            log.debug(f"Loaded {len(words)} words from {path}")
            break

    if not words:
        log.warning("No system dictionary found. Dictionary matching will use medical terms only.")

    # Add medical vocabulary
    words.update(w.lower() for w in MEDICAL_DICTIONARY)
    words.update(w.lower() for w in DICTATION_VOCABULARY)

    # Load optional extra dictionary file (one word per line)
    if extra_dict_path and os.path.exists(extra_dict_path):
        with open(extra_dict_path, 'r') as f:
            for line in f:
                w = line.strip().lower()
                if len(w) >= 2:
                    words.add(w)
        log.debug(f"Loaded extra dictionary from {extra_dict_path}")

    return words


def _deduplicate_chars(word: str) -> set:
    """
    Generate variants of a word by collapsing doubled characters.
    'nnumber' → {'nnumber', 'number'}
    'therappy' → {'therappy', 'therapy'}
    'conttinuue' → {'conttinuue', 'conttinue', 'continuue', 'continue'}
    
    Handles up to 6 doubled-character positions to avoid combinatorial explosion.
    """
    word_lower = word.lower()

    # Find positions where characters are doubled
    double_positions = []
    i = 0
    while i < len(word_lower) - 1:
        if word_lower[i] == word_lower[i + 1]:
            double_positions.append(i)
            i += 2
        else:
            i += 1

    if not double_positions:
        return {word_lower}

    # Cap at 6 to limit search space (2^6 = 64 variants max)
    double_positions = double_positions[:6]
    results = set()

    for combo in itertools.product([True, False], repeat=len(double_positions)):
        chars = list(word_lower)
        # Process in reverse so index shifts don't matter
        for keep_double, pos in sorted(zip(combo, double_positions), reverse=True):
            if not keep_double and pos < len(chars):
                del chars[pos]
        results.add(''.join(chars))

    return results


def _find_best_match(word: str, dictionary: set, min_similarity: float = 0.80) -> Optional[str]:
    """
    Find the best dictionary match for a misspelled word.
    
    Strategy (in priority order):
      1. Character deduplication — try collapsing doubled chars (HIGH confidence)
      2. difflib fuzzy match — Levenshtein-like similarity (MEDIUM confidence)
    
    Returns None if no confident match found.
    """
    word_lower = word.lower()

    # Already in dictionary?
    if word_lower in dictionary:
        return None

    # Strategy 1: Deduplicate characters
    # This is the most reliable for CTC artifacts — the word structure
    # is correct, just has doubled chars from the decoder
    variants = _deduplicate_chars(word_lower)
    dedup_matches = [v for v in variants if v in dictionary and v != word_lower]
    if dedup_matches:
        # Pick the shortest match (most doubles collapsed)
        best = min(dedup_matches, key=len)
        return best

    # Strategy 2: difflib fuzzy match
    # More aggressive requirements to avoid false positives:
    #   - Word must be >= 6 chars (short words have too many spurious matches)
    #   - Match must have same first letter
    #   - Match must be within ±2 chars of same length
    #   - Higher similarity threshold than dedup
    if len(word_lower) >= 6:
        candidates = [w for w in dictionary
                      if abs(len(w) - len(word_lower)) <= 2
                      and w[0] == word_lower[0]
                      and len(w) >= 5]

        matches = difflib.get_close_matches(word_lower, candidates, n=3, cutoff=min_similarity)
        if matches:
            # Among matches, prefer one closest in length
            best = min(matches, key=lambda m: abs(len(m) - len(word_lower)))
            return best

    return None


# Words that should NEVER be "corrected" by the dictionary matcher.
# These are either intentional short fragments, format commands,
# markers, or words that look wrong but are domain-specific.
DICTIONARY_SKIP_PATTERNS = {
    # Markers and tags
    re.compile(r'^\['),       # [uh], [um], [unintelligible]
    re.compile(r'^\{'),       # {period}, {comma}, {new line}
    re.compile(r'^[A-Z]{2,}$'),  # Abbreviations: MCP, IP, CT, MRI, ER, ED
    re.compile(r'^\d'),       # Numbers: 100, 66, 15mg
    re.compile(r'^[a-z]{1,3}$'),  # Very short fragments (leave for MedGamma)
    re.compile(r"^[a-z]'s$", re.IGNORECASE),  # Possessives like "s's"
}


def dictionary_match(text: str, metrics: CleanupMetrics,
                     dictionary: Optional[set] = None,
                     extra_dict_path: Optional[str] = None,
                     min_similarity: float = 0.80) -> str:
    """
    Match words against combined English + medical dictionary.
    Fix words that are close to a dictionary entry.
    
    This stage runs AFTER all mechanical cleanup (stutters, doubles, etc.)
    to catch remaining misspellings from partially-cleaned CTC output.
    
    Parameters:
        text: input text
        metrics: tracking object
        dictionary: pre-loaded dictionary set (loaded automatically if None)
        extra_dict_path: path to additional dictionary file (one word per line)
        min_similarity: minimum difflib ratio for fuzzy matching (0-1, default 0.80)
    """
    if dictionary is None:
        dictionary = _load_dictionary(extra_dict_path)

    words = text.split()
    result = []
    cache = {}  # avoid repeated lookups for the same word

    for w in words:
        # Extract the alphabetic core (preserve punctuation)
        match = re.match(r'^([^a-zA-Z]*)(.*?)([^a-zA-Z]*)$', w)
        if not match:
            result.append(w)
            continue

        prefix, core, suffix = match.groups()

        # Skip words matching skip patterns
        skip = False
        for pattern in DICTIONARY_SKIP_PATTERNS:
            if pattern.match(w):
                skip = True
                break
        if skip or len(core) < 3:
            result.append(w)
            continue

        # Check cache
        core_lower = core.lower()
        if core_lower in cache:
            replacement = cache[core_lower]
        else:
            replacement = _find_best_match(core_lower, dictionary, min_similarity)
            cache[core_lower] = replacement

        if replacement and replacement != core_lower:
            # Preserve original capitalization pattern
            if core[0].isupper() and not core.isupper():
                replacement = replacement.capitalize()
            elif core.isupper():
                replacement = replacement.upper()

            log.debug(f"  dict_match: '{core}' → '{replacement}'")
            result.append(prefix + replacement + suffix)
            metrics.dictionary_corrections += 1
        else:
            result.append(w)

    return ' '.join(result)


# ============================================================
# STAGE 11: MEDICAL SPELL CHECK (OpenMedSpel + SymSpell)
# ============================================================
# Uses the 98K-term OpenMedSpel dictionary with SymSpell's
# symmetric delete algorithm for fast, medically-aware correction.
#
# This stage runs AFTER Stage 10's character deduplication,
# catching remaining misspellings that dedup alone can't fix
# (transpositions, missing chars, single-char substitutions).
#
# Dependencies (optional — stage is skipped if unavailable):
#   pip install symspellpy
#   Medical wordlist: github.com/glutanimate/wordlist-medicalterms-en
# ============================================================

# Try to import symspellpy; stage is gracefully skipped if not installed
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False


def _load_medical_spellchecker(
    medical_wordlist_path: Optional[str] = None,
    max_edit_distance: int = 2,
) -> Optional[object]:
    """
    Build a SymSpell instance loaded with medical + English terms.
    
    Searches for the medical wordlist in this order:
      1. Explicit path from --medical-wordlist argument
      2. ./medical_wordlist.txt (current directory)
      3. ./wordlist-medicalterms-en/wordlist.txt (git clone location)
      4. ~/.medasr/medical_wordlist.txt (user config)
      5. /usr/share/medasr/medical_wordlist.txt (system-wide)
    
    Returns None if symspellpy is not installed or no wordlist is found.
    """
    if not SYMSPELL_AVAILABLE:
        log.info("symspellpy not installed — skipping medical spell check. "
                 "Install with: pip install symspellpy")
        return None

    # Search for medical wordlist
    search_paths = [
        medical_wordlist_path,
        'medical_wordlist.txt',
        'wordlist-medicalterms-en/wordlist.txt',
        os.path.expanduser('~/.medasr/medical_wordlist.txt'),
        '/usr/share/medasr/medical_wordlist.txt',
    ]

    wordlist_path = None
    for p in search_paths:
        if p and os.path.exists(p):
            wordlist_path = p
            break

    if not wordlist_path:
        log.info("Medical wordlist not found — skipping medical spell check. "
                 "Download from: https://github.com/glutanimate/wordlist-medicalterms-en")
        return None

    sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)

    # Load medical terms with higher frequency (priority in ranking)
    med_count = 0
    with open(wordlist_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            term = line.strip()
            if term and len(term) >= 2:
                # Medical terms get frequency 10 (higher priority than general English)
                sym.create_dictionary_entry(term.lower(), 10)
                med_count += 1

    # Load general English dictionary with lower frequency
    eng_count = 0
    dict_paths = [
        '/usr/share/dict/words',
        '/usr/share/dict/american-english',
        '/usr/share/dict/british-english',
    ]
    for dict_path in dict_paths:
        if os.path.exists(dict_path):
            with open(dict_path, 'r') as f:
                for line in f:
                    w = line.strip()
                    if len(w) >= 2:
                        # English words get frequency 1 (medical terms take precedence)
                        sym.create_dictionary_entry(w.lower(), 1)
                        eng_count += 1
            break

    log.info(f"Medical spell checker loaded: {med_count} medical + {eng_count} English terms")
    return sym


def medical_spellcheck(text: str, metrics: CleanupMetrics,
                       spellchecker: Optional[object] = None,
                       max_edit_distance: int = 2) -> str:
    """
    Correct remaining misspellings using the 98K-term medical spell checker.
    
    This is a CONSERVATIVE stage that only corrects words when:
      - Word is >= 5 characters (short fragments are too ambiguous)
      - Word is NOT already a known term
      - A correction is found within max_edit_distance
      - The correction looks plausible (same first letter, similar length)
    
    Medical terms are prioritized over general English via frequency weighting.
    """
    if spellchecker is None:
        return text

    words = text.split()
    result = []
    cache = {}

    for w in words:
        # Extract alphabetic core, preserve surrounding punctuation
        # Special handling: keep possessives intact ('s at end)
        match = re.match(r'^([^a-zA-Z]*)(.*?)([^a-zA-Z]*)$', w)
        if not match:
            result.append(w)
            continue

        prefix, core, suffix = match.groups()

        # Handle possessives: "Spurling's" → core="Spurling's", not "Spurling"
        if suffix.startswith("'s") or suffix.startswith("'s"):
            core = core + suffix[:2]
            suffix = suffix[2:]

        # Skip conditions: short words, markers, abbreviations, numbers
        if (len(core) < 6
                or core_lower_startswith_bracket(core)
                or core.isupper()
                or any(c.isdigit() for c in core)):
            result.append(w)
            continue

        core_lower = core.lower()

        # Skip possessives of known terms (don't strip 's from medical terms)
        if core_lower.endswith("'s") or core_lower.endswith("'s"):
            base = core_lower[:-2]
            # If the base is already a known word, skip
            exact = spellchecker.lookup(base, Verbosity.CLOSEST, max_edit_distance=0)
            if exact:
                result.append(w)
                continue

        # Check cache
        if core_lower in cache:
            correction = cache[core_lower]
        else:
            correction = _medical_correct(core_lower, spellchecker, max_edit_distance)
            cache[core_lower] = correction

        if correction and correction != core_lower:
            # Preserve original capitalization
            if core[0].isupper() and not core.isupper():
                correction = correction.capitalize()
            elif core.isupper():
                correction = correction.upper()

            log.debug(f"  med_spellcheck: '{core}' → '{correction}'")
            result.append(prefix + correction + suffix)
            metrics.medical_corrections += 1
        else:
            result.append(w)

    return ' '.join(result)


def core_lower_startswith_bracket(core: str) -> bool:
    """Helper: check if core starts with bracket/brace markers."""
    cl = core.lower()
    return cl.startswith('[') or cl.startswith('{')


def _medical_correct(word: str, sym: object, max_dist: int = 2) -> Optional[str]:
    """
    Find the best medical spelling correction for a word.
    
    Returns the correction, or None if the word is already correct
    or no confident correction is found.
    """
    # Check if word is already known (edit distance 0)
    exact = sym.lookup(word, Verbosity.CLOSEST, max_edit_distance=0)
    if exact:
        return None  # already correct

    # Look for corrections
    suggestions = sym.lookup(word, Verbosity.CLOSEST, max_edit_distance=max_dist)
    if not suggestions:
        return None

    best = suggestions[0]

    # Validation filters to avoid false positives:

    # 1. First letter must match (CTC artifacts rarely change the first letter)
    if best.term[0] != word[0]:
        # Exception: doubled first letter like "ssymptoms" → "symptoms"
        if not (len(word) >= 2 and word[0] == word[1] and best.term[0] == word[1]):
            return None

    # 2. Length must be similar (within ±3 chars)
    if abs(len(best.term) - len(word)) > 3:
        return None

    # 3. For edit distance 2, require the word to be >= 7 chars
    #    (short words with 2 edits have too many false matches)
    if best.distance == 2 and len(word) < 7:
        return None

    # 4. Skip if correction looks like a completely different word
    #    (check that at least 50% of characters positionally overlap)
    common = sum(1 for a, b in zip(word, best.term) if a == b)
    overlap = common / max(len(word), len(best.term))
    if overlap < 0.5:
        return None

    return best.term


# ============================================================
# STAGE 12: MEDASR SYSTEM ARTIFACT REMOVAL
# ============================================================
# Removes or converts MedASR-inserted content that is NOT part
# of the dictated audio:
#   - [unintelligible] markers → removed
#   - [[uh]] / [um] filler markers → removed
#   - {period} / {comma} / etc. → converted to actual punctuation
#   - {end of dictation} / {next} → removed
#
# Section headers ([ASSESSMENT], [PLAN], etc.) are PRESERVED
# as they provide useful structural context for MedGamma.
# ============================================================

# Section headers to preserve (matched case-insensitively)
KNOWN_SECTION_HEADERS = {
    'assessment', 'plan', 'physical exam', 'physical examination',
    'past medical history', 'past surgical history',
    'family history', 'social history',
    'review of systems', 'allergies', 'medications',
    'current medications', 'chief complaint', 'history of present illness',
    'impression', 'diagnosis', 'radiographs', 'imaging',
    'location', 'subjective', 'objective',
}


def _is_section_header(text_inside_brackets: str) -> bool:
    """Check if bracketed text is a known section header."""
    cleaned = text_inside_brackets.strip().lower()
    # Exact match
    if cleaned in KNOWN_SECTION_HEADERS:
        return True
    # Fuzzy match — check if it starts with a known header
    for header in KNOWN_SECTION_HEADERS:
        if cleaned.startswith(header):
            return True
    return False


def _is_unintelligible_marker(text_inside_brackets: str) -> bool:
    """
    Check if bracketed text is an [unintelligible] marker.
    Handles many garbled MedASR spellings:
      [unintelligible], [unintelligigible], [ununtellelligibleible],
      [untelligible], [intelligible], [invielligible], [unytelligal], etc.
    """
    cleaned = text_inside_brackets.strip().lower()
    # Remove doubled characters for matching
    deduped = re.sub(r'(.)\1+', r'\1', cleaned)

    # Core patterns that indicate "unintelligible"
    unintel_patterns = [
        'unintel', 'untel', 'inteligib', 'intellig', 'inteleg',
        'unytel', 'inviel', 'unintell',
    ]
    for pat in unintel_patterns:
        if pat in deduped or pat in cleaned:
            return True
    return False


def _is_filler_marker(text_inside_brackets: str) -> bool:
    """Check if bracketed text is a filler marker [uh] or [um]."""
    cleaned = text_inside_brackets.strip().lower()
    return cleaned in ('uh', 'um', 'uhh', 'umm', 'uhm')


# Format command → replacement mapping
# These are dictation commands that MedASR represents in braces.
FORMAT_COMMAND_MAP = {
    'period':         '.',
    'comma':          ',',
    'colon':          ':',
    'semicolon':      ';',
    'question mark':  '?',
    'exclamation':    '!',
    'slash':          '/',
    'hyphen':         '-',
    'dash':           ' — ',
    'new line':       '\n',
    'new paragraph':  '\n\n',
    'next':           '',        # navigation command, remove
    'end of dictation': '',      # remove
}


def _resolve_format_command(text_inside_braces: str) -> Optional[str]:
    """
    Try to match a brace-enclosed string to a known format command.
    Returns the replacement string, or None if not recognized.
    
    Handles garbled variants like 'poestion mark' → 'question mark',
    'slassh' → 'slash', 'eend of dictation' → 'end of dictation'.
    """
    cleaned = text_inside_braces.strip().lower()

    # Remove stutter artifacts from command name
    # e.g., "eend of di dictctationation" → "end of dictation"
    deduped = re.sub(r'(.{2,}?)\1+', r'\1', cleaned)
    deduped = re.sub(r'(.)\1+', r'\1', deduped)
    deduped = re.sub(r'\s+', ' ', deduped).strip()

    # Direct match
    if cleaned in FORMAT_COMMAND_MAP:
        return FORMAT_COMMAND_MAP[cleaned]

    # Match after dedup
    if deduped in FORMAT_COMMAND_MAP:
        return FORMAT_COMMAND_MAP[deduped]

    # Fuzzy substring match for common commands
    command_keywords = {
        'period': '.', 'comma': ',', 'colon': ':', 'semicolon': ';',
        'question': '?', 'slash': '/', 'hyphen': '-',
        'new paragraph': '\n\n', 'new line': '\n', 'paragraph': '\n\n',
        'next': '', 'end of dict': '', 'dictation': '',
    }
    for keyword, replacement in command_keywords.items():
        if keyword in cleaned or keyword in deduped:
            return replacement

    return None


def remove_medasr_artifacts(text: str, metrics: CleanupMetrics) -> str:
    """
    Remove or convert MedASR system-inserted artifacts.
    
    Processes three types of artifacts:
    1. Bracketed markers: [unintelligible], [uh], [um] → removed
    2. Brace format commands: {period} → ".", {comma} → "," etc.
    3. Garbled/unrecognizable brace commands → removed
    
    Section headers ([ASSESSMENT], [PLAN], etc.) are preserved.
    """
    # ---- PHASE 1: Process bracketed content [...]  ----
    # Match all bracket groups including garbled double-brackets
    def replace_brackets(m):
        full = m.group(0)
        # Extract text inside outermost brackets
        inner = re.sub(r'^\[+\s*|\s*\]+$', '', full).strip()

        # Check if it's a section header → preserve
        if _is_section_header(inner):
            # Normalize to clean single-bracket format
            header = inner.strip().upper()
            # Try to find closest known header
            for h in KNOWN_SECTION_HEADERS:
                if inner.lower().startswith(h):
                    header = h.upper()
                    break
            return f'[{header}]'

        # Check if it's an unintelligible marker → remove
        if _is_unintelligible_marker(inner):
            metrics.medasr_artifacts_removed += 1
            return ''

        # Check if it's a filler marker → remove
        if _is_filler_marker(inner):
            metrics.medasr_artifacts_removed += 1
            return ''

        # If it contains "unintelligible" mixed with other text,
        # it's usually a garbled passage → remove the marker portion
        if _is_unintelligible_marker(inner) or 'telligib' in inner.lower():
            metrics.medasr_artifacts_removed += 1
            return ''

        # Unknown bracketed content — keep it (might be meaningful)
        return full

    text = re.sub(r'\[+\s*[^\]]*?\s*\]+', replace_brackets, text)

    # Also catch standalone broken unintelligible fragments not in brackets
    # e.g., "andunintelligible", "Faceinttelligible", bare "unintelligible"
    text = re.sub(
        r'\b\w*(?:unintel|untel|inteleg|intelig)\w*\b',
        lambda m: (
            metrics.__setattr__('medasr_artifacts_removed',
                                metrics.medasr_artifacts_removed + 1) or ''
        ),
        text, flags=re.IGNORECASE
    )

    # ---- PHASE 2: Process brace format commands {...}  ----
    def replace_braces(m):
        full = m.group(0)
        inner = re.sub(r'^\{+\s*|\s*\}+$', '', full).strip()

        # Skip empty braces
        if not inner:
            metrics.medasr_artifacts_removed += 1
            return ''

        # Try to resolve as a known format command
        replacement = _resolve_format_command(inner)
        if replacement is not None:
            metrics.format_commands_resolved += 1
            return replacement

        # If the content is very long (>30 chars), it's likely a garbled
        # passage that got trapped inside braces — extract and keep the text
        if len(inner) > 30:
            # Strip the braces but keep the content
            return inner

        # Short unrecognizable brace content → remove
        metrics.medasr_artifacts_removed += 1
        return ''

    text = re.sub(r'\{+\s*[^}]*?\s*\}+', replace_braces, text)

    # ---- PHASE 3: Clean up orphaned and garbled brackets  ----
    # Remove garbled bracket groups that aren't section headers
    # e.g., [[MedicalProfes N of J H CC P DDr. [M alP [[uh]]
    def cleanup_remaining_brackets(m):
        full = m.group(0)
        inner = re.sub(r'^\[+\s*|\s*\]+$', '', full).strip()
        if _is_section_header(inner):
            return full  # keep
        if _is_filler_marker(inner):
            metrics.medasr_artifacts_removed += 1
            return ''  # already should have been caught, but safety net
        # If it's a long garbled mess with nested brackets, remove it
        if '[' in inner or ']' in inner:
            metrics.medasr_artifacts_removed += 1
            return ''
        return full

    text = re.sub(r'\[+\s*[^\]]*?\s*\]+', cleanup_remaining_brackets, text)

    # Remove orphaned bracket fragments: stray ]] or [[ not part of a pair
    text = re.sub(r'(?<!\[)\]\]+', '', text)   # orphaned ]]
    text = re.sub(r'\[+(?![A-Z\]])', '', text)  # orphaned [[ not before header

    # ---- PHASE 4: Final spacing cleanup after all removals  ----
    # Multiple spaces from removed markers
    text = re.sub(r'  +', ' ', text)
    # Space before punctuation that was inserted by format command resolution
    text = re.sub(r'\s+([.,;:!?/])', r'\1', text)
    # Double punctuation from adjacent resolved commands
    text = re.sub(r'([.,;:!?])\s*\1+', r'\1', text)
    # Blank lines from removed markers
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # Trim whitespace on each line
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text


# ============================================================
# MAIN PIPELINE
# ============================================================

def postprocess(text: str, verbose: bool = False,
                use_dictionary: bool = True,
                extra_dict_path: Optional[str] = None,
                min_similarity: float = 0.80,
                use_medical_spellcheck: bool = True,
                medical_wordlist_path: Optional[str] = None,
                max_edit_distance: int = 2) -> Tuple[str, CleanupMetrics]:
    """
    Run the full post-processing pipeline.
    
    Returns: (cleaned_text, metrics)
    """
    metrics = CleanupMetrics()
    metrics.words_before = len(text.split())

    if verbose:
        log.setLevel(logging.DEBUG)

    # Pre-load dictionary once if using dictionary matching
    dictionary = None
    if use_dictionary:
        dictionary = _load_dictionary(extra_dict_path)

    # Pre-load medical spellchecker once if using medical spell check
    med_spellchecker = None
    if use_medical_spellcheck:
        med_spellchecker = _load_medical_spellchecker(
            medical_wordlist_path=medical_wordlist_path,
            max_edit_distance=max_edit_distance)

    # Stage 0: Offensive ASR misrecognition filter
    text = remove_offensive_misrecognitions(text, metrics)
    if metrics.offensive_misrecognitions_removed:
        log.warning(
            f"Stage 0: removed {metrics.offensive_misrecognitions_removed} "
            f"offensive ASR misrecognition(s) from transcript"
        )

    # Stage 1: Format command doubles
    text = fix_format_command_doubles(text, metrics)
    log.debug(f"After stage 1 (format cmds): {metrics.format_cmd_doubles} fixes")

    # Stage 2: Punctuation doubles
    text = fix_punctuation_doubles(text, metrics)
    log.debug(f"After stage 2 (punctuation): {metrics.punctuation_doubles} fixes")

    # Stage 3: CTC stutter pair merging (MOST IMPACTFUL)
    text = merge_stutter_pairs(text, metrics)
    log.debug(f"After stage 3 (stutters): {metrics.stutter_pairs_merged} pairs merged")

    # Stage 4: Internal character stuttering
    text = fix_char_stutters(text, metrics)
    log.debug(f"After stage 4 (char stutters): {metrics.char_stutters_fixed} fixed")

    # Stage 5: Filler word normalization
    text = normalize_fillers(text, metrics)
    log.debug(f"After stage 5 (fillers): {metrics.fillers_normalized} normalized")

    # Stage 6: Broken word merging (single-char fragments)
    text = merge_broken_words(text, metrics)
    log.debug(f"After stage 6 (broken words): {metrics.broken_words_merged} merged")

    # Stage 7: Section header cleanup
    text = clean_section_headers(text, metrics)
    log.debug(f"After stage 7 (headers): {metrics.headers_cleaned} cleaned")

    # Stage 8: Whitespace normalization
    text = normalize_whitespace(text, metrics)

    # Stage 9: Trailing artifacts
    text = remove_trailing_artifacts(text, metrics)

    # Stage 10: Dictionary matching (correct remaining misspellings)
    if use_dictionary:
        text = dictionary_match(text, metrics, dictionary=dictionary,
                                min_similarity=min_similarity)
        log.debug(f"After stage 10 (dictionary): {metrics.dictionary_corrections} corrections")

    # Stage 11: Medical spell check (OpenMedSpel + SymSpell)
    if med_spellchecker is not None:
        text = medical_spellcheck(text, metrics, spellchecker=med_spellchecker,
                                  max_edit_distance=max_edit_distance)
        log.debug(f"After stage 11 (medical): {metrics.medical_corrections} corrections")

    # Stage 12: MedASR system artifact removal
    text = remove_medasr_artifacts(text, metrics)
    log.debug(f"After stage 12 (MedASR artifacts): "
              f"{metrics.medasr_artifacts_removed} removed, "
              f"{metrics.format_commands_resolved} format cmds resolved")

    metrics.words_after = len(text.split())
    return text, metrics


def generate_diff_report(original: str, cleaned: str, metrics: CleanupMetrics) -> str:
    """Generate a before/after comparison report."""
    lines = [
        "=" * 70,
        "MEDASR POST-PROCESSOR REPORT",
        "=" * 70,
        "",
        metrics.summary(),
        "",
        "=" * 70,
        "BEFORE (first 500 chars):",
        "=" * 70,
        original[:500],
        "",
        "=" * 70,
        "AFTER (first 500 chars):",
        "=" * 70,
        cleaned[:500],
        "",
    ]
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def process_file(input_path: str, output_path: str, verbose: bool = False, 
                 report: bool = False, use_dictionary: bool = True,
                 extra_dict_path: Optional[str] = None,
                 min_similarity: float = 0.80,
                 use_medical_spellcheck: bool = True,
                 medical_wordlist_path: Optional[str] = None,
                 max_edit_distance: int = 2) -> CleanupMetrics:
    """Process a single file."""
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        original = f.read()

    cleaned, metrics = postprocess(original, verbose=verbose,
                                   use_dictionary=use_dictionary,
                                   extra_dict_path=extra_dict_path,
                                   min_similarity=min_similarity,
                                   use_medical_spellcheck=use_medical_spellcheck,
                                   medical_wordlist_path=medical_wordlist_path,
                                   max_edit_distance=max_edit_distance)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    log.info(f"Processed: {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    log.info(f"  Stutters: {metrics.stutter_pairs_merged}, "
             f"Char: {metrics.char_stutters_fixed}, "
             f"Dict: {metrics.dictionary_corrections}, "
             f"Med: {metrics.medical_corrections}, "
             f"Artifacts: {metrics.medasr_artifacts_removed}, "
             f"FmtCmds: {metrics.format_commands_resolved}, "
             f"Words: {metrics.words_before}→{metrics.words_after}")

    if report:
        report_path = output_path.replace('.txt', '_report.txt')
        report_text = generate_diff_report(original, cleaned, metrics)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        log.info(f"  Report: {report_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="MedASR Transcript Post-Processor: Clean CTC artifacts before MedGamma LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.txt -o cleaned.txt
  %(prog)s input.txt -o cleaned.txt --report
  %(prog)s /transcripts/ -o /cleaned/ --batch
  %(prog)s input.txt -o cleaned.txt --extra-dict my_terms.txt
  %(prog)s input.txt -o cleaned.txt --no-dict
  %(prog)s input.txt -o cleaned.txt --medical-wordlist /path/to/wordlist.txt
  %(prog)s input.txt -o cleaned.txt --no-medical-spellcheck
  cat input.txt | %(prog)s - > cleaned.txt
        """
    )
    parser.add_argument('input', help='Input file, directory (with --batch), or - for stdin')
    parser.add_argument('-o', '--output', help='Output file or directory')
    parser.add_argument('--batch', action='store_true', help='Process all .txt files in input directory')
    parser.add_argument('--report', action='store_true', help='Generate before/after diff report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed processing logs')
    parser.add_argument('--suffix', default='_cleaned', help='Suffix for output files in batch mode (default: _cleaned)')
    parser.add_argument('--no-dict', action='store_true', help='Disable dictionary matching stage (Stage 10)')
    parser.add_argument('--extra-dict', metavar='PATH', help='Path to additional dictionary file (one word per line)')
    parser.add_argument('--min-similarity', type=float, default=0.80,
                        help='Minimum similarity for fuzzy dictionary matching (0-1, default: 0.80)')
    parser.add_argument('--no-medical-spellcheck', action='store_true',
                        help='Disable medical spell check stage (Stage 11)')
    parser.add_argument('--medical-wordlist', metavar='PATH',
                        help='Path to medical wordlist file (default: auto-detect). '
                             'Download from: https://github.com/glutanimate/wordlist-medicalterms-en')
    parser.add_argument('--max-edit-distance', type=int, default=2,
                        help='Max edit distance for medical spell check (1 or 2, default: 2)')

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    use_dict = not args.no_dict
    extra_dict = args.extra_dict
    min_sim = args.min_similarity
    use_med = not args.no_medical_spellcheck
    med_wordlist = args.medical_wordlist
    max_ed = args.max_edit_distance

    # Common kwargs for postprocess / process_file
    pipeline_kwargs = dict(
        use_dictionary=use_dict,
        extra_dict_path=extra_dict,
        min_similarity=min_sim,
        use_medical_spellcheck=use_med,
        medical_wordlist_path=med_wordlist,
        max_edit_distance=max_ed,
    )

    # Stdin mode
    if args.input == '-':
        original = sys.stdin.read()
        cleaned, metrics = postprocess(original, verbose=args.verbose,
                                       **pipeline_kwargs)
        sys.stdout.write(cleaned)
        log.info(metrics.summary())
        return

    # Batch mode
    if args.batch:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            log.error(f"Input must be a directory in batch mode: {args.input}")
            sys.exit(1)

        output_dir = Path(args.output) if args.output else input_dir / "cleaned"
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("*.txt"))
        if not files:
            log.error(f"No .txt files found in {input_dir}")
            sys.exit(1)

        total_metrics = CleanupMetrics()
        for f in files:
            out_name = f.stem + args.suffix + f.suffix
            out_path = output_dir / out_name
            m = process_file(str(f), str(out_path), verbose=args.verbose,
                             report=args.report, **pipeline_kwargs)
            total_metrics.format_cmd_doubles += m.format_cmd_doubles
            total_metrics.punctuation_doubles += m.punctuation_doubles
            total_metrics.stutter_pairs_merged += m.stutter_pairs_merged
            total_metrics.char_stutters_fixed += m.char_stutters_fixed
            total_metrics.fillers_normalized += m.fillers_normalized
            total_metrics.broken_words_merged += m.broken_words_merged
            total_metrics.headers_cleaned += m.headers_cleaned
            total_metrics.whitespace_fixes += m.whitespace_fixes
            total_metrics.trailing_artifacts += m.trailing_artifacts
            total_metrics.dictionary_corrections += m.dictionary_corrections
            total_metrics.medical_corrections += m.medical_corrections
            total_metrics.medasr_artifacts_removed += m.medasr_artifacts_removed
            total_metrics.format_commands_resolved += m.format_commands_resolved
            total_metrics.words_before += m.words_before
            total_metrics.words_after += m.words_after

        log.info(f"\n{'='*50}")
        log.info(f"BATCH COMPLETE: {len(files)} files processed")
        log.info(total_metrics.summary())
        return

    # Single file mode
    if not os.path.isfile(args.input):
        log.error(f"File not found: {args.input}")
        sys.exit(1)

    output = args.output or args.input.replace('.txt', '_cleaned.txt')
    process_file(args.input, output, verbose=args.verbose, report=args.report,
                 **pipeline_kwargs)


if __name__ == '__main__':
    main()
