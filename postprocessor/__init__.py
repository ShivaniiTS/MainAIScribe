"""
ASR post-processing package.

The main entry point is `run_postprocessor(text, config)` which wraps
medasr_postprocessor.postprocess() with a clean dict-in / dict-out interface
for the transcribe node.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

_WORDLIST_PATH = Path(__file__).parent / "medical_wordlist.txt"


def run_postprocessor(
    text: str,
    use_medical_spellcheck: bool = True,
    min_similarity: float = 0.80,
    max_edit_distance: int = 2,
    medical_wordlist_path: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Run the 12-stage MedASR post-processing pipeline on a transcript string.

    Args:
        text:                   Raw ASR transcript text.
        use_medical_spellcheck: Enable OpenMedSpel medical spell check (slower).
        min_similarity:         Minimum Levenshtein similarity for dictionary match.
        max_edit_distance:      Max edit distance for medical spell check.
        medical_wordlist_path:  Override path to medical_wordlist.txt.

    Returns:
        Tuple of (cleaned_text, metrics_dict).
        metrics_dict contains counts from all 12 stages.
    """
    from postprocessor.medasr_postprocessor import postprocess

    wordlist = medical_wordlist_path or (
        str(_WORDLIST_PATH) if _WORDLIST_PATH.exists() else None
    )

    cleaned_text, metrics = postprocess(
        text,
        use_dictionary=True,
        use_medical_spellcheck=use_medical_spellcheck and wordlist is not None,
        medical_wordlist_path=wordlist,
        min_similarity=min_similarity,
        max_edit_distance=max_edit_distance,
    )

    return cleaned_text, asdict(metrics)
