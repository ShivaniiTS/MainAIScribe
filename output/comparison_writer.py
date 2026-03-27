"""
Comparison writer — renders side-by-side gold vs generated Markdown comparison tables.

Output format (per CLAUDE.md spec):

    # Comparison — sample_001
    **Version:** v1 (Session 4) vs Gold Standard

    ## Subjective
    | Gold | Generated |
    |------|-----------|
    | ... | ... |

    ## Summary
    | Metric | Value |
    |--------|-------|
    | Keyword overlap | 40% |

Usage:
    from output.comparison_writer import write_comparison
    write_comparison(path, sample_id, generated_note_text, gold_note_text,
                     transcript=..., metrics={...}, version="v1")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ── Section extraction ──────────────────────────────────────────────────────

# Handles both gold note format (SUBJECTIVE:) and generated format (## Subjective)
_SECTION_NAMES = [
    "SUBJECTIVE", "OBJECTIVE", "ASSESSMENT AND PLAN", "ASSESSMENT", "PLAN",
    "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "HPI",
    "PAST MEDICAL HISTORY", "PAST SURGICAL HISTORY",
    "CURRENT MEDICATIONS", "MEDICATIONS", "ALLERGIES",
    "FAMILY HISTORY", "SOCIAL HISTORY",
    "REVIEW OF SYSTEMS", "PHYSICAL EXAMINATION", "PHYSICAL EXAM",
    "EXAMINATION", "INTERVAL HISTORY",
    "IMAGING / DIAGNOSTICS", "IMAGING AND DIAGNOSTICS", "IMAGING",
    "DIAGNOSTIC EXAM", "DIAGNOSTICS",
]

_SECTION_ALIASES: dict[str, str] = {
    "SUBJECTIVE": "Subjective",
    "OBJECTIVE": "Objective",
    "ASSESSMENT": "Assessment",
    "PLAN": "Plan",
    "ASSESSMENT AND PLAN": "Assessment and Plan",
    "CHIEF COMPLAINT": "Chief Complaint",
    "HISTORY OF PRESENT ILLNESS": "History of Present Illness",
    "HPI": "History of Present Illness",
    "PAST MEDICAL HISTORY": "Past Medical History",
    "PAST SURGICAL HISTORY": "Past Surgical History",
    "CURRENT MEDICATIONS": "Current Medications",
    "MEDICATIONS": "Current Medications",
    "ALLERGIES": "Allergies",
    "FAMILY HISTORY": "Family History",
    "SOCIAL HISTORY": "Social History",
    "REVIEW OF SYSTEMS": "Review of Systems",
    "PHYSICAL EXAMINATION": "Physical Examination",
    "PHYSICAL EXAM": "Physical Examination",
    "EXAMINATION": "Examination",
    "INTERVAL HISTORY": "Interval History",
    "IMAGING / DIAGNOSTICS": "Imaging / Diagnostics",
    "IMAGING AND DIAGNOSTICS": "Imaging / Diagnostics",
    "IMAGING": "Imaging / Diagnostics",
    "DIAGNOSTIC EXAM": "Imaging / Diagnostics",
    "DIAGNOSTICS": "Imaging / Diagnostics",
}


def _extract_sections(text: str) -> dict[str, str]:
    """
    Extract SOAP/clinical sections from a note string.

    Handles:
        SUBJECTIVE:  text...      (gold note format)
        ## Subjective             (generated markdown format)
        **SUBJECTIVE:**           (bold markdown format)
    """
    escaped = [re.escape(s) for s in _SECTION_NAMES]
    pattern = re.compile(
        r"^(?:\*{1,2}|#{1,3}\s*)?("
        + "|".join(escaped)
        + r")(?:\s*\([^)]*\))?[\s:*#]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        raw = match.group(1).strip().upper()
        label = _SECTION_ALIASES.get(raw, raw.title())
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[label] = content
    return sections


def _md_table_row(*cells: str) -> str:
    """Render a Markdown table row, escaping pipe characters in cell content."""
    escaped = [c.replace("|", "\\|").replace("\n", " ") for c in cells]
    return "| " + " | ".join(escaped) + " |"


def _truncate(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


# ── Keyword overlap ─────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was",
    "for", "she", "her", "he", "his", "with", "has", "have", "be", "that",
    "this", "from", "as", "at", "by", "on", "not", "no", "but", "it", "its",
    "will", "may", "can", "we", "our", "patient", "presents", "today", "due",
    "also", "been", "were", "they", "them", "their", "which", "who", "what",
    "after", "about", "over", "than", "into", "upon", "would", "could",
    "should", "some", "such", "each", "further", "there", "here",
}


def _keywords(text: str, min_len: int = 4) -> set[str]:
    return {
        w.lower()
        for w in re.findall(r"[a-z]+", text.lower())
        if len(w) >= min_len and w.lower() not in _STOPWORDS
    }


def _keyword_overlap(generated: str, reference: str, top_n: int = 50) -> float:
    ref_kw = _keywords(reference)
    gen_kw = _keywords(generated)
    if not ref_kw:
        return 0.0
    important = sorted(ref_kw, key=len, reverse=True)[:top_n]
    matched = sum(1 for w in important if w in gen_kw)
    return matched / len(important)


def _missing_keywords(generated: str, reference: str, top_n: int = 20) -> list[str]:
    ref_kw = _keywords(reference)
    gen_kw = _keywords(generated)
    important = sorted(ref_kw, key=len, reverse=True)[:top_n]
    return [w for w in important if w not in gen_kw]


# ── Main writer ─────────────────────────────────────────────────────────────

def write_comparison(
    path: Path,
    sample_id: str,
    generated_note: str,
    gold_note: str,
    transcript: str = "",
    metrics: Optional[dict] = None,
    version: str = "v1",
    session: int = 4,
) -> float:
    """
    Write a side-by-side Markdown comparison of generated vs gold note.

    Args:
        path:           Output file path (should end in .md).
        sample_id:      Human-readable sample identifier.
        generated_note: Full generated note text.
        gold_note:      Full gold-standard note text.
        transcript:     Post-processed transcript text (optional).
        metrics:        Dict of pipeline metrics (optional).
        version:        Pipeline version label (v1, v2, v3).
        session:        Session number.

    Returns:
        Keyword overlap score (0–1).
    """
    metrics = metrics or {}
    overlap = _keyword_overlap(generated_note, gold_note) if gold_note else 0.0
    missing = _missing_keywords(generated_note, gold_note)

    gen_sections = _extract_sections(generated_note)
    gold_sections = _extract_sections(gold_note)
    all_section_labels = list(dict.fromkeys(
        list(gold_sections.keys()) + list(gen_sections.keys())
    ))

    lines: list[str] = [
        f"# Comparison — {sample_id}",
        f"**Version:** {version} (Session {session}) vs Gold Standard  ",
        f"**Keyword Overlap:** {overlap:.0%}  ",
        "",
        "---",
        "",
    ]

    # Section-by-section table
    for label in all_section_labels:
        gold_text = gold_sections.get(label, "*— not present —*")
        gen_text = gen_sections.get(label, "*— not present —*")
        lines += [
            f"## {label}",
            "",
            "| Gold Standard | Generated |",
            "|---------------|-----------|",
            _md_table_row(_truncate(gold_text), _truncate(gen_text)),
            "",
        ]

    # Summary table
    lines += [
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Keyword overlap | {overlap:.0%} |",
        f"| Missing keywords | {', '.join(missing[:10]) if missing else '—'} |",
        f"| ASR engine | {metrics.get('asr_engine', '—')} |",
        f"| ASR confidence | {metrics.get('asr_conf', '—')} |",
        f"| Note confidence | {metrics.get('note_conf', '—')} |",
        f"| PP corrections | {metrics.get('pp_corrections', '—')} |",
        f"| ASR duration | {metrics.get('asr_ms', '—')} ms |",
        f"| LLM duration | {metrics.get('llm_ms', '—')} ms |",
        "",
    ]

    # Transcript (collapsed, optional)
    if transcript:
        lines += [
            "---",
            "",
            "<details>",
            "<summary>Transcript (post-processed)</summary>",
            "",
            transcript.strip(),
            "",
            "</details>",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return overlap
