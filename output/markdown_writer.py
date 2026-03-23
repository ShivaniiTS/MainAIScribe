"""
Markdown writer — renders pipeline results as properly formatted .md files.

All clinical notes are written with:
  - Metadata header (sample ID, engine, version, date, scores)
  - Section headings as ## H2
  - Footer with pipeline metrics

Usage:
    from output.markdown_writer import write_clinical_note, write_transcript
    write_clinical_note(state, path, version="v1", sample_id="224889")
    write_transcript(state, path, sample_id="224889")
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from orchestrator.state import EncounterState


# Section label display names (storage key → display label)
_SECTION_LABELS: dict[str, str] = {
    "subjective":              "Subjective",
    "objective":               "Objective",
    "assessment":              "Assessment",
    "plan":                    "Plan",
    "assessment_and_plan":     "Assessment and Plan",
    "chief_complaint":         "Chief Complaint",
    "history_of_present_illness": "History of Present Illness",
    "past_medical_history":    "Past Medical History",
    "past_surgical_history":   "Past Surgical History",
    "medications":             "Current Medications",
    "allergies":               "Allergies",
    "family_history":          "Family History",
    "social_history":          "Social History",
    "review_of_systems":       "Review of Systems",
    "physical_examination":    "Physical Examination",
    "examination":             "Examination",
    "interval_history":        "Interval History",
    "imaging":                 "Imaging / Diagnostics",
}


def _label(section_type: str) -> str:
    return _SECTION_LABELS.get(section_type, section_type.replace("_", " ").title())


def write_clinical_note(
    state: EncounterState,
    path: Path,
    version: str = "v1",
    sample_id: str = "",
) -> None:
    """
    Render the final clinical note from EncounterState as a Markdown file.

    Args:
        state:     Final EncounterState (post-pipeline).
        path:      Output file path (should end in .md).
        version:   Pipeline version label (v1, v2, v3).
        sample_id: Human-readable sample identifier.
    """
    note = state.final_note or state.generated_note
    if not note:
        path.write_text("# Clinical Note\n\n*No note generated.*\n")
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    provider = state.provider_profile

    # Build demographic header from context packet if available
    ctx = state.context_packet
    header_parts: list[str] = []
    if ctx and ctx.patient and ctx.patient.name:
        p = ctx.patient
        line1_parts = []
        if p.name:
            line1_parts.append(f"**Patient:** {p.name}")
        if p.dob:
            line1_parts.append(f"**DOB:** {p.dob}")
        if p.sex:
            line1_parts.append(f"**Sex:** {p.sex}")
        if p.mrn:
            line1_parts.append(f"**MRN:** {p.mrn}")
        header_parts.append(" | ".join(line1_parts))

    if ctx and ctx.encounter:
        e = ctx.encounter
        line2_parts = []
        if e.date_of_service:
            line2_parts.append(f"**Date of Service:** {e.date_of_service}")
        if e.visit_type:
            line2_parts.append(f"**Visit Type:** {e.visit_type.replace('_', ' ').title()}")
        if e.date_of_injury:
            line2_parts.append(f"**Date of Injury:** {e.date_of_injury}")
        if e.case_number:
            line2_parts.append(f"**Case:** {e.case_number}")
        if line2_parts:
            header_parts.append(" | ".join(line2_parts))

    if ctx and ctx.provider_context and ctx.provider_context.name:
        prov = ctx.provider_context
        prov_str = f"**Provider:** {prov.name}"
        if prov.credentials:
            prov_str += f", {prov.credentials}"
        if prov.specialty:
            prov_str += f" | **Specialty:** {prov.specialty}"
        header_parts.append(prov_str)
    else:
        header_parts.append(
            f"**Provider:** {provider.name} | **Specialty:** {provider.specialty.title()}"
        )

    if ctx and ctx.facility and (ctx.facility.name or ctx.facility.location):
        fac = ctx.facility
        fac_parts = []
        if fac.name:
            fac_parts.append(fac.name)
        if fac.location:
            fac_parts.append(fac.location)
        header_parts.append(f"**Facility:** {', '.join(fac_parts)}")

    lines: list[str] = [
        f"# Clinical Note — {sample_id or state.patient_id}",
        "",
    ]
    lines.extend(header_parts)
    lines += [
        f"**Pipeline Version:** {version} | "
        f"**ASR:** {state.asr_engine_used or '—'} | "
        f"**LLM:** {state.llm_engine_used or '—'}",
        "",
        "---",
        "",
    ]

    for section in note.sections:
        lines.append(f"## {_label(section.type)}")
        lines.append("")
        lines.append(section.content.strip())
        lines.append("")

    lines += [
        "---",
        "",
        f"*AI Scribe {version} | "
        f"ASR conf: {state.metrics.asr_confidence or '—'} | "
        f"Note conf: {state.metrics.note_confidence or '—'} | "
        f"PP corrections: {state.metrics.postprocessor_corrections or 0}*",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def write_transcript(
    state: EncounterState,
    path: Path,
    sample_id: str = "",
) -> None:
    """
    Write the post-processed transcript as a Markdown file with speaker labels.

    Args:
        state:     Final EncounterState.
        path:      Output file path (should end in .md).
        sample_id: Human-readable sample identifier.
    """
    transcript = state.transcript
    if not transcript:
        path.write_text("# Transcript\n\n*No transcript.*\n")
        return

    lines: list[str] = [
        f"# Transcript — {sample_id or state.patient_id}",
        "",
        f"**ASR Engine:** {state.asr_engine_used or '—'} | "
        f"**Duration:** {transcript.audio_duration_ms / 1000:.1f}s | "
        f"**Confidence:** {state.metrics.asr_confidence or '—'} | "
        f"**Post-processor:** {state.postprocessor_version or 'none'}",
        "",
        "---",
        "",
        "## Full Text (Post-Processed)",
        "",
        transcript.full_text.strip(),
        "",
    ]

    # Speaker-segmented view (if diarization was applied)
    speaker_segs = [s for s in transcript.segments if s.speaker]
    if speaker_segs:
        lines += [
            "---",
            "",
            "## Speaker Segments",
            "",
        ]
        current_speaker = None
        for seg in transcript.segments:
            spk = seg.speaker or "SPEAKER_00"
            ts = f"[{seg.start_ms // 1000:02d}:{(seg.start_ms % 1000) // 10:02d}]"
            if spk != current_speaker:
                lines.append(f"\n**{spk}** {ts}")
                current_speaker = spk
            lines.append(seg.text.strip())

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
