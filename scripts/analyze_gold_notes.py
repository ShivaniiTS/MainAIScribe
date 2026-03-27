#!/usr/bin/env python3
"""
Analyze gold-standard notes to extract structure patterns and generate
specialty-specific template YAML files.

What it does:
    1. Reads all gold notes from data/dictation/ (soap_final.md) and
       data/conversations/ (soap_initial.md)
    2. Sends each note to the LLM to extract: sections present, section order,
       style patterns, specialty indicators
    3. Aggregates findings across all notes
    4. Generates YAML template files in config/templates/ for detected note types

Output:
    config/templates/<name>.yaml  — one per detected template type
    output/gold_analysis.md       — human-readable analysis report

Usage:
    python scripts/analyze_gold_notes.py
    python scripts/analyze_gold_notes.py --data-dir data/dictation
    python scripts/analyze_gold_notes.py --dry-run   # print templates, don't write
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import DATA_DIR as _DATA_ROOT, OUTPUT_DIR as _OUTPUT_DIR, CONFIG_DIR

_MODES = ("conversation", "dictation")
_TEMPLATES_DIR = CONFIG_DIR / "templates"


def _collect_gold_notes(data_root: Path) -> list[tuple[Path, str]]:
    """Return list of (note_path, source_type) for all gold notes found.

    Walks: data_root/<mode>/<physician>/<encounter>/final_soap_note.md
    """
    notes = []
    for mode in _MODES:
        mode_dir = data_root / mode
        if not mode_dir.exists():
            continue
        source = "dictation" if mode == "dictation" else "conversation"
        for physician_dir in sorted(mode_dir.iterdir()):
            if not physician_dir.is_dir():
                continue
            for encounter_dir in sorted(physician_dir.iterdir()):
                if not encounter_dir.is_dir():
                    continue
                gold = encounter_dir / "final_soap_note.md"
                if gold.exists():
                    notes.append((gold, source))
    return notes


def _extract_sections_with_llm(note_text: str, engine) -> dict:
    """
    Ask the LLM to extract structured metadata from a gold note.

    Returns dict with keys: sections (ordered list), specialty, visit_type,
    style_notes, header_fields
    """
    from mcp_servers.llm.base import LLMConfig, LLMMessage

    system = (
        "You are a medical documentation analyst. Analyze the provided clinical note "
        "and extract its structure. Respond with ONLY valid JSON, no commentary."
    )

    prompt = f"""Analyze this clinical note and extract its structure.

CLINICAL NOTE:
{note_text[:4000]}

Respond with valid JSON only:
{{
  "specialty": "orthopedic|cardiology|general|...",
  "visit_type": "initial_evaluation|follow_up|progress_note|...",
  "sections": [
    {{"label": "SECTION NAME AS WRITTEN", "id": "snake_case_id", "required": true}}
  ],
  "header_fields": ["patient_name", "date_of_service", ...],
  "style_notes": ["third person", "past tense", "numbered assessments", ...]
}}"""

    cfg = LLMConfig(model=None, temperature=0.0, max_tokens=800)
    try:
        response = engine.generate_sync(
            system_prompt=system,
            messages=[LLMMessage(role="user", content=prompt)],
            config=cfg,
            task="command_parse",
        )
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception as exc:
        print(f"  [warn] LLM extraction failed: {exc}")
        return {}


def _aggregate(analyses: list[dict]) -> dict[str, list[dict]]:
    """
    Group analyses by (specialty, visit_type) and aggregate section patterns.

    Returns dict mapping "specialty/visit_type" → aggregated data.
    """
    groups: dict[str, list[dict]] = {}
    for a in analyses:
        if not a:
            continue
        key = f"{a.get('specialty', 'general')}/{a.get('visit_type', 'default')}"
        groups.setdefault(key, []).append(a)
    return groups


def _build_template(group_key: str, group: list[dict]) -> dict:
    """Build a template dict from a group of analyses for the same note type."""
    specialty, visit_type = group_key.split("/", 1)

    # Count section occurrences across all notes in group
    section_counts: Counter = Counter()
    section_order: dict[str, list[int]] = {}
    header_fields: Counter = Counter()
    style_notes: Counter = Counter()

    for analysis in group:
        sections = analysis.get("sections", [])
        for i, sec in enumerate(sections):
            label = sec.get("label", "").upper().strip()
            if label:
                section_counts[label] += 1
                section_order.setdefault(label, []).append(i)
        for hf in analysis.get("header_fields", []):
            header_fields[hf] += 1
        for note in analysis.get("style_notes", []):
            style_notes[note] += 1

    n = len(group)
    threshold = max(1, n // 2)  # section must appear in ≥50% of notes

    # Build ordered section list
    ordered_sections = sorted(
        [(label, sum(positions) / len(positions)) for label, positions in section_order.items()
         if section_counts[label] >= threshold],
        key=lambda x: x[1],
    )

    # Build standard section ids
    _label_to_id = {
        "SUBJECTIVE": "subjective",
        "OBJECTIVE": "objective",
        "ASSESSMENT": "assessment",
        "PLAN": "plan",
        "ASSESSMENT AND PLAN": "assessment_and_plan",
        "CHIEF COMPLAINT": "chief_complaint",
        "HISTORY OF PRESENT ILLNESS": "history_of_present_illness",
        "HPI": "history_of_present_illness",
        "PAST MEDICAL HISTORY": "past_medical_history",
        "PMH": "past_medical_history",
        "PAST SURGICAL HISTORY": "past_surgical_history",
        "PSH": "past_surgical_history",
        "MEDICATIONS": "medications",
        "CURRENT MEDICATIONS": "medications",
        "ALLERGIES": "allergies",
        "FAMILY HISTORY": "family_history",
        "SOCIAL HISTORY": "social_history",
        "REVIEW OF SYSTEMS": "review_of_systems",
        "ROS": "review_of_systems",
        "PHYSICAL EXAMINATION": "physical_examination",
        "PHYSICAL EXAM": "physical_examination",
        "EXAMINATION": "examination",
        "IMAGING": "imaging",
        "IMAGING / DIAGNOSTICS": "imaging",
        "INTERVAL HISTORY": "interval_history",
    }

    sections_out = []
    for label, _ in ordered_sections:
        sec_id = _label_to_id.get(label, label.lower().replace(" ", "_").replace("/", "_"))
        required = section_counts[label] >= max(1, int(n * 0.75))
        sections_out.append({
            "id": sec_id,
            "label": label,
            "required": required,
        })

    # Header fields present in >50% of notes
    headers_out = [hf for hf, cnt in header_fields.most_common() if cnt >= threshold]

    return {
        "name": f"{specialty.title()} {visit_type.replace('_', ' ').title()}",
        "specialty": specialty,
        "visit_type": visit_type,
        "header_fields": headers_out or ["patient_name", "date_of_service", "provider_name"],
        "sections": sections_out,
        "formatting": {"voice": "active", "tense": "past", "person": "third",
                       "abbreviations": "spell_out", "measurements": "include_units"},
        "style_notes": [note for note, cnt in style_notes.most_common(5)],
    }


def _template_to_yaml(tpl: dict) -> str:
    """Render a template dict to YAML string (without pyyaml dependency for clean output)."""
    lines = [
        f"# Auto-generated from gold note analysis — edit as needed",
        f"name: \"{tpl['name']}\"",
        f"specialty: {tpl['specialty']}",
        f"visit_type: {tpl['visit_type']}",
        "",
        "header_fields:",
    ]
    for hf in tpl["header_fields"]:
        lines.append(f"  - {hf}")
    lines += ["", "sections:"]
    for sec in tpl["sections"]:
        lines.append(f"  - id: {sec['id']}")
        lines.append(f"    label: \"{sec['label']}\"")
        lines.append(f"    required: {'true' if sec['required'] else 'false'}")
    lines += [
        "",
        "formatting:",
    ]
    for k, v in tpl["formatting"].items():
        lines.append(f"  {k}: {v}")
    if tpl.get("style_notes"):
        lines += ["", "# Style patterns observed in gold notes:"]
        for note in tpl["style_notes"]:
            lines.append(f"# - {note}")
    lines.append("")
    return "\n".join(lines)


def _write_report(analyses_by_file: list[tuple[str, str, dict]], groups: dict, out_path: Path) -> None:
    lines = [
        "# Gold Note Analysis Report",
        "",
        f"**Notes analyzed:** {len(analyses_by_file)}",
        f"**Template groups detected:** {len(groups)}",
        "",
        "---",
        "",
        "## Per-Note Extraction",
        "",
        "| File | Source | Specialty | Visit Type | Sections |",
        "|------|--------|-----------|------------|---------|",
    ]
    for fname, source, analysis in analyses_by_file:
        spec = analysis.get("specialty", "—")
        vtype = analysis.get("visit_type", "—")
        secs = ", ".join(s["label"] for s in analysis.get("sections", []))
        lines.append(f"| {fname} | {source} | {spec} | {vtype} | {secs} |")

    lines += ["", "---", "", "## Template Groups", ""]
    for key, group in groups.items():
        lines.append(f"### {key} ({len(group)} notes)")
        lines.append("")
        # Aggregate section frequency
        counts: Counter = Counter()
        for a in group:
            for s in a.get("sections", []):
                counts[s["label"].upper()] += 1
        for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = int(cnt / len(group) * 100)
            lines.append(f"- {label}: {cnt}/{len(group)} ({pct}%)")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Analysis report : {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Single data dir (default: both)")
    parser.add_argument("--dry-run", action="store_true", help="Print templates, don't write files")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else _DATA_ROOT
    notes = _collect_gold_notes(data_root)

    if not notes:
        print("No gold notes found.")
        sys.exit(1)

    print(f"Found {len(notes)} gold notes")

    # Set up LLM engine
    from orchestrator.nodes.note_node import set_llm_engine_factory
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = resp.json().get("models", [])
        model = models[0]["name"] if models else "qwen2.5:32b"
    except Exception:
        model = "qwen2.5:32b"

    from mcp_servers.llm.ollama_server import OllamaServer
    engine = OllamaServer(model_overrides={"command_parse": model, "note_generation": model})
    set_llm_engine_factory(lambda: engine)
    print(f"LLM model      : {model}")
    print()

    analyses_by_file: list[tuple[str, str, dict]] = []
    for i, (note_path, source) in enumerate(notes, 1):
        print(f"[{i}/{len(notes)}] {note_path} ({source}) ... ", end="", flush=True)
        text = note_path.read_text()
        analysis = _extract_sections_with_llm(text, engine)
        if analysis:
            print(f"{analysis.get('specialty', '?')}/{analysis.get('visit_type', '?')} "
                  f"— {len(analysis.get('sections', []))} sections")
        else:
            print("extraction failed")
        analyses_by_file.append((note_path.name, source, analysis))

    valid = [a for _, _, a in analyses_by_file if a]
    groups = _aggregate(valid)

    print(f"\nDetected {len(groups)} template groups: {list(groups.keys())}")
    print()

    # Generate and write templates
    _TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for group_key, group in groups.items():
        tpl = _build_template(group_key, group)
        yaml_str = _template_to_yaml(tpl)
        fname = f"{tpl['specialty']}_{tpl['visit_type']}.yaml".replace(" ", "_")
        out_path = _TEMPLATES_DIR / fname

        if args.dry_run:
            print(f"=== {fname} ===")
            print(yaml_str)
        else:
            out_path.write_text(yaml_str)
            print(f"Written : {out_path}")
            written.append(fname)

    _write_report(analyses_by_file, groups, Path(args.output_dir) / "gold_analysis.md")

    if written:
        print(f"\n{len(written)} template(s) written to {_TEMPLATES_DIR}/")
        print("Note: manually review generated templates and merge with hand-crafted ones as needed.")


if __name__ == "__main__":
    main()
