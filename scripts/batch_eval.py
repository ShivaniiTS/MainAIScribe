#!/usr/bin/env python3
"""
Batch evaluation: run pipeline on all dictation + conversation samples.

Data layout (ai-scribe-data/):
  conversation/<physician>/<encounter>/
    conversation_audio.mp3  → RecordingMode.AMBIENT
    note_audio.mp3          → optional physician dictation
    final_soap_note.md      → gold standard
    patient_demographics.json + encounter_details.json

  dictation/<physician>/<encounter>/
    dictation.mp3           → RecordingMode.DICTATION
    final_soap_note.md      → gold standard
    patient_demographics.json + encounter_details.json

Saves per-sample artifacts:
    output/<mode>/<physician>/<encounter>/generated_note_v{N}.md
    output/<mode>/<physician>/<encounter>/comparison_v{N}.md
    output/<mode>/<physician>/<encounter>/audio_transcript_v{N}.txt

Saves aggregate report:
    output/batch_report_v{N}.md

Usage:
    python scripts/batch_eval.py --version v7
    python scripts/batch_eval.py --version v7 --two-pass
    python scripts/batch_eval.py --version v7 --data-dir ai-scribe-data/dictation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import DATA_DIR as _DATA_ROOT, OUTPUT_DIR as _OUTPUT_ROOT
_MODES = ("conversation", "dictation")

# Audio filename per mode
_AUDIO_NAMES = {
    "conversation": ["conversation_audio.mp3", "note_audio.mp3"],
    "dictation": ["dictation.mp3"],
}

_GOLD_NAME = "final_soap_note.md"


def _discover_ollama_model() -> str:
    import httpx
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = resp.json().get("models", [])
        if models:
            return models[0]["name"]
    except Exception:
        pass
    return "qwen2.5:14b"


def _resolve_provider_id(encounter_dir: Path) -> str:
    """Derive provider_id from the physician folder name (parent of encounter)."""
    return encounter_dir.parent.name


def _resolve_sample(sample_dir: Path, mode: str) -> tuple[Path | None, Path | None, str, Path | None]:
    """
    Resolve audio files, recording mode string, and gold note path.

    Returns:
        (audio_path, note_audio_path, mode_str, gold_path)
        audio_path is the primary audio; note_audio_path is the physician
        dictation audio (conversation mode only). Either may be None.
    """
    audio_path = None
    note_audio_path = None

    if mode == "conversation":
        conv = sample_dir / "conversation_audio.mp3"
        note = sample_dir / "note_audio.mp3"
        if conv.exists():
            audio_path = conv
        if note.exists():
            note_audio_path = note
        # Fallback: if only note_audio exists, use it as primary
        if audio_path is None and note_audio_path is not None:
            audio_path = note_audio_path
            note_audio_path = None
    else:
        for name in _AUDIO_NAMES.get(mode, ["dictation.mp3"]):
            candidate = sample_dir / name
            if candidate.exists():
                audio_path = candidate
                break

    mode_str = "ambient" if mode == "conversation" else "dictation"
    gold = sample_dir / _GOLD_NAME
    gold_path = gold if gold.exists() else None

    if audio_path is None:
        return None, None, mode_str, gold_path

    return audio_path, note_audio_path, mode_str, gold_path


def _collect_samples(data_root: Path) -> list[tuple[Path, Path | None, str, Path | None, str]]:
    """
    Return list of (audio_path, note_audio_path, mode_str, gold_path, physician)
    for all runnable samples.
    Walks: data_root/<mode>/<physician>/<encounter>/
    """
    samples = []
    for mode in _MODES:
        mode_dir = data_root / mode
        if not mode_dir.exists():
            continue
        for physician_dir in sorted(mode_dir.iterdir()):
            if not physician_dir.is_dir():
                continue
            physician = physician_dir.name
            for encounter_dir in sorted(physician_dir.iterdir()):
                if not encounter_dir.is_dir():
                    continue
                audio_path, note_audio_path, mode_str, gold_path = _resolve_sample(encounter_dir, mode)
                if audio_path:
                    samples.append((audio_path, note_audio_path, mode_str, gold_path, physician))
    return samples


def _output_subpath(audio_path: Path, data_root: Path) -> str:
    """
    Compute the output subdirectory path that mirrors the input hierarchy.
    e.g. ai-scribe-data/conversation/dr_faraz/enc_001/ → conversation/dr_faraz/enc_001
    """
    try:
        rel = audio_path.parent.relative_to(data_root)
        return str(rel)
    except ValueError:
        # Fallback: use mode/physician/encounter from parent dirs
        encounter = audio_path.parent.name
        physician = audio_path.parent.parent.name
        mode_dir = audio_path.parent.parent.parent.name
        return f"{mode_dir}/{physician}/{encounter}"


def _load_provider_profile(physician: str):
    """Load provider profile, mapping physician folder name to provider_id."""
    from config.provider_manager import get_provider_manager
    mgr = get_provider_manager()
    # Try the physician folder name directly as provider_id
    profile = mgr.load_or_default(physician)
    return profile


def run_sample(audio_path: Path, note_audio_path: Path | None, mode: str,
               gold_path: Path | None, graph,
               version: str, out_dir: Path, data_root: Path, physician: str) -> dict:
    from orchestrator.graph import run_encounter
    from orchestrator.state import (
        DeliveryMethod, EncounterState,
        RecordingMode,
    )
    from output.markdown_writer import write_clinical_note
    from output.comparison_writer import write_comparison

    sample_id = audio_path.parent.name
    gold_note = gold_path.read_text() if gold_path else ""
    recording_mode = RecordingMode.AMBIENT if mode == "ambient" else RecordingMode.DICTATION

    # Mirror input hierarchy: output/<mode>/<physician>/<encounter>/
    sub = _output_subpath(audio_path, data_root)
    sample_out_dir = out_dir / sub
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    provider_profile = _load_provider_profile(physician)

    state = EncounterState(
        provider_id=provider_profile.id,
        patient_id=f"patient-{sample_id}",
        provider_profile=provider_profile,
        recording_mode=recording_mode,
        delivery_method=DeliveryMethod.CLIPBOARD,
        audio_file_path=str(audio_path),
        note_audio_file_path=str(note_audio_path) if note_audio_path else None,
    )

    t0 = time.time()
    final = run_encounter(graph, state)
    elapsed = time.time() - t0

    generated_note = final.final_note.to_text() if final.final_note else ""

    write_clinical_note(
        final,
        path=sample_out_dir / f"generated_note_{version}.md",
        version=version,
        sample_id=sample_id,
    )

    # Save standalone transcript
    if final.transcript and final.transcript.full_text.strip():
        transcript_path = sample_out_dir / f"audio_transcript_{version}.txt"
        transcript_path.write_text(final.transcript.full_text.strip())

    metrics = {
        "asr_engine": final.asr_engine_used,
        "asr_conf": f"{final.metrics.asr_confidence:.2f}" if final.metrics.asr_confidence else "—",
        "note_conf": f"{final.metrics.note_confidence:.2f}" if final.metrics.note_confidence else "—",
        "pp_corrections": final.metrics.postprocessor_corrections or 0,
        "asr_ms": final.metrics.asr_duration_ms or 0,
        "llm_ms": final.metrics.note_gen_ms or 0,
    }
    overlap = write_comparison(
        path=sample_out_dir / f"comparison_{version}.md",
        sample_id=sample_id,
        generated_note=generated_note,
        gold_note=gold_note,
        transcript=final.transcript.full_text if final.transcript else "",
        metrics=metrics,
        version=version,
    )

    return {
        "id": sample_id,
        "physician": physician,
        "mode": mode,
        "audio": audio_path.name,
        "elapsed": elapsed,
        "asr_ms": final.metrics.asr_duration_ms or 0,
        "llm_ms": final.metrics.note_gen_ms or 0,
        "asr_engine": final.asr_engine_used,
        "llm_engine": final.llm_engine_used,
        "asr_conf": final.metrics.asr_confidence or 0.0,
        "note_conf": final.metrics.note_confidence or 0.0,
        "pp_corrections": final.metrics.postprocessor_corrections or 0,
        "overlap": overlap,
        "has_gold": bool(gold_note),
        "errors": final.errors,
        "segments": len(final.transcript.segments) if final.transcript else 0,
        "audio_duration_s": (final.transcript.audio_duration_ms or 0) / 1000 if final.transcript else 0,
    }


def write_batch_report(results: list[dict], model: str, version: str, total_elapsed: float, out_dir: Path) -> None:
    overlaps = [r["overlap"] for r in results if r.get("has_gold") and not r.get("errors")]
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    avg_elapsed = total_elapsed / len(results) if results else 0

    dictation_results = [r for r in results if r["mode"] == "dictation"]
    ambient_results = [r for r in results if r["mode"] == "ambient"]

    lines = [
        f"# Batch Report — Pipeline {version}",
        "",
        f"**Model:** {model}  ",
        f"**Samples:** {len(results)} ({len(dictation_results)} dictation, {len(ambient_results)} ambient)  ",
        f"**Total time:** {total_elapsed:.0f}s  ",
        f"**Avg per sample:** {avg_elapsed:.1f}s  ",
        f"**Avg keyword overlap:** {avg_overlap:.0%}  ",
        "",
        "---",
        "",
        "## Per-Sample Results",
        "",
        "| Sample | Physician | Mode | Audio | Duration | ASR ms | LLM ms | ASR conf | Note conf | PP | Overlap | Status |",
        "|--------|-----------|------|-------|----------|--------|--------|----------|-----------|----|---------|--------|",
    ]

    for r in results:
        status = "⚠ " + "; ".join(r["errors"]) if r.get("errors") else "✓"
        overlap_str = f"{r['overlap']:.0%}" if r.get("has_gold") else "—"
        lines.append(
            f"| {r['id']} "
            f"| {r.get('physician', '—')} "
            f"| {r['mode']} "
            f"| {r['audio']} "
            f"| {r['elapsed']:.1f}s "
            f"| {r['asr_ms']} "
            f"| {r['llm_ms']} "
            f"| {r['asr_conf']:.2f} "
            f"| {r['note_conf']:.2f} "
            f"| {r['pp_corrections']} "
            f"| {overlap_str} "
            f"| {status} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Averages",
        "",
        "| Metric | All | Dictation | Ambient |",
        "|--------|-----|-----------|---------|",
    ]

    def _avg(lst, key):
        vals = [r[key] for r in lst if not r.get("errors")]
        return f"{sum(vals)/len(vals):.2f}" if vals else "—"

    def _avg_overlap(lst):
        vals = [r["overlap"] for r in lst if r.get("has_gold") and not r.get("errors")]
        return f"{sum(vals)/len(vals):.0%}" if vals else "—"

    for label, key in [("Elapsed", "elapsed"), ("ASR conf", "asr_conf"), ("Note conf", "note_conf"), ("PP corrections", "pp_corrections")]:
        lines.append(f"| {label} | {_avg(results, key)} | {_avg(dictation_results, key)} | {_avg(ambient_results, key)} |")
    lines.append(f"| Keyword overlap | {_avg_overlap(results)} | {_avg_overlap(dictation_results)} | {_avg_overlap(ambient_results)} |")
    lines.append(f"| Errors | {sum(1 for r in results if r.get('errors'))}/{len(results)} | {sum(1 for r in dictation_results if r.get('errors'))}/{len(dictation_results)} | {sum(1 for r in ambient_results if r.get('errors'))}/{len(ambient_results)} |")

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- Keyword overlap measures domain-relevant word matches between generated and gold notes",
        "- Ambient mode uses diarization (pyannote); dictation mode is single-speaker",
        f"- Data source: ai-scribe-data/  (physician-organized encounters)",
        "",
        f"*Generated by AI Scribe batch_eval.py — version {version}*",
        "",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"batch_report_{version}.md"
    report_path.write_text("\n".join(lines))
    print(f"Batch report : {report_path}")


def _two_pass_main(args, samples: list, model: str) -> tuple[list, float]:
    """
    Two-pass execution for large LLMs that cannot share VRAM with WhisperX.

    Pass 1 — ASR only: transcription pipeline, saves transcript cache JSON.
    Pass 2 — LLM only: unload WhisperX, load transcripts from cache.
    """
    import torch
    from orchestrator.graph import build_graph
    from orchestrator.nodes.note_node import set_llm_engine_factory
    from orchestrator.nodes.transcribe_node import set_asr_engine_factory
    from mcp_servers.llm.base import LLMEngine, LLMConfig, LLMMessage, LLMResponse, ModelInfo
    from mcp_servers.llm.ollama_server import OllamaServer
    from mcp_servers.asr.whisperx_server import WhisperXServer
    from orchestrator.state import UnifiedTranscript, TranscriptSegment
    from config.loader import get_asr_config

    data_root = Path(args.data_dir) if args.data_dir else _DATA_ROOT
    out_dir = Path(args.output_dir)

    class _NoOpLLM(LLMEngine):
        """Zero-VRAM stub for Pass 1."""
        def generate_sync(self, system_prompt, messages, config, task="note_generation") -> LLMResponse:
            return LLMResponse(content="", model="noop", prompt_tokens=0, completion_tokens=0)
        async def generate(self, system_prompt, messages, config) -> LLMResponse:
            return LLMResponse(content="", model="noop", prompt_tokens=0, completion_tokens=0)
        async def generate_stream(self, system_prompt, messages, config):
            return; yield
        async def get_model_info(self) -> ModelInfo:
            return ModelInfo(model_name="noop", context_window=128000)

    # ── Pass 1: ASR ──────────────────────────────────────────────────────────
    print("=== Pass 1: ASR (WhisperX) ===")
    asr_cfg = get_asr_config()
    asr_engine = WhisperXServer.from_config(asr_cfg)
    set_asr_engine_factory(lambda: asr_engine)
    set_llm_engine_factory(lambda: _NoOpLLM())
    asr_graph = build_graph()

    transcript_cache: dict[str, dict] = {}

    for i, (audio_path, note_audio_path, mode, gold_path, physician) in enumerate(samples, 1):
        sample_id = audio_path.parent.name
        dual_tag = " +note_audio" if note_audio_path else ""
        print(f"  [{i}/{len(samples)}] {sample_id} ASR{dual_tag} ... ", end="", flush=True)
        from orchestrator.state import DeliveryMethod, EncounterState, RecordingMode
        provider_profile = _load_provider_profile(physician)
        recording_mode = RecordingMode.AMBIENT if mode == "ambient" else RecordingMode.DICTATION
        state = EncounterState(
            provider_id=provider_profile.id,
            patient_id=f"patient-{sample_id}",
            provider_profile=provider_profile,
            recording_mode=recording_mode,
            delivery_method=DeliveryMethod.CLIPBOARD,
            audio_file_path=str(audio_path),
            note_audio_file_path=str(note_audio_path) if note_audio_path else None,
        )
        from orchestrator.graph import run_encounter
        try:
            final = run_encounter(asr_graph, state)
            if final.transcript:
                transcript_cache[sample_id] = {
                    "full_text": final.transcript.full_text,
                    "segments": [
                        {"text": s.text, "start_ms": s.start_ms, "end_ms": s.end_ms,
                         "speaker": s.speaker, "confidence": s.confidence}
                        for s in (final.transcript.segments or [])
                    ],
                    "audio_duration_ms": final.transcript.audio_duration_ms,
                    "asr_conf": final.metrics.asr_confidence or 0.0,
                    "pp_corrections": final.metrics.postprocessor_corrections or 0,
                    "errors": final.errors,
                }
                print(f"conf={final.metrics.asr_confidence:.2f} pp={final.metrics.postprocessor_corrections}")
            else:
                transcript_cache[sample_id] = {"full_text": "", "segments": [], "errors": final.errors}
                print(f"FAILED: {final.errors}")
        except Exception as exc:
            transcript_cache[sample_id] = {"full_text": "", "segments": [], "errors": [str(exc)]}
            print(f"FAILED: {exc}")

    cache_path = out_dir / f"transcript_cache_{args.version}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(transcript_cache, indent=2))
    print(f"  Transcripts saved: {cache_path}")

    # ── Free WhisperX VRAM ───────────────────────────────────────────────────
    print("\nFreeing WhisperX VRAM ...")
    del asr_engine
    torch.cuda.empty_cache()
    import time as _time; _time.sleep(3)

    # ── Pass 2: LLM ──────────────────────────────────────────────────────────
    print(f"\n=== Pass 2: LLM ({model}) ===")

    set_llm_engine_factory(lambda: OllamaServer(model_overrides={"note_generation": model}, keep_alive=0))
    llm_graph = build_graph()

    results = []
    total_t0 = _time.time()
    from output.markdown_writer import write_clinical_note
    from output.comparison_writer import write_comparison
    from orchestrator.state import (
        DeliveryMethod, EncounterState, RecordingMode,
        UnifiedTranscript, TranscriptSegment,
    )
    from orchestrator.graph import run_encounter

    for i, (audio_path, note_audio_path, mode, gold_path, physician) in enumerate(samples, 1):
        sample_id = audio_path.parent.name
        gold_str = gold_path.name if gold_path else "no gold"
        print(f"  [{i}/{len(samples)}] {sample_id} LLM ... ", end="", flush=True)
        cached = transcript_cache.get(sample_id, {})

        sub = _output_subpath(audio_path, data_root)
        sample_out_dir = out_dir / sub
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        gold_note = gold_path.read_text() if gold_path else ""

        try:
            provider_profile = _load_provider_profile(physician)
            recording_mode = RecordingMode.AMBIENT if mode == "ambient" else RecordingMode.DICTATION

            segments = [
                TranscriptSegment(
                    text=s["text"], start_ms=s.get("start_ms", 0), end_ms=s.get("end_ms", 0),
                    speaker=s.get("speaker"), confidence=s.get("confidence") or 1.0,
                )
                for s in cached.get("segments", [])
            ]
            transcript = UnifiedTranscript(
                full_text=cached.get("full_text", ""),
                segments=segments,
            )

            state = EncounterState(
                provider_id=provider_profile.id,
                patient_id=f"patient-{sample_id}",
                provider_profile=provider_profile,
                recording_mode=recording_mode,
                delivery_method=DeliveryMethod.CLIPBOARD,
                audio_file_path=str(audio_path),
                transcript=transcript,
            )

            t0 = _time.time()
            final = run_encounter(llm_graph, state)
            elapsed = _time.time() - t0

            generated_note = final.final_note.to_text() if final.final_note else ""
            write_clinical_note(final, path=sample_out_dir / f"generated_note_{args.version}.md",
                                version=args.version, sample_id=sample_id)

            cached_text = cached.get("full_text", "").strip()
            if cached_text:
                (sample_out_dir / f"audio_transcript_{args.version}.txt").write_text(cached_text)

            metrics = {
                "asr_engine": "preloaded",
                "asr_conf": f"{cached.get('asr_conf', 0):.2f}",
                "note_conf": f"{final.metrics.note_confidence:.2f}" if final.metrics.note_confidence else "—",
                "pp_corrections": cached.get("pp_corrections", 0),
                "asr_ms": 0,
                "llm_ms": final.metrics.note_gen_ms or 0,
            }
            overlap = write_comparison(
                path=sample_out_dir / f"comparison_{args.version}.md",
                sample_id=sample_id, generated_note=generated_note,
                gold_note=gold_note,
                transcript=final.transcript.full_text if final.transcript else "",
                metrics=metrics, version=args.version,
            )
            overlap_str = f"{overlap:.0%}" if gold_note else "—"
            print(f"{elapsed:.1f}s | note={final.metrics.note_confidence:.2f} overlap={overlap_str}")
            results.append({
                "id": sample_id, "physician": physician, "mode": mode, "audio": audio_path.name,
                "elapsed": elapsed, "asr_ms": 0, "llm_ms": final.metrics.note_gen_ms or 0,
                "asr_engine": "preloaded", "llm_engine": final.llm_engine_used,
                "asr_conf": cached.get("asr_conf", 0.0),
                "note_conf": final.metrics.note_confidence or 0.0,
                "pp_corrections": cached.get("pp_corrections", 0),
                "overlap": overlap, "has_gold": bool(gold_note),
                "errors": final.errors,
                "segments": len(segments),
                "audio_duration_s": (cached.get("audio_duration_ms") or 0) / 1000,
            })
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"FAILED: {exc}")
            results.append({
                "id": sample_id, "physician": physician, "mode": mode, "audio": audio_path.name,
                "elapsed": 0, "asr_ms": 0, "llm_ms": 0,
                "asr_conf": 0, "note_conf": 0, "pp_corrections": 0,
                "overlap": 0, "has_gold": bool(gold_note), "errors": [str(exc)],
                "segments": 0, "audio_duration_s": 0,
                "asr_engine": "—", "llm_engine": "—",
            })

    return results, _time.time() - total_t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v7", help="Pipeline version label")
    parser.add_argument("--data-dir", default=None,
                        help="Data root (default: ai-scribe-data). Can point to a subdirectory.")
    parser.add_argument("--output-dir", default=str(_OUTPUT_ROOT))
    parser.add_argument("--model", default=None,
                        help="Ollama model name (default: auto-discover)")
    parser.add_argument("--two-pass", action="store_true",
                        help="Two-pass mode: ASR first, then LLM. Required for large models on shared GPU.")
    parser.add_argument("--mode", default=None, choices=["ambient", "dictation"],
                        help="Filter to only run samples of this mode")
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else _DATA_ROOT
    samples = _collect_samples(data_root)

    if args.mode:
        samples = [s for s in samples if s[2] == args.mode]

    if not samples:
        print(f"No audio samples found in: {data_root}")
        sys.exit(1)

    model = args.model or _discover_ollama_model()
    dictation_count = sum(1 for _, _, m, _, _ in samples if m == "dictation")
    ambient_count = sum(1 for _, _, m, _, _ in samples if m == "ambient")
    dual_audio_count = sum(1 for _, na, m, _, _ in samples if na is not None)
    physicians = sorted(set(p for _, _, _, _, p in samples))
    print(f"Ollama model : {model}")
    print(f"Version      : {args.version}")
    print(f"Data root    : {data_root}")
    print(f"Samples      : {len(samples)} ({dictation_count} dictation, {ambient_count} ambient, {dual_audio_count} dual-audio)")
    print(f"Physicians   : {', '.join(physicians)}")
    print(f"Two-pass     : {'yes' if args.two_pass else 'no'}")
    print()

    if args.two_pass:
        results, total_elapsed = _two_pass_main(args, samples, model)
        write_batch_report(results, model, args.version, total_elapsed, Path(args.output_dir))
        overlaps = [r["overlap"] for r in results if r.get("has_gold") and not r.get("errors")]
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        print(f"\n{'─'*70}")
        print(f"Done  : {len(results)} samples in {total_elapsed:.0f}s")
        print(f"Avg overlap : {avg_overlap:.0%}")
        print(f"{'─'*70}")
        return

    from orchestrator.graph import build_graph
    from orchestrator.nodes.note_node import set_llm_engine_factory
    from orchestrator.nodes.transcribe_node import set_asr_engine_factory
    from mcp_servers.llm.ollama_server import OllamaServer
    from mcp_servers.asr.whisperx_server import WhisperXServer
    from config.loader import get_asr_config

    asr_cfg = get_asr_config()
    asr_engine = WhisperXServer.from_config(asr_cfg)
    set_asr_engine_factory(lambda: asr_engine)
    set_llm_engine_factory(lambda: OllamaServer(model_overrides={"note_generation": model}, keep_alive=0))
    graph = build_graph()

    results = []
    total_t0 = time.time()

    for i, (audio_path, note_audio_path, mode, gold_path, physician) in enumerate(samples, 1):
        sample_id = audio_path.parent.name
        gold_str = gold_path.name if gold_path else "no gold"
        dual_tag = " +note_audio" if note_audio_path else ""
        print(f"[{i}/{len(samples)}] {sample_id} ({mode}, {audio_path.name}{dual_tag}, gold={gold_str}) ... ", end="", flush=True)
        try:
            r = run_sample(audio_path, note_audio_path, mode, gold_path, graph, args.version, Path(args.output_dir), data_root, physician)
            overlap_str = f"{r['overlap']:.0%}" if r["has_gold"] else "—"
            print(
                f"{r['elapsed']:.1f}s | "
                f"ASR={r['asr_conf']:.2f} note={r['note_conf']:.2f} "
                f"overlap={overlap_str} pp={r['pp_corrections']}"
                + (f" ERRORS: {r['errors']}" if r["errors"] else "")
            )
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"FAILED: {exc}")
            traceback.print_exc()
            results.append({
                "id": sample_id, "physician": physician, "mode": mode, "audio": audio_path.name,
                "elapsed": 0, "asr_ms": 0, "llm_ms": 0,
                "asr_conf": 0, "note_conf": 0, "pp_corrections": 0,
                "overlap": 0, "has_gold": gold_path is not None, "errors": [str(exc)],
                "segments": 0, "audio_duration_s": 0,
                "asr_engine": "—", "llm_engine": "—",
            })

    total_elapsed = time.time() - total_t0
    overlaps = [r["overlap"] for r in results if r.get("has_gold") and not r.get("errors")]
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    print(f"\n{'─'*70}")
    print(f"Done  : {len(results)} samples in {total_elapsed:.0f}s")
    print(f"Avg overlap : {avg_overlap:.0%}")
    print(f"{'─'*70}")

    write_batch_report(results, model, args.version, total_elapsed, Path(args.output_dir))


if __name__ == "__main__":
    main()
