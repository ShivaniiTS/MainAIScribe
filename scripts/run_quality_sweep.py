#!/usr/bin/env python3
"""
Run quality evaluation sweep across all pipeline output samples.

For each sample that has a generated note + gold standard:
  1. Score with LLM-as-judge (6 dimensions)
  2. Extract and compare facts (medications, diagnoses, findings, plan)
  3. Write aggregate quality_report_v{N}.md to output/

Data layout:
  ai-scribe-data/<mode>/<physician>/<encounter>/final_soap_note.md  (gold)
  output/<mode>/<physician>/<encounter>/generated_note_v{N}.md      (generated)

Usage:
    python scripts/run_quality_sweep.py --version v7
    python scripts/run_quality_sweep.py --version v7 --judge-model llama3.1:latest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import DATA_DIR as _DATA_ROOT, OUTPUT_DIR as _OUTPUT_DIR
_MODES = ("conversation", "dictation")
_GOLD_NAME = "final_soap_note.md"


def _collect_samples(version: str) -> list[tuple[str, str, str, Path, Path]]:
    """
    Return (mode, physician, sample_id, generated_note_path, gold_path)
    for all samples that have both a generated note and a gold standard.
    """
    samples = []
    for mode in _MODES:
        out_mode_dir = _OUTPUT_DIR / mode
        data_mode_dir = _DATA_ROOT / mode
        if not out_mode_dir.exists():
            continue
        for physician_dir in sorted(out_mode_dir.iterdir()):
            if not physician_dir.is_dir():
                continue
            physician = physician_dir.name
            for encounter_dir in sorted(physician_dir.iterdir()):
                if not encounter_dir.is_dir():
                    continue
                sample_id = encounter_dir.name
                note_path = encounter_dir / f"generated_note_{version}.md"
                gold_path = data_mode_dir / physician / sample_id / _GOLD_NAME
                if note_path.exists() and gold_path.exists():
                    samples.append((mode, physician, sample_id, note_path, gold_path))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v7", help="Pipeline version to evaluate")
    parser.add_argument("--no-fact-check", action="store_true", help="Skip fact extraction (faster)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--judge-model", default=None,
                        help="Ollama model to use as LLM judge (default: auto-discover)")
    args = parser.parse_args()

    samples = _collect_samples(args.version)
    if not samples:
        print(f"No samples found for version {args.version}. Run batch_eval.py first.")
        sys.exit(1)

    print(f"Evaluating {len(samples)} samples (version={args.version})")

    if args.judge_model:
        model = args.judge_model
    else:
        import httpx
        try:
            resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
            models = resp.json().get("models", [])
            model = models[0]["name"] if models else "qwen2.5:14b"
        except Exception:
            model = "qwen2.5:14b"

    from mcp_servers.llm.ollama_server import OllamaServer
    engine = OllamaServer(model_overrides={"command_parse": model, "note_generation": model})
    print(f"Judge model    : {model}")
    print()

    from quality.evaluator import QualityEvaluator
    from quality.report import write_quality_report, write_aggregate_report

    evaluator = QualityEvaluator(engine, run_fact_check=not args.no_fact_check)
    results = []
    out_dir = Path(args.output_dir)

    for i, (mode, physician, sample_id, note_path, gold_path) in enumerate(samples, 1):
        display_mode = "ambient" if mode == "conversation" else "dictation"
        print(f"[{i}/{len(samples)}] {sample_id} ({display_mode}, {physician}) ... ", end="", flush=True)
        generated = note_path.read_text()
        gold = gold_path.read_text()

        # Read transcript if available
        transcript = ""
        comparison_path = note_path.parent / f"comparison_{args.version}.md"
        if comparison_path.exists():
            comp_text = comparison_path.read_text()
            if "<summary>Transcript" in comp_text:
                parts = comp_text.split("</summary>")
                if len(parts) > 1:
                    transcript = parts[1].split("</details>")[0].strip()

        try:
            result = evaluator.evaluate(
                sample_id=sample_id,
                generated_note=generated,
                gold_note=gold,
                transcript=transcript,
                version=args.version,
            )
            results.append(result)

            fc_str = ""
            if result.fact_check:
                fc = result.fact_check
                fc_str = f" | dx={fc.diagnoses[0]}/{fc.diagnoses[1]} meds={fc.medications[0]}/{fc.medications[1]}"
            print(
                f"{result.elapsed_s:.1f}s | "
                f"score={result.overall_score:.2f} "
                f"overlap={result.keyword_overlap:.0%}"
                f"{fc_str}"
            )
        except Exception as exc:
            import traceback
            print(f"FAILED: {exc}")
            traceback.print_exc()

    if results:
        scored = [r for r in results if r.has_gold]
        avg = sum(r.overall_score for r in scored) / len(scored) if scored else 0
        avg_overlap = sum(r.keyword_overlap for r in scored) / len(scored) if scored else 0
        print(f"\n{'─'*60}")
        print(f"Average score   : {avg:.2f} / 5.0")
        print(f"Average overlap : {avg_overlap:.0%}")
        print(f"{'─'*60}")

        agg_path = out_dir / f"quality_report_{args.version}.md"
        write_aggregate_report(results, args.version, agg_path)

        # ── Per-provider quality tracking ────────────────────────────────
        from config.provider_manager import get_provider_manager
        mgr = get_provider_manager()
        for provider_id in mgr.list_providers():
            dim_keys = ["medical_accuracy", "completeness", "no_hallucination",
                        "structure_compliance", "clinical_language", "readability"]
            dim_totals: dict[str, list[float]] = {k: [] for k in dim_keys}
            for r in scored:
                for k in dim_keys:
                    v = getattr(r, k, None)
                    if v is not None:
                        dim_totals[k].append(v)
            dim_avgs = {k: round(sum(v) / len(v), 3) for k, v in dim_totals.items() if v}
            mgr.update_quality_score(
                provider_id=provider_id,
                version=args.version,
                score=avg,
                sample_count=len(scored),
                dimension_scores=dim_avgs or None,
            )
            print(f"Quality scores updated for provider: {provider_id}")

    print(f"\nPer-sample files: output/{{conversation,dictation}}/<physician>/<encounter>/")


if __name__ == "__main__":
    main()
