"""
api/quality_runner.py — Run quality evaluation after pipeline completion.

Provides helpers for:
1. Per-sample quality evaluation (on-demand pipeline)
2. Aggregate quality report generation (batch pipeline)

Quality evaluation requires:
- A gold standard note (final_soap_note.md) for the sample
- An LLM engine (Ollama) for the judge model
- The generated note and transcript from pipeline output

If no gold standard exists (e.g., live encounters), quality evaluation is skipped.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def evaluate_sample(
    sample_id: str,
    generated_note: str,
    gold_note: Optional[str],
    transcript: str = "",
    version: str = "v1",
    output_dir: Optional[Path] = None,
    judge_model: str = "llama3.1:latest",
) -> Optional[dict]:
    """
    Run quality evaluation for a single sample.

    Returns the QualityResult dict if evaluation succeeded, None if skipped.
    Writes per-sample quality_report_{version}.md to output_dir if provided.
    """
    if not gold_note or not gold_note.strip():
        logger.info("quality_skip_no_gold: %s", sample_id)
        return None

    if not generated_note or not generated_note.strip():
        logger.warning("quality_skip_no_generated: %s", sample_id)
        return None

    try:
        from mcp_servers.llm.ollama_server import OllamaServer
        from quality.evaluator import QualityEvaluator
        from quality.report import write_quality_report

        engine = OllamaServer(model_overrides={
            "command_parse": judge_model,
            "note_generation": judge_model,
        })

        evaluator = QualityEvaluator(engine, run_fact_check=True)
        result = evaluator.evaluate(
            sample_id=sample_id,
            generated_note=generated_note,
            gold_note=gold_note,
            transcript=transcript,
            version=version,
        )

        # Write per-sample quality report
        if output_dir:
            report_path = output_dir / f"quality_report_{version}.md"
            write_quality_report(result, report_path)
            logger.info(
                "quality_report_written: sample=%s version=%s score=%s path=%s",
                sample_id, version, result.overall_score, str(report_path),
            )

        return result.to_dict()

    except Exception as e:
        logger.error("quality_evaluation_failed: sample=%s error=%s", sample_id, str(e))
        return None


def generate_aggregate_report(
    version: str,
    judge_model: str = "llama3.1:latest",
) -> Optional[str]:
    """
    Run quality evaluation for ALL samples that have a generated note + gold standard
    for the given version, and generate the aggregate quality_report_{version}.md.

    Returns the path to the aggregate report, or None if no samples to evaluate.
    """
    from config.paths import DATA_DIR, OUTPUT_DIR
    from quality.report import write_aggregate_report

    _MODES = ("conversation", "dictation")
    _GOLD_NAME = "final_soap_note.md"

    # Collect samples
    samples = []
    for mode in _MODES:
        out_mode_dir = OUTPUT_DIR / mode
        data_mode_dir = DATA_DIR / mode
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
                    samples.append((sample_id, note_path, gold_path, encounter_dir))

    if not samples:
        logger.info("quality_sweep_no_samples: version=%s", version)
        return None

    logger.info("quality_sweep_start: version=%s count=%d", version, len(samples))

    try:
        from mcp_servers.llm.ollama_server import OllamaServer
        from quality.evaluator import QualityEvaluator
        from quality.report import write_quality_report

        engine = OllamaServer(model_overrides={
            "command_parse": judge_model,
            "note_generation": judge_model,
        })

        evaluator = QualityEvaluator(engine, run_fact_check=True)
        results = []

        for i, (sample_id, note_path, gold_path, out_dir) in enumerate(samples, 1):
            logger.info("quality_sweep_sample: %d/%d sample=%s", i, len(samples), sample_id)
            generated = note_path.read_text()
            gold = gold_path.read_text()

            # Read transcript if available
            transcript = ""
            tx_path = out_dir / f"audio_transcript_{version}.txt"
            if tx_path.exists():
                transcript = tx_path.read_text()

            try:
                result = evaluator.evaluate(
                    sample_id=sample_id,
                    generated_note=generated,
                    gold_note=gold,
                    transcript=transcript,
                    version=version,
                )
                results.append(result)

                # Write per-sample quality report
                report_path = out_dir / f"quality_report_{version}.md"
                write_quality_report(result, report_path)

            except Exception as e:
                logger.error("quality_sweep_sample_failed: sample=%s error=%s", sample_id, str(e))

        if results:
            agg_path = OUTPUT_DIR / f"quality_report_{version}.md"
            write_aggregate_report(results, version, agg_path)

            # Invalidate cached quality data so API returns fresh results
            from api.data_loader import clear_quality_cache
            clear_quality_cache(version)

            logger.info(
                "quality_sweep_complete: version=%s count=%d path=%s",
                version, len(results), str(agg_path),
            )
            return str(agg_path)

    except Exception as e:
        logger.error("quality_sweep_failed: version=%s error=%s", version, str(e))

    return None
