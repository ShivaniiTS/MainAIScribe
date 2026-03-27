"""
scripts/eval_asr_quality.py

Compare ASR quality: base Whisper large-v3 vs provider LoRA-fine-tuned model.

Metrics computed per sample:
  - Word Error Rate (WER)     — industry-standard ASR quality metric
  - Character Error Rate (CER) — better for medical abbreviations
  - Medical Term Accuracy (MTA) — precision/recall on specialty vocabulary terms

Output:
    output/asr_eval_{provider_id}.md  — full Markdown report
    output/asr_eval_{provider_id}.json — machine-readable scores

Usage:
    # Compare base vs LoRA for a provider
    python scripts/eval_asr_quality.py --provider dr_faraz_rahman

    # Evaluate only base model (no LoRA comparison)
    python scripts/eval_asr_quality.py --provider dr_faraz_rahman --base-only

    # Use custom reference transcripts
    python scripts/eval_asr_quality.py --provider dr_faraz_rahman --ref-dir data/transcripts/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.paths import DATA_DIR, OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    sample_id: str
    mode: str
    audio_path: str
    ref_text: str
    base_hyp: str = ""
    lora_hyp: str = ""
    base_wer: Optional[float] = None
    base_cer: Optional[float] = None
    lora_wer: Optional[float] = None
    lora_cer: Optional[float] = None
    base_mta: Optional[float] = None   # Medical Term Accuracy
    lora_mta: Optional[float] = None
    error: Optional[str] = None


@dataclass
class EvalReport:
    provider_id: str
    samples: list[SampleResult] = field(default_factory=list)
    has_lora: bool = False

    @property
    def base_avg_wer(self) -> Optional[float]:
        vals = [s.base_wer for s in self.samples if s.base_wer is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    @property
    def lora_avg_wer(self) -> Optional[float]:
        vals = [s.lora_wer for s in self.samples if s.lora_wer is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    @property
    def base_avg_cer(self) -> Optional[float]:
        vals = [s.base_cer for s in self.samples if s.base_cer is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    @property
    def lora_avg_cer(self) -> Optional[float]:
        vals = [s.lora_cer for s in self.samples if s.lora_cer is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    @property
    def base_avg_mta(self) -> Optional[float]:
        vals = [s.base_mta for s in self.samples if s.base_mta is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    @property
    def lora_avg_mta(self) -> Optional[float]:
        vals = [s.lora_mta for s in self.samples if s.lora_mta is not None]
        return round(sum(vals) / len(vals), 4) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: (substitutions + deletions + insertions) / len(reference words)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 0.0

    # Dynamic programming edit distance on words
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp

    return round(dp[m] / n, 4)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit distance on characters / len(reference chars)."""
    ref = reference.lower().replace(" ", "")
    hyp = hypothesis.lower().replace(" ", "")
    if not ref:
        return 0.0

    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp

    return round(dp[m] / n, 4)


def compute_medical_term_accuracy(
    hypothesis: str,
    medical_terms: list[str],
) -> float:
    """
    Fraction of medical_terms that appear correctly in the hypothesis.
    Case-insensitive exact match (after normalisation).
    Returns precision: correctly recognised / total medical terms in reference.
    """
    if not medical_terms:
        return 1.0

    hyp_lower = hypothesis.lower()
    found = sum(1 for t in medical_terms if t.lower() in hyp_lower)
    return round(found / len(medical_terms), 4)


def load_medical_terms(provider_id: str) -> list[str]:
    """Load provider custom_vocabulary + specialty hotwords."""
    try:
        from config.provider_manager import get_provider_manager
        pm = get_provider_manager()
        profile = pm.load(provider_id)
        terms = list(profile.custom_vocabulary or [])

        # Add specialty terms
        specialty = profile.specialty or "general"
        if specialty != "general":
            try:
                from mcp_servers.data.medical_dict_server import get_dict_server
                terms.extend(get_dict_server().get_hotwords(specialty, max_terms=100))
            except Exception:
                pass
        return terms
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Transcription runners
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_with_base(audio_path: str) -> str:
    """Run base WhisperX on audio; return transcript text."""
    import whisperx, torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language="en")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return " ".join(seg["text"].strip() for seg in result.get("segments", []))


def transcribe_with_lora(audio_path: str, provider_id: str) -> str:
    """Run LoRA-fine-tuned WhisperX on audio; return transcript text."""
    from mcp_servers.asr.whisperx_lora_server import WhisperXLoRAServer, adapter_exists
    import torch

    if not adapter_exists(provider_id):
        raise FileNotFoundError(
            f"No LoRA adapter for provider '{provider_id}'. "
            f"Run scripts/finetune_whisper_lora.py first."
        )

    server = WhisperXLoRAServer.for_provider(
        provider_id=provider_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    import whisperx
    audio = whisperx.load_audio(audio_path)
    server._load_model()
    result = server._model.transcribe(audio, batch_size=16, language="en")

    return " ".join(seg["text"].strip() for seg in result.get("segments", []))


# ─────────────────────────────────────────────────────────────────────────────
# Reference text extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_reference_text(sample_dir: Path) -> str:
    """Extract plain text from the gold note to use as reference transcript."""
    from scripts.prepare_asr_training_data import extract_plain_text

    # New format
    gold = sample_dir / "final_soap_note.md"
    if gold.exists():
        return extract_plain_text(gold)
    # Legacy fallback
    for fname in ("soap_final.md", "soap_initial.md"):
        gold = sample_dir / fname
        if gold.exists():
            return extract_plain_text(gold)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Sample discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_eval_samples() -> list[dict]:
    """Find all audio samples with a gold note for evaluation.

    Walks: ai-scribe-data/<mode>/<physician>/<encounter>/
    """
    samples = []
    _audio_names = {
        "dictation": ["dictation.mp3"],
        "conversation": ["conversation_audio.mp3", "note_audio.mp3"],
    }
    for mode in ("dictation", "conversation"):
        mode_dir = DATA_DIR / mode
        if not mode_dir.exists():
            continue
        display_mode = "dictation" if mode == "dictation" else "ambient"
        for physician_dir in sorted(mode_dir.iterdir()):
            if not physician_dir.is_dir():
                continue
            for sample_dir in sorted(physician_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                # Find audio file
                audio = None
                for name in _audio_names[mode]:
                    candidate = sample_dir / name
                    if candidate.exists():
                        audio = candidate
                        break
                if audio is None:
                    continue
                ref = get_reference_text(sample_dir)
                if len(ref) < 50:
                    continue
                samples.append({
                    "sample_id": sample_dir.name,
                    "mode": display_mode,
                    "audio_path": str(audio),
                    "sample_dir": sample_dir,
                    "ref_text": ref,
                })
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown_report(report: EvalReport) -> str:
    lines = [
        f"# ASR Quality Evaluation — {report.provider_id}",
        "",
        f"**Samples evaluated:** {len(report.samples)}  ",
        f"**LoRA adapter available:** {'Yes' if report.has_lora else 'No — base model only'}  ",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Base Whisper | LoRA Fine-tuned | Delta |",
        "|--------|-------------|-----------------|-------|",
    ]

    def delta(base, lora):
        if base is None or lora is None:
            return "—"
        d = lora - base
        sign = "+" if d > 0 else ""
        better = "↑" if d > 0 else "↓"  # for WER lower is better
        return f"{sign}{d:.4f} {better}"

    def fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    base_wer = report.base_avg_wer
    lora_wer = report.lora_avg_wer
    base_cer = report.base_avg_cer
    lora_cer = report.lora_avg_cer
    base_mta = report.base_avg_mta
    lora_mta = report.lora_avg_mta

    # For WER/CER: lower is better (↓ = improvement)
    def delta_wer(base, lora):
        if base is None or lora is None:
            return "—"
        d = lora - base
        sign = "+" if d > 0 else ""
        tag = "✗ worse" if d > 0 else "✓ better" if d < 0 else "="
        return f"{sign}{d:.4f} ({tag})"

    # For MTA: higher is better
    def delta_mta(base, lora):
        if base is None or lora is None:
            return "—"
        d = lora - base
        sign = "+" if d > 0 else ""
        tag = "✓ better" if d > 0 else "✗ worse" if d < 0 else "="
        return f"{sign}{d:.4f} ({tag})"

    lines += [
        f"| **WER** (lower=better) | {fmt(base_wer)} | {fmt(lora_wer)} | {delta_wer(base_wer, lora_wer)} |",
        f"| **CER** (lower=better) | {fmt(base_cer)} | {fmt(lora_cer)} | {delta_wer(base_cer, lora_cer)} |",
        f"| **Med. Term Acc** (higher=better) | {fmt(base_mta)} | {fmt(lora_mta)} | {delta_mta(base_mta, lora_mta)} |",
    ]

    # ── Per-mode breakdown ────────────────────────────────────────────────────
    # Dictation and ambient are fundamentally different tasks; mixing them masks
    # the true impact of dictation-trained LoRA on dictation samples.
    for mode_label, mode_key in [("Dictation samples", "dictation"), ("Ambient samples", "ambient")]:
        mode_samples = [s for s in report.samples if s.mode == mode_key]
        if not mode_samples:
            continue
        m_base_wer = sum(s.base_wer for s in mode_samples if s.base_wer is not None) / len(mode_samples)
        m_lora_wer = (
            sum(s.lora_wer for s in mode_samples if s.lora_wer is not None) / len(mode_samples)
            if any(s.lora_wer is not None for s in mode_samples) else None
        )
        lines += [
            "",
            f"### {mode_label} (n={len(mode_samples)})",
            "",
            "| Metric | Base | LoRA | Delta |",
            "|--------|------|------|-------|",
            f"| WER | {fmt(m_base_wer)} | {fmt(m_lora_wer)} | {delta_wer(m_base_wer, m_lora_wer)} |",
        ]

    lines += [
        "",
        "---",
        "",
        "## Per-Sample Results",
        "",
        "| Sample | Mode | Base WER | LoRA WER | Base CER | LoRA CER | Base MTA | LoRA MTA |",
        "|--------|------|----------|----------|----------|----------|----------|----------|",
    ]

    for s in report.samples:
        row = (
            f"| {s.sample_id[:30]} | {s.mode} "
            f"| {fmt(s.base_wer)} | {fmt(s.lora_wer)} "
            f"| {fmt(s.base_cer)} | {fmt(s.lora_cer)} "
            f"| {fmt(s.base_mta)} | {fmt(s.lora_mta)} |"
        )
        lines.append(row)

    # Worst WER examples (show first 3)
    worst = sorted(
        [s for s in report.samples if s.base_wer is not None],
        key=lambda s: s.base_wer or 0,
        reverse=True,
    )[:3]

    if worst:
        lines += [
            "",
            "---",
            "",
            "## Worst-Error Samples (Base Model)",
            "",
        ]
        for s in worst:
            lines += [
                f"### {s.sample_id}",
                "",
                f"**WER:** {fmt(s.base_wer)} | **CER:** {fmt(s.base_cer)}",
                "",
                "**Reference (first 300 chars):**",
                f"> {s.ref_text[:300]}",
                "",
                "**Base hypothesis (first 300 chars):**",
                f"> {s.base_hyp[:300]}",
                "",
            ]
            if s.lora_hyp:
                lines += [
                    "**LoRA hypothesis (first 300 chars):**",
                    f"> {s.lora_hyp[:300]}",
                    "",
                ]

    lines += [
        "",
        "---",
        "",
        "*Generated by eval_asr_quality.py*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    provider_id: str,
    base_only: bool = False,
    max_samples: Optional[int] = None,
) -> EvalReport:
    from mcp_servers.asr.whisperx_lora_server import adapter_exists

    has_lora = adapter_exists(provider_id) and not base_only
    medical_terms = load_medical_terms(provider_id)

    log.info(
        "Evaluating ASR for provider '%s' — LoRA: %s, medical terms: %d",
        provider_id, has_lora, len(medical_terms),
    )

    samples = discover_eval_samples()
    if max_samples:
        samples = samples[:max_samples]

    log.info("Found %d samples for evaluation", len(samples))

    report = EvalReport(provider_id=provider_id, has_lora=has_lora)

    for i, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        log.info("[%d/%d] Evaluating sample %s", i + 1, len(samples), sample_id)

        result = SampleResult(
            sample_id=sample_id,
            mode=sample["mode"],
            audio_path=sample["audio_path"],
            ref_text=sample["ref_text"],
        )

        # Base model transcription
        try:
            result.base_hyp = transcribe_with_base(sample["audio_path"])
            result.base_wer = compute_wer(result.ref_text, result.base_hyp)
            result.base_cer = compute_cer(result.ref_text, result.base_hyp)
            result.base_mta = compute_medical_term_accuracy(result.base_hyp, medical_terms)
            log.info(
                "  Base WER=%.4f, CER=%.4f, MTA=%.4f",
                result.base_wer, result.base_cer, result.base_mta,
            )
        except Exception as exc:
            log.warning("  Base transcription failed: %s", exc)
            result.error = str(exc)

        # LoRA transcription
        if has_lora:
            try:
                result.lora_hyp = transcribe_with_lora(sample["audio_path"], provider_id)
                result.lora_wer = compute_wer(result.ref_text, result.lora_hyp)
                result.lora_cer = compute_cer(result.ref_text, result.lora_hyp)
                result.lora_mta = compute_medical_term_accuracy(result.lora_hyp, medical_terms)
                log.info(
                    "  LoRA WER=%.4f, CER=%.4f, MTA=%.4f",
                    result.lora_wer, result.lora_cer, result.lora_mta,
                )
            except Exception as exc:
                log.warning("  LoRA transcription failed: %s", exc)

        report.samples.append(result)

    return report


def save_report(report: EvalReport, provider_id: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    md_path = OUTPUT_DIR / f"asr_eval_{provider_id}.md"
    json_path = OUTPUT_DIR / f"asr_eval_{provider_id}.json"

    md_path.write_text(render_markdown_report(report))
    log.info("Markdown report: %s", md_path)

    json_data = {
        "provider_id": provider_id,
        "has_lora": report.has_lora,
        "summary": {
            "base_avg_wer": report.base_avg_wer,
            "lora_avg_wer": report.lora_avg_wer,
            "base_avg_cer": report.base_avg_cer,
            "lora_avg_cer": report.lora_avg_cer,
            "base_avg_mta": report.base_avg_mta,
            "lora_avg_mta": report.lora_avg_mta,
        },
        "samples": [asdict(s) for s in report.samples],
    }
    json_path.write_text(json.dumps(json_data, indent=2))
    log.info("JSON report: %s", json_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR quality: base Whisper vs LoRA fine-tuned"
    )
    parser.add_argument("--provider", required=True, help="Provider ID")
    parser.add_argument(
        "--base-only", action="store_true",
        help="Only evaluate base model (skip LoRA even if adapter exists)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation to N samples (for quick testing)",
    )
    args = parser.parse_args()

    report = evaluate(
        provider_id=args.provider,
        base_only=args.base_only,
        max_samples=args.max_samples,
    )
    save_report(report, args.provider)

    # Print summary
    print(f"\n{'='*50}")
    print(f"ASR EVAL SUMMARY — {args.provider}")
    print(f"{'='*50}")
    print(f"Samples:      {len(report.samples)}")
    print(f"LoRA adapter: {'Yes' if report.has_lora else 'No'}")
    print(f"\nOverall (all modes):")
    print(f"  Base:  WER={report.base_avg_wer:.4f}  CER={report.base_avg_cer:.4f}")
    if report.has_lora:
        print(f"  LoRA:  WER={report.lora_avg_wer:.4f}  CER={report.lora_avg_cer:.4f}")

    # Mode-split summary — the critical metric for LoRA trained on dictation only
    for mode_label, mode_key in [("Dictation (trained)", "dictation"), ("Ambient (not trained)", "ambient")]:
        mode_samples = [s for s in report.samples if s.mode == mode_key]
        if not mode_samples:
            continue
        m_base = sum(s.base_wer for s in mode_samples if s.base_wer is not None) / len(mode_samples)
        print(f"\n{mode_label} (n={len(mode_samples)}):")
        print(f"  Base WER: {m_base:.4f}")
        if report.has_lora:
            m_lora = sum(s.lora_wer for s in mode_samples if s.lora_wer is not None) / len(mode_samples)
            delta = m_lora - m_base
            pct = -delta / m_base * 100
            tag = f"↑{pct:.1f}% improvement" if pct > 0 else f"↓{abs(pct):.1f}% worse"
            print(f"  LoRA WER: {m_lora:.4f}  ({tag})")


if __name__ == "__main__":
    main()
