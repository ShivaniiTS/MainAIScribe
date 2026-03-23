#!/usr/bin/env python3
"""
Run the full AI Scribe pipeline on an audio file and print the result.

Usage:
    python scripts/run_pipeline.py <audio_file> [--mode dictation|ambient] [--save]

Examples:
    python scripts/run_pipeline.py ai-scribe-data/dictation/dr_faraz_rahman/riley_dew_226680_20260219/dictation.mp3
    python scripts/run_pipeline.py ai-scribe-data/conversation/dr_faraz_rahman/javier_waters_227534_20260303/conversation_audio.mp3 --mode ambient
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI Scribe pipeline on an audio file.")
    parser.add_argument("audio", help="Path to audio file (.mp3, .wav, .m4a, ...)")
    parser.add_argument("--mode", choices=["dictation", "ambient"], default="dictation")
    parser.add_argument("--save", action="store_true", help="Save transcript + note to output/ dir")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    from mcp_servers.llm.ollama_server import OllamaServer
    from orchestrator.graph import build_graph, run_encounter
    from orchestrator.nodes.note_node import set_llm_engine_factory
    from orchestrator.nodes.transcribe_node import set_asr_engine_factory
    from orchestrator.state import (
        DeliveryMethod, EncounterState, NoteType,
        ProviderProfile, RecordingMode,
    )

    model = _discover_ollama_model()
    set_asr_engine_factory(None)   # use real WhisperX
    set_llm_engine_factory(lambda: OllamaServer(model_overrides={"note_generation": model}))

    mode = RecordingMode.DICTATION if args.mode == "dictation" else RecordingMode.AMBIENT

    state = EncounterState(
        provider_id="cli-provider",
        patient_id="cli-patient",
        provider_profile=ProviderProfile(
            id="cli-provider",
            name="CLI Provider",
            specialty="general",
            note_format=NoteType.SOAP,
            template_id="soap_default",
        ),
        recording_mode=mode,
        delivery_method=DeliveryMethod.CLIPBOARD,
        audio_file_path=str(audio_path),
    )

    print(f"Audio:  {audio_path}")
    print(f"Mode:   {args.mode}")
    print(f"Model:  {model}")
    print("=" * 70)

    t0 = time.time()
    final = run_encounter(build_graph(), state)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s  |  ASR {final.metrics.asr_duration_ms}ms  |  LLM {final.metrics.note_gen_ms}ms")
    print(f"ASR engine: {final.asr_engine_used}  |  confidence: {final.metrics.asr_confidence}")
    print(f"Status: {final.status}  |  Errors: {final.errors or 'none'}")

    print("\n" + "=" * 70)
    print("TRANSCRIPT")
    print("=" * 70)
    print(final.transcript.full_text)

    print("\n" + "=" * 70)
    print("GENERATED NOTE")
    print("=" * 70)
    print(final.final_note.to_text())

    if args.save:
        from config.paths import OUTPUT_DIR
        out_dir = OUTPUT_DIR / audio_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "transcript.txt").write_text(final.transcript.full_text)
        (out_dir / "note.md").write_text(final.final_note.to_text())
        print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
