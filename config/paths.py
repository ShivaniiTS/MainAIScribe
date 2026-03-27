"""
config/paths.py — Centralized, role-aware data paths.

All data directory paths are resolved here. Modules import from this file
instead of hardcoding paths.

Path resolution order:
1. Environment variables (highest priority — useful for production/Docker)
2. Deployment config (deployment.yaml + AI_SCRIBE_SERVER_ROLE)
3. Defaults (ai-scribe-data/ and output/)

Environment variables:
    AI_SCRIBE_ROOT        Base project root (default: auto-detected)
    AI_SCRIBE_DATA_DIR    Patient encounter data
    AI_SCRIBE_OUTPUT_DIR  Pipeline output
    AI_SCRIBE_CONFIG_DIR  Config files (default: {root}/config)

When running dual servers from the same codebase, the server role determines
which data/output directories are used:
    provider-facing     → ai-scribe-data/ + output/
    processing-pipeline → pipeline-data/  + pipeline-output/
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root: two levels up from config/paths.py → repo root
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

ROOT = Path(os.environ.get("AI_SCRIBE_ROOT", str(_DEFAULT_ROOT)))
CONFIG_DIR = Path(os.environ.get("AI_SCRIBE_CONFIG_DIR", str(ROOT / "config")))
PROVIDERS_DIR = CONFIG_DIR / "providers"


def _resolve_data_dirs() -> tuple[Path, Path]:
    """Resolve DATA_DIR and OUTPUT_DIR based on env vars, then deployment config."""
    # Environment variables take highest priority
    env_data = os.environ.get("AI_SCRIBE_DATA_DIR")
    env_output = os.environ.get("AI_SCRIBE_OUTPUT_DIR")
    if env_data and env_output:
        return Path(env_data), Path(env_output)

    # Determine role from env var or YAML config
    role = os.environ.get("AI_SCRIBE_SERVER_ROLE", "").strip()
    if not role:
        try:
            import yaml
            config_path = CONFIG_DIR / "deployment.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    raw = yaml.safe_load(f) or {}
                role = raw.get("server", {}).get("role", "provider-facing")
        except Exception:
            role = "provider-facing"

    # Resolve paths based on role
    try:
        import yaml
        config_path = CONFIG_DIR / "deployment.yaml"
        if config_path.exists():
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            data_section = raw.get("data", {})

            if role == "processing-pipeline":
                pp = data_section.get("processing_pipeline", {})
                data_dir = ROOT / pp.get("data_dir", "pipeline-data")
                output_dir = ROOT / pp.get("output_dir", "pipeline-output")
                return data_dir, output_dir
            else:
                pf = data_section.get("provider_facing", {})
                data_dir = ROOT / pf.get("data_dir", "ai-scribe-data")
                output_dir = ROOT / pf.get("output_dir", "output")
                return data_dir, output_dir
    except Exception:
        pass  # Fall through to defaults

    # Defaults
    data_default = ROOT / "ai-scribe-data"
    output_default = ROOT / "output"
    return (
        Path(env_data) if env_data else data_default,
        Path(env_output) if env_output else output_default,
    )


DATA_DIR, OUTPUT_DIR = _resolve_data_dirs()
