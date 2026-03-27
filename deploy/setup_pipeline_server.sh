#!/usr/bin/env bash
# deploy/setup_pipeline_server.sh — Setup script for the processing pipeline server.
#
# Installs GPU dependencies (CUDA, WhisperX, Ollama), configures the server
# for processing-pipeline role, and optionally sets up systemd services.
#
# Usage:
#   bash deploy/setup_pipeline_server.sh [--with-systemd] [--skip-ollama]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "AI Scribe — Processing Pipeline Server Setup"
echo "============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
WITH_SYSTEMD=false
SKIP_OLLAMA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-systemd) WITH_SYSTEMD=true; shift ;;
        --skip-ollama) SKIP_OLLAMA=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Check GPU availability
# ---------------------------------------------------------------------------
echo "[1/7] Checking GPU..."

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  ✓ GPU detected"
else
    echo "  ⚠ No GPU detected. WhisperX and Ollama will run on CPU (very slow)"
fi

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------
echo "[2/7] Setting up Python environment..."

cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --quiet --upgrade pip

# Install full dependencies including GPU extras
pip install --quiet -e ".[api]" 2>/dev/null || pip install --quiet -e .

echo "  ✓ Python environment ready"

# ---------------------------------------------------------------------------
# 3. Ollama (LLM inference)
# ---------------------------------------------------------------------------
if [ "$SKIP_OLLAMA" = false ]; then
    echo "[3/7] Setting up Ollama..."

    if command -v ollama &> /dev/null; then
        echo "  Ollama already installed"
    else
        echo "  Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    # Pull required models
    echo "  Pulling models..."
    ollama pull qwen2.5:14b 2>/dev/null && echo "  ✓ qwen2.5:14b ready" \
        || echo "  ⚠ Failed to pull qwen2.5:14b (pull manually: ollama pull qwen2.5:14b)"

    ollama pull llama3.1:latest 2>/dev/null && echo "  ✓ llama3.1:latest ready (judge model)" \
        || echo "  ⚠ Failed to pull llama3.1:latest"
else
    echo "[3/7] Ollama setup skipped"
fi

# ---------------------------------------------------------------------------
# 4. Configuration
# ---------------------------------------------------------------------------
echo "[4/7] Configuring deployment..."

cat > "$PROJECT_ROOT/.env.pipeline" << EOF
# AI Scribe — Processing Pipeline Server Configuration
AI_SCRIBE_SERVER_ROLE=processing-pipeline
AI_SCRIBE_DATA_DIR=$PROJECT_ROOT/ai-scribe-data
AI_SCRIBE_OUTPUT_DIR=$PROJECT_ROOT/output
AI_SCRIBE_CONFIG_DIR=$PROJECT_ROOT/config

# Ollama LLM server
OLLAMA_HOST=http://localhost:11434

# HuggingFace token (required for pyannote diarization)
# HF_TOKEN=your_token_here
EOF

echo "  ✓ Configuration written to .env.pipeline"

# ---------------------------------------------------------------------------
# 5. Data directories
# ---------------------------------------------------------------------------
echo "[5/7] Creating data directories..."

mkdir -p "$PROJECT_ROOT/ai-scribe-data/dictation"
mkdir -p "$PROJECT_ROOT/ai-scribe-data/conversation"
mkdir -p "$PROJECT_ROOT/output/dictation"
mkdir -p "$PROJECT_ROOT/output/conversation"
mkdir -p "$PROJECT_ROOT/config/providers"
mkdir -p "$PROJECT_ROOT/config/templates"
mkdir -p "$PROJECT_ROOT/config/dictionaries"

echo "  ✓ Data directories created"

# ---------------------------------------------------------------------------
# 6. Verify pipeline components
# ---------------------------------------------------------------------------
echo "[6/7] Verifying pipeline components..."

python3 -c "
import importlib
checks = [
    ('api.main', 'FastAPI app'),
    ('api.pipeline.routes', 'Pipeline API routes'),
    ('config.deployment', 'Deployment config'),
    ('config.paths', 'Path config'),
]
for mod, label in checks:
    try:
        importlib.import_module(mod)
        print(f'  ✓ {label}')
    except Exception as e:
        print(f'  ✗ {label}: {e}')
" 2>/dev/null || echo "  ⚠ Some components failed to import (run tests to diagnose)"

# ---------------------------------------------------------------------------
# 7. Systemd (optional)
# ---------------------------------------------------------------------------
if [ "$WITH_SYSTEMD" = true ]; then
    echo "[7/7] Setting up systemd services..."

    sudo tee /etc/systemd/system/ai-scribe-pipeline.service > /dev/null << SYSTEMD_EOF
[Unit]
Description=AI Scribe Processing Pipeline API
After=network.target ollama.service

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$PROJECT_ROOT/.env.pipeline
ExecStart=$PROJECT_ROOT/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8100
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

    sudo systemctl daemon-reload
    sudo systemctl enable ai-scribe-pipeline
    echo "  ✓ Systemd service created (start with: sudo systemctl start ai-scribe-pipeline)"
else
    echo "[7/7] Systemd setup skipped (use --with-systemd to enable)"
fi

echo ""
echo "============================================="
echo "Setup complete!"
echo ""
echo "To start the server:"
echo "  cd $PROJECT_ROOT"
echo "  set -a && source .env.pipeline && set +a"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8100"
echo ""
echo "API docs: http://localhost:8100/docs"
echo "============================================="
