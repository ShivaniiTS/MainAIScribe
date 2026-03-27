#!/usr/bin/env bash
# deploy/setup_provider_server.sh — Setup script for the provider-facing server.
#
# Installs CPU-only dependencies, configures the server for provider-facing role,
# and optionally sets up systemd services and nginx reverse proxy.
#
# Usage:
#   bash deploy/setup_provider_server.sh [--with-nginx] [--with-systemd]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "AI Scribe — Provider-Facing Server Setup"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
WITH_NGINX=false
WITH_SYSTEMD=false
PIPELINE_URL="http://localhost:8100"

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-nginx) WITH_NGINX=true; shift ;;
        --with-systemd) WITH_SYSTEMD=true; shift ;;
        --pipeline-url) PIPELINE_URL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------
echo "[1/6] Setting up Python environment..."

cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --quiet --upgrade pip

# Install API dependencies only (no GPU deps)
pip install --quiet -e ".[api]" 2>/dev/null || pip install --quiet -e .

echo "  ✓ Python environment ready"

# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------
echo "[2/6] Configuring deployment..."

# Create .env file for provider-facing role
cat > "$PROJECT_ROOT/.env.provider" << EOF
# AI Scribe — Provider-Facing Server Configuration
AI_SCRIBE_SERVER_ROLE=provider-facing
AI_SCRIBE_DATA_DIR=$PROJECT_ROOT/ai-scribe-data
AI_SCRIBE_OUTPUT_DIR=$PROJECT_ROOT/output
AI_SCRIBE_CONFIG_DIR=$PROJECT_ROOT/config

# Pipeline server URL (for proxying pipeline requests)
# Update this to point to the actual pipeline server
PIPELINE_API_URL=$PIPELINE_URL
EOF

echo "  ✓ Configuration written to .env.provider"

# ---------------------------------------------------------------------------
# 3. Data directories
# ---------------------------------------------------------------------------
echo "[3/6] Creating data directories..."

mkdir -p "$PROJECT_ROOT/ai-scribe-data/dictation"
mkdir -p "$PROJECT_ROOT/ai-scribe-data/conversation"
mkdir -p "$PROJECT_ROOT/output/dictation"
mkdir -p "$PROJECT_ROOT/output/conversation"
mkdir -p "$PROJECT_ROOT/config/providers"
mkdir -p "$PROJECT_ROOT/config/templates"
mkdir -p "$PROJECT_ROOT/config/dictionaries"

echo "  ✓ Data directories created"

# ---------------------------------------------------------------------------
# 4. Initial config sync from pipeline server
# ---------------------------------------------------------------------------
echo "[4/6] Running initial config sync..."

set -a && source "$PROJECT_ROOT/.env.provider" && set +a

python scripts/sync_templates.py 2>/dev/null && echo "  ✓ Config synced from pipeline server" \
    || echo "  ⚠ Config sync failed (pipeline server may not be running yet)"

# ---------------------------------------------------------------------------
# 5. Nginx (optional)
# ---------------------------------------------------------------------------
if [ "$WITH_NGINX" = true ]; then
    echo "[5/6] Setting up nginx..."

    sudo tee /etc/nginx/sites-available/ai-scribe-provider > /dev/null << 'NGINX_EOF'
server {
    listen 80;
    server_name _;

    # API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Web UI
    location / {
        proxy_pass http://127.0.0.1:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
NGINX_EOF

    sudo ln -sf /etc/nginx/sites-available/ai-scribe-provider /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx
    echo "  ✓ Nginx configured"
else
    echo "[5/6] Nginx setup skipped (use --with-nginx to enable)"
fi

# ---------------------------------------------------------------------------
# 6. Systemd (optional)
# ---------------------------------------------------------------------------
if [ "$WITH_SYSTEMD" = true ]; then
    echo "[6/6] Setting up systemd services..."

    sudo tee /etc/systemd/system/ai-scribe-api.service > /dev/null << SYSTEMD_EOF
[Unit]
Description=AI Scribe Provider-Facing API
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$PROJECT_ROOT
EnvironmentFile=$PROJECT_ROOT/.env.provider
ExecStart=$PROJECT_ROOT/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

    sudo systemctl daemon-reload
    sudo systemctl enable ai-scribe-api
    echo "  ✓ Systemd service created (start with: sudo systemctl start ai-scribe-api)"
else
    echo "[6/6] Systemd setup skipped (use --with-systemd to enable)"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo ""
echo "To start the server:"
echo "  cd $PROJECT_ROOT"
echo "  set -a && source .env.provider && set +a"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Pipeline server URL: $PIPELINE_URL"
echo "========================================"
