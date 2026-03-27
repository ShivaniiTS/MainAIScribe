#!/usr/bin/env bash
# scripts/start_services.sh — Clean up stale processes, then start all AI Scribe services.
#
# Usage:
#   bash scripts/start_services.sh              # Start all services
#   bash scripts/start_services.sh --no-mobile  # Skip Expo mobile app
#
# Services started:
#   1. Provider-facing API    (port 8000)
#   2. Processing-pipeline API (port 8100)
#   3. Provider web UI         (port 3000)
#   4. Admin web UI            (port 3100)
#   5. Expo mobile app         (port 8081) — unless --no-mobile

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/tmp/ai-scribe-logs"
SKIP_MOBILE=false

for arg in "$@"; do
    case $arg in
        --no-mobile) SKIP_MOBILE=true ;;
    esac
done

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$LOG_DIR"

# ── Step 1: Clean up stale processes ──
echo "=== Step 1: Cleanup ==="
bash "$SCRIPT_DIR/cleanup_services.sh"
echo ""

# ── Step 2: Start services ──
echo "=== Step 2: Starting services ==="
echo ""

# Activate Python venv
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# 2a. Provider-facing API (port 8000)
echo -e "${YELLOW}Starting provider-facing API on port 8000...${NC}"
AI_SCRIBE_SERVER_ROLE=provider-facing \
    nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 \
    > "$LOG_DIR/api-provider.log" 2>&1 &
echo "  PID: $! → log: $LOG_DIR/api-provider.log"

# 2b. Processing-pipeline API (port 8100)
echo -e "${YELLOW}Starting pipeline API on port 8100...${NC}"
AI_SCRIBE_SERVER_ROLE=processing-pipeline \
    nohup uvicorn api.main:app --host 0.0.0.0 --port 8100 \
    > "$LOG_DIR/api-pipeline.log" 2>&1 &
echo "  PID: $! → log: $LOG_DIR/api-pipeline.log"

# Wait for APIs to be ready
sleep 3

# 2c. Provider web UI (port 3000)
echo -e "${YELLOW}Starting provider web UI on port 3000...${NC}"
cd "$PROJECT_ROOT/client/web"
NEXT_PUBLIC_API_URL=http://localhost:8000 \
    nohup npx next dev --port 3000 \
    > "$LOG_DIR/web-provider.log" 2>&1 &
echo "  PID: $! → log: $LOG_DIR/web-provider.log"

# 2d. Admin web UI (port 3100)
echo -e "${YELLOW}Starting admin web UI on port 3100...${NC}"
NEXT_PUBLIC_API_URL=http://localhost:8100 \
    nohup npx next dev --port 3100 \
    > "$LOG_DIR/web-admin.log" 2>&1 &
echo "  PID: $! → log: $LOG_DIR/web-admin.log"

cd "$PROJECT_ROOT"

# 2e. Expo mobile app (port 8081)
if [ "$SKIP_MOBILE" = false ]; then
    echo -e "${YELLOW}Starting Expo mobile app on port 8081...${NC}"
    cd "$PROJECT_ROOT/client/mobile"
    nohup npx expo start --tunnel \
        > "$LOG_DIR/mobile-expo.log" 2>&1 &
    echo "  PID: $! → log: $LOG_DIR/mobile-expo.log"
    cd "$PROJECT_ROOT"
fi

# ── Step 3: Wait and verify ──
echo ""
echo "=== Step 3: Verifying ==="
sleep 8

PASS=0
FAIL=0

check_service() {
    local url=$1
    local name=$2
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
        echo -e "  ${GREEN}✓${NC} $name ($url) — OK"
        PASS=$((PASS+1))
    else
        echo -e "  ${RED}✗${NC} $name ($url) — HTTP $code"
        FAIL=$((FAIL+1))
    fi
}

check_service "http://localhost:8000/health" "Provider API"
check_service "http://localhost:8100/health" "Pipeline API"
check_service "http://localhost:3000" "Provider Web UI"
check_service "http://localhost:3100" "Admin Web UI"
if [ "$SKIP_MOBILE" = false ]; then
    check_service "http://localhost:8081" "Expo Mobile"
fi

echo ""
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All services running ($PASS/$PASS)${NC}"
else
    echo -e "${RED}$FAIL service(s) failed to start. Check logs in $LOG_DIR/${NC}"
fi

echo ""
echo "Logs: $LOG_DIR/"
ls -1 "$LOG_DIR/"*.log 2>/dev/null | while read f; do echo "  $f"; done
