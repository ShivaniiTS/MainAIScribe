#!/usr/bin/env bash
# scripts/cleanup_services.sh — Kill stale processes and clean up before (re)starting services.
#
# Usage:
#   bash scripts/cleanup_services.sh          # Clean all service ports
#   bash scripts/cleanup_services.sh 8000     # Clean specific port
#   bash scripts/cleanup_services.sh all      # Clean all + lock files + caches
#
# Called automatically before each service restart.

set -euo pipefail

# All ports used by AI Scribe services
SERVICE_PORTS=(8000 8100 3000 3100 8081)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo -e "${YELLOW}Killing processes on port $port: $pids${NC}"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
        # Verify
        local remaining
        remaining=$(lsof -ti ":$port" 2>/dev/null || true)
        if [ -n "$remaining" ]; then
            echo -e "${RED}WARNING: Port $port still in use by: $remaining${NC}"
            return 1
        fi
        echo -e "${GREEN}Port $port is free${NC}"
    else
        echo -e "${GREEN}Port $port already free${NC}"
    fi
}

clean_lock_files() {
    local web_dir="$1"
    local lock="$web_dir/.next/dev/lock"
    if [ -f "$lock" ]; then
        echo -e "${YELLOW}Removing stale Next.js lock: $lock${NC}"
        rm -f "$lock"
    fi
}

clean_turbopack_cache() {
    local web_dir="$1"
    local cache_dir="$web_dir/.next/dev/cache"
    if [ -d "$cache_dir" ]; then
        # Check for corrupted cache (files > 100MB suggest corruption)
        local large_files
        large_files=$(find "$cache_dir" -size +100M 2>/dev/null | head -1)
        if [ -n "$large_files" ]; then
            echo -e "${YELLOW}Clearing corrupted Turbopack cache in $web_dir${NC}"
            rm -rf "$web_dir/.next"
        fi
    fi
}

kill_stale_node_processes() {
    # Kill any orphaned next dev / expo processes
    local stale
    stale=$(ps aux | grep -E 'next dev|expo start|metro' | grep -v grep | awk '{print $2}' 2>/dev/null || true)
    if [ -n "$stale" ]; then
        echo -e "${YELLOW}Killing stale Node.js processes: $stale${NC}"
        echo "$stale" | xargs kill -9 2>/dev/null || true
    fi
}

kill_stale_uvicorn() {
    local stale
    stale=$(ps aux | grep 'uvicorn api.main:app' | grep -v grep | awk '{print $2}' 2>/dev/null || true)
    if [ -n "$stale" ]; then
        echo -e "${YELLOW}Killing stale uvicorn processes: $stale${NC}"
        echo "$stale" | xargs kill -9 2>/dev/null || true
    fi
}

# ── Main ──

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== AI Scribe Service Cleanup ==="
echo ""

if [ "${1:-all}" = "all" ] || [ $# -eq 0 ]; then
    # Kill all stale processes first
    kill_stale_uvicorn
    kill_stale_node_processes
    sleep 1

    # Free all ports
    for port in "${SERVICE_PORTS[@]}"; do
        kill_port "$port"
    done

    # Clean lock files and corrupted caches
    clean_lock_files "$PROJECT_ROOT/client/web"
    clean_turbopack_cache "$PROJECT_ROOT/client/web"

    echo ""
    echo -e "${GREEN}All services cleaned up${NC}"
else
    # Clean specific port
    kill_port "$1"
fi
