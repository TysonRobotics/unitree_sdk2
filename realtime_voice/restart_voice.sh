#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[restart_voice] Stopping processes..."
pkill -f 'realtime_client.py' 2>/dev/null || true
pkill -f 'web/app.py' 2>/dev/null || true
pkill -f '/realtime_voice/run.sh' 2>/dev/null || true
pkill -f '/realtime_voice/run_web.sh' 2>/dev/null || true
sleep 0.5

echo "[restart_voice] Starting services..."
# Load env and start voice client
nohup bash -lc "cd '$SCRIPT_DIR' && set -a && [ -f .env ] && . ./.env && set +a && ./run.sh" >>"$SCRIPT_DIR/logs/realtime_client.py.log" 2>&1 &
# Start web UI
nohup bash -lc "cd '$SCRIPT_DIR' && set -a && [ -f .env ] && . ./.env && set +a && ./run_web.sh" >>"$SCRIPT_DIR/logs/app.py.log" 2>&1 &
disown || true

echo "[restart_voice] Done."


