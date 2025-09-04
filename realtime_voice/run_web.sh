#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d .venv ]; then
  source .venv/bin/activate
fi

export VOICE_WEB_PORT="${VOICE_WEB_PORT:-9000}"
export VOICE_CONTROL_FIFO="${VOICE_CONTROL_FIFO:-$SCRIPT_DIR/control.fifo}"

python3 "$SCRIPT_DIR/web/app.py"



