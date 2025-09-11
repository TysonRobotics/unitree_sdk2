#!/usr/bin/env bash
set -euo pipefail

# This script:
# 1) Selects the AB13X USB mic and JBL Go 4 speaker in PulseAudio
# 2) Loads env from .env
# 3) Starts the realtime voice client and web UI in the background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Ensure we have a runtime dir for Pulse when running non-interactively
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

# 1) Ensure PulseAudio is available, then set default source/sink
if command -v pactl >/dev/null 2>&1; then
  if ! pactl info >/dev/null 2>&1; then
    echo "[boot_voice] Starting PulseAudio for user $(id -un)"
    if command -v pulseaudio >/dev/null 2>&1; then
      pulseaudio --check >/dev/null 2>&1 || pulseaudio --start || true
      sleep 1
    fi
  fi

  # Wait up to 5s for sources/sinks to appear
  for i in {1..5}; do
    SSRC_CNT=$(pactl list short sources 2>/dev/null | wc -l | tr -d ' ')
    SSNK_CNT=$(pactl list short sinks 2>/dev/null | wc -l | tr -d ' ')
    [ "${SSRC_CNT:-0}" -gt 0 ] && [ "${SSNK_CNT:-0}" -gt 0 ] && break || sleep 1
  done

  # Source (mic): prefer AB13X, fallback to GVAUDIO or generic USB
  SRC_NAME=$(pactl list short sources 2>/dev/null | awk 'BEGIN{IGNORECASE=1} $2 ~ /ab13x/ {print $2; exit} $2 ~ /gvaudio/ {print $2; exit} ($2 ~ /usb/ && $2 !~ /monitor/) {print $2; exit}') || true
  if [ -z "${SRC_NAME:-}" ]; then
    # Fallback to first non-monitor source
    SRC_NAME=$(pactl list short sources 2>/dev/null | awk '$2 !~ /monitor/ {print $2; exit}') || true
  fi
  if [ -n "${SRC_NAME:-}" ]; then
    pactl set-default-source "$SRC_NAME" || true
    pactl set-source-mute "$SRC_NAME" 0 || true
    pactl set-source-volume "$SRC_NAME" 100% || true
    echo "[boot_voice] Default source: $SRC_NAME"
  else
    echo "[boot_voice] WARN: No matching mic source found"
  fi

  # Sink (speaker): prefer JBL, fallback to Zgmicro or generic USB sink
  SINK_NAME=$(pactl list short sinks 2>/dev/null | awk 'BEGIN{IGNORECASE=1} $2 ~ /jbl/ {print $2; exit} $2 ~ /zgmicro/ {print $2; exit} ($2 ~ /usb/) {print $2; exit}') || true
  if [ -z "${SINK_NAME:-}" ]; then
    SINK_NAME=$(pactl list short sinks 2>/dev/null | awk '{print $2; exit}') || true
  fi
  if [ -n "${SINK_NAME:-}" ]; then
    pactl set-default-sink "$SINK_NAME" || true
    pactl set-sink-mute "$SINK_NAME" 0 || true
    pactl set-sink-volume "$SINK_NAME" 70% || true
    echo "[boot_voice] Default sink: $SINK_NAME"
  else
    echo "[boot_voice] WARN: No matching speaker sink found"
  fi
else
  echo "[boot_voice] WARN: pactl not found; skipping PulseAudio routing"
fi

# 2) Load environment (API key, device preferences, etc.)
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
else
  echo "[boot_voice] WARN: $ENV_FILE not found; using defaults"
fi

# Sensible defaults if not provided in .env
export PREFERRED_INPUT="${PREFERRED_INPUT:-ab13x}"
export PREFERRED_OUTPUT="${PREFERRED_OUTPUT:-jbl}"
export AUDIO_SAMPLE_RATE="${AUDIO_SAMPLE_RATE:-48000}"
export AUDIO_BLOCK_SIZE="${AUDIO_BLOCK_SIZE:-2048}"
export VOICE="${VOICE:-marin}"
export SKIP_INTRO="${SKIP_INTRO:-1}"
export VOICE_WEB_PORT="${VOICE_WEB_PORT:-9000}"

# 3) Start processes if not already running
cd "$SCRIPT_DIR"

start_if_not_running() {
  local pattern="$1"; shift
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "[boot_voice] Already running: $pattern"
  else
    echo "[boot_voice] Starting: $*"
    nohup "$@" >>"$LOG_DIR/$(basename "$pattern").log" 2>&1 &
    disown || true
  fi
}

# Voice client
start_if_not_running "realtime_client.py" bash -lc "exec ./run.sh"

# Web UI
start_if_not_running "web/app.py" bash -lc "exec ./run_web.sh"

echo "[boot_voice] Done. Logs in $LOG_DIR"



