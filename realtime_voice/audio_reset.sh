#!/usr/bin/env bash
set -euo pipefail

# Quick helper to re-select mic and speaker after reboot
# - Prefers AB13X USB mic; falls back to other USB/GVAUDIO sources
# - Prefers JBL Go 4 sink; falls back to other USB sinks

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

if ! command -v pactl >/dev/null 2>&1; then
  echo "pactl not found" >&2
  exit 1
fi

# Ensure pulseaudio is running for this user
if ! pactl info >/dev/null 2>&1; then
  if command -v pulseaudio >/dev/null 2>&1; then
    pulseaudio --check >/dev/null 2>&1 || pulseaudio --start || true
    sleep 1
  fi
fi

# Wait briefly for devices to register
for i in {1..5}; do
  ssrc=$(pactl list short sources 2>/dev/null | wc -l | tr -d ' ')
  ssnk=$(pactl list short sinks 2>/dev/null | wc -l | tr -d ' ')
  [ "${ssrc:-0}" -gt 0 ] && [ "${ssnk:-0}" -gt 0 ] && break || sleep 1
done

# Prefer selecting by Pulse card index → ALSA card mapping (unambiguous)
# Defaults (from your system): AB13X = Card #1 (alsa.card "0"), JBL = Card #2 (alsa.card "3")
MIC_PULSE_CARD_INDEX="${MIC_PULSE_CARD_INDEX:-1}"
SINK_PULSE_CARD_INDEX="${SINK_PULSE_CARD_INDEX:-2}"

# Ensure profiles are enabled on those cards (ignore errors if not available)
pactl set-card-profile "$MIC_PULSE_CARD_INDEX" input:analog-stereo 2>/dev/null || \
  pactl set-card-profile "$MIC_PULSE_CARD_INDEX" input:mono-fallback 2>/dev/null || true
pactl set-card-profile "$SINK_PULSE_CARD_INDEX" output:analog-stereo 2>/dev/null || true

# Resolve the ALSA card numbers for those Pulse cards
MIC_ALSA_CARD=$(pactl list cards | awk -v IDX="$MIC_PULSE_CARD_INDEX" '
  $0 ~ "Card #"IDX"$" {f=1} f && /alsa.card =/ {gsub(/\"/,"",$3); print $3; exit}
')
SINK_ALSA_CARD=$(pactl list cards | awk -v IDX="$SINK_PULSE_CARD_INDEX" '
  $0 ~ "Card #"IDX"$" {f=1} f && /alsa.card =/ {gsub(/\"/,"",$3); print $3; exit}
')

# Find the default source name that belongs to MIC_ALSA_CARD
SRC_NAME=$(pactl list sources | awk -v ACARD="$MIC_ALSA_CARD" '
  /^Source #/ {name=""; card=""}
  /\tName: / {name=$2}
  /\talsa.card = "/ {gsub(/\"/,"",$3); card=$3}
  /\talsa.card = "/ && card==ACARD {print name; exit}
')
if [ -z "${SRC_NAME:-}" ]; then
  # Fallback to any non-monitor source
  SRC_NAME=$(pactl list short sources 2>/dev/null | awk '$2 !~ /monitor/ {print $2; exit}') || true
fi
if [ -n "${SRC_NAME:-}" ]; then
  pactl set-default-source "$SRC_NAME" || true
  pactl set-source-mute   "$SRC_NAME" 0 || true
  pactl set-source-volume "$SRC_NAME" 100% || true
  echo "Mic source: $SRC_NAME"
else
  echo "WARN: No mic source found" >&2
fi

# Find the default sink name that belongs to SINK_ALSA_CARD
SINK_NAME=$(pactl list sinks | awk -v ACARD="$SINK_ALSA_CARD" '
  /^Sink #/ {name=""; card=""}
  /\tName: / {name=$2}
  /\talsa.card = "/ {gsub(/\"/,"",$3); card=$3}
  /\talsa.card = "/ && card==ACARD {print name; exit}
')
if [ -z "${SINK_NAME:-}" ]; then
  SINK_NAME=$(pactl list short sinks 2>/dev/null | awk '{print $2; exit}') || true
fi
if [ -z "${SINK_NAME:-}" ]; then
  SINK_NAME=$(pactl list short sinks 2>/dev/null | awk '{print $2; exit}') || true
fi
if [ -n "${SINK_NAME:-}" ]; then
  pactl set-default-sink "$SINK_NAME" || true
  pactl set-sink-mute   "$SINK_NAME" 0 || true
  pactl set-sink-volume "$SINK_NAME" 80% || true
  echo "Speaker sink: $SINK_NAME"
else
  echo "WARN: No speaker sink found" >&2
fi

echo "Audio routing refreshed."


