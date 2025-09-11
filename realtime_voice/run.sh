#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python3 -m pip install -q --upgrade pip setuptools wheel
python3 -m pip install -q websockets numpy sounddevice soundfile pyalsaaudio requests pynacl

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set" >&2
  exit 2
fi

# Default male voice
export VOICE="${VOICE:-alloy}"

# Defaults for VAD if not provided
export VAD_SILENCE_MS="${VAD_SILENCE_MS:-2000}"
export VAD_START_GATE="${VAD_START_GATE:-0.010}"

# Disable echo-cancel device selection by default (we hard-mute during playback)
export DISABLE_AEC="${DISABLE_AEC:-1}"

# Enable conversation logging by default
export LOG_CONVERSATION="${LOG_CONVERSATION:-1}"

# Longer hangover to prevent mic pickup of speaker output
export HANGOVER_MS="${HANGOVER_MS:-500}"

# Default to server VAD (OpenAI handles speech detection)
export VAD_MODE="${VAD_MODE:-server}"

# VAD timeout to prevent getting stuck
export VAD_TIMEOUT_MS="${VAD_TIMEOUT_MS:-10000}"

exec python3 realtime_client.py


