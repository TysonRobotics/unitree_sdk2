#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
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

exec python3 realtime_client.py


