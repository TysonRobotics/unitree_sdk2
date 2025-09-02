Realtime voice client (OpenAI Realtime API)

Requirements
- Python 3.8+
- OpenAI API key in env: OPENAI_API_KEY
- System audio libs (ALSA/PortAudio). On Ubuntu/Jetson: sudo apt install libportaudio2

Quick start
1) Create/activate venv and install deps (already handled by run.sh on first run)
2) Export your key
   export OPENAI_API_KEY="..."
3) Optional: skip model intro (reduces initial feedback)
   export SKIP_INTRO=1
4) Run (basic non-AEC, non-barge)
   AUDIO_SAMPLE_RATE=48000 AUDIO_BLOCK_SIZE=2048 ./run.sh

Echo cancellation (PulseAudio WebRTC AEC)
1) Find devices
   pactl list short sources
   pactl list short sinks
2) Load echo-cancel (48 kHz, mono)
   pactl unload-module module-echo-cancel 2>/dev/null || true
   pactl load-module module-echo-cancel \
     source_master=<your_mic_source_name> \
     sink_master=<your_speaker_sink_name> \
     aec_method=webrtc rate=48000 channels=1 \
     source_name=ec_source sink_name=ec_sink
3) Make them defaults (optional)
   pactl set-default-source ec_source
   pactl set-default-sink  ec_sink
   pactl info | grep 'Default Sink\|Default Source'
4) Point the client to AEC devices
   export PREFERRED_INPUT=ec_source
   export PREFERRED_OUTPUT=ec_sink
5) Reduce speaker volume
   pactl set-sink-volume ec_sink 40%

VAD and barge-in controls (env vars)
- INPUT_RMS_GATE: base RMS gate (default 0.010)
- VAD_START_GATE: start threshold (default ≈ 1.3× base)
- VAD_STOP_GATE: stop threshold (default ≈ base)
- VAD_MIN_SPEECH_MS: min speech before commit (default 300)
- VAD_SILENCE_MS: trailing silence to commit (default 400)
- HANGOVER_MS: mic re-open delay after bot reply (default 400)
- BARGE_IN=1: enable interruptible mode (response.cancel on user speech)
- BARGE_CANCEL_COOLDOWN_MS: avoid rapid re-cancels (default 250)

Profiles
Dictation (single long utterance, no barge-in)
   export BARGE_IN=0
   export SKIP_INTRO=1
   INPUT_RMS_GATE=0.012 VAD_START_GATE=0.017 VAD_STOP_GATE=0.010 \
   VAD_MIN_SPEECH_MS=2000 VAD_SILENCE_MS=1000 HANGOVER_MS=1200 \
   AUDIO_SAMPLE_RATE=48000 AUDIO_BLOCK_SIZE=2048 ./run.sh

Interruptible but stable
   export BARGE_IN=1
   export BARGE_CANCEL_COOLDOWN_MS=600
   export SKIP_INTRO=1
   INPUT_RMS_GATE=0.012 VAD_START_GATE=0.016 VAD_STOP_GATE=0.010 \
   VAD_MIN_SPEECH_MS=1500 VAD_SILENCE_MS=800 HANGOVER_MS=1000 \
   AUDIO_SAMPLE_RATE=48000 AUDIO_BLOCK_SIZE=2048 ./run.sh

Tuning workflow
1) With DEBUG=1, note "Mic level ~ ..." while quiet (noise_RMS) and while speaking (speech_RMS).
2) Choose gates:
   - VAD_STOP_GATE ≈ noise_RMS × 1.15
   - VAD_START_GATE ≈ min(noise_RMS × 1.7, speech_RMS × 0.8)
   - INPUT_RMS_GATE ≈ VAD_STOP_GATE
3) Prevent early commits: increase VAD_MIN_SPEECH_MS (700–2000) and VAD_SILENCE_MS (300–1200)
4) If sluggish: lower VAD_SILENCE_MS; if still hears itself: raise gates slightly and/or HANGOVER_MS

Device routing tips
- The client prefers env-specified devices first (PREFERRED_INPUT, PREFERRED_OUTPUT), then ec_* / pulse, then USB devices by name.
- Verify routing:
   pactl list short sources | grep -E 'ec_|pulse'
   pactl list short sinks   | grep -E 'ec_|pulse'

Known limitations (current)
- Barge-in mode works but may still exhibit delay depending on thresholds and AEC convergence.
- If you see input_audio_buffer_commit_empty, raise VAD_MIN_SPEECH_MS or speak a bit longer before pausing.
- If you see conversation_already_has_active_response, wait for reply end or increase VAD_SILENCE_MS/HANGOVER_MS.

Setup
1) Create venv and install deps
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip setuptools wheel
   pip install websockets numpy sounddevice soundfile pyalsaaudio requests pynacl

2) Export API key
   export OPENAI_API_KEY="..."

3) Optional env tuning
   export AUDIO_SAMPLE_RATE=24000
   export AUDIO_BLOCK_SIZE=1024
   export VOICE=marin

Run
   python realtime_client.py

Config
- Persona file: realtime_voice/config/persona.json
  - JSON format example:
    {
      "intro": "Hi, I'm Nova, your helpful workspace assistant.",
      "style": "Friendly, concise, with a touch of humor",
      "behaviors": [
        "Always confirm safety before executing commands",
        "Ask clarifying questions when context is missing"
      ]
    }
- Facts file: realtime_voice/config/facts.json
  - JSON array of strings, example:
    [
      "This robot runs on Jetson hardware.",
      "Primary mic is GVAUDIO; speaker is JBL Go 4.",
      "Network is WPA2 with limited bandwidth in lab."
    ]
- Optional env overrides:
  - PERSONA_PATH=/abs/path/to/persona.json
  - FACTS_PATH=/abs/path/to/facts.json

Usage
- The program will auto-select input device by names containing "gvaudio" or "usb audio" and output device containing "jbl" or "go 4".
- Interactive controls:
  - m + Enter: toggle microphone mute/unmute
  - s + Enter: stop/interrupt current assistant speech immediately
  - r + Enter: reload persona/facts configs, update LLM instructions, and request a fresh intro
  - q + Enter: quit

Quick test commands
- Run with defaults (auto device selection, 24 kHz):
  OPENAI_API_KEY=... ./run.sh
- With DEBUG and barge-in enabled at 48 kHz for AEC setups:
  DEBUG=1 BARGE_IN=1 AUDIO_SAMPLE_RATE=48000 AUDIO_BLOCK_SIZE=2048 OPENAI_API_KEY=... ./run.sh
- Target echo-cancel devices explicitly:
  PREFERRED_INPUT=ec_source PREFERRED_OUTPUT=ec_sink OPENAI_API_KEY=... ./run.sh

Notes
- If sounddevice import fails, install PortAudio: sudo apt install libportaudio2 then reinstall sounddevice in the venv.
- Default Realtime URL can be overridden with OPENAI_REALTIME_URL.


