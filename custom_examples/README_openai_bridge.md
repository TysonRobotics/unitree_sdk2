OpenAI voice bridge (simple)

Overview
- The C++ example `g1_audio_play_wav` now exposes a TCP speaker bridge on 127.0.0.1:5002.
- Send raw 16 kHz mono 16-bit PCM to that socket; it streams to the robot speaker via AudioClient.
- You can still record (`r`) and play files (`p`) from the same app, and use TTS (`t`).

Python client outline (cloud LLM, simple TTS)
1) Capture mic (16k mono s16le)
2) Send to OpenAI for ASR + chat completion
3) Synthesize TTS (e.g., OpenAI TTS or any provider); decode to 16k mono s16le
4) Stream PCM to 127.0.0.1:5002

Example (pseudo):
```python
import socket, sounddevice as sd
import numpy as np

def send_pcm_to_robot(pcm_s16):
    s = socket.socket(); s.connect(("127.0.0.1", 5002))
    s.sendall(pcm_s16.tobytes()); s.close()

# 1) record 2 seconds as an example
sr = 16000
audio = sd.rec(int(2*sr), samplerate=sr, channels=1, dtype='int16'); sd.wait()

# 2) call OpenAI ASR + LLM -> response_text (omitted)
response_text = "Hello from the cloud"

# 3) TTS -> pcm_s16 (omitted); ensure 16k mono int16 numpy array
tts_pcm = np.zeros((sr,), dtype=np.int16)  # placeholder

# 4) stream to robot
send_pcm_to_robot(tts_pcm)
```

Run
```bash
# build
cd /home/pim/Desktop/unitree_sdk2/build
make -j"$(nproc)" g1_audio_play_wav

# run bridge + controls
./bin/g1_audio_play_wav enp62s0
# send PCM to 127.0.0.1:5002
```

Notes
- PCM must be 16kHz mono signed 16-bit.
- Bridge uses 1s chunks and paces at ~1s/chunk.
- For Realtime/WebRTC integrations, feed microphone upstream and stream returned TTS audio down to the TCP bridge.


