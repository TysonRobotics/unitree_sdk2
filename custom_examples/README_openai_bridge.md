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



## Wi‑Fi setup on NX (saved steps)

- Unblock and enable Wi‑Fi
```bash
sudo rfkill unblock wifi
sudo nmcli radio wifi on
sudo ip link set wlan0 up
```

- Create an explicit WPA2 profile for iPhone hotspot (handles curly apostrophe in SSID)
```bash
sudo nmcli con add type wifi ifname wlan0 con-name iphone2 ssid "Pim’s iPhone"
sudo nmcli con modify iphone2 \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk 'komdandreirie'
sudo nmcli con up iphone2
```

- Verify
```bash
ip a show wlan0
ping -c 3 8.8.8.8
```

- Notes
  - Joining an external Wi‑Fi may temporarily disable the robot’s own AP; reboot restores it.
  - If a stale connection exists, remove it: `nmcli connection show | grep -i "pim"` then `sudo nmcli connection delete "<name>"`.

## Apt mirrors fix on NX (saved steps)

- Reason: Image shipped with China mirrors (Tsinghua/USTC) and ROS apt lists; in EU these can fail GPG or be slow. We switch to official Ubuntu ports and disable ROS lists (we don’t need ROS here). Backup is kept.

- Commands
```bash
# Backup existing sources
sudo mkdir -p /etc/apt/backup && sudo cp -a /etc/apt/sources.list /etc/apt/sources.list.d /etc/apt/backup/

# Replace with official Ubuntu ports (ARM) for focal
sudo bash -c 'cat >/etc/apt/sources.list <<EOF
deb http://ports.ubuntu.com/ubuntu-ports focal main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports focal-updates main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports focal-backports main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports focal-security main restricted universe multiverse
EOF'

# Remove ROS repo entries (optional for this project)
sudo rm -f /etc/apt/sources.list.d/ros*.list /etc/apt/sources.list.d/ros2*.list

# Update and install audio deps
sudo apt-get update
sudo apt-get install -y libportaudio2 portaudio19-dev libportaudiocpp0 ffmpeg
```

- Backup location
  - `/etc/apt/backup/sources.list`
  - `/etc/apt/backup/sources.list.d/`

