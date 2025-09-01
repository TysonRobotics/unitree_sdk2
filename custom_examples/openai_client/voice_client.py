#!/usr/bin/env python3
import io
import os
import sys
import socket
import wave
import time
import tempfile
import subprocess
import numpy as np
import sounddevice as sd
import json
import webrtcvad
import base64
from websocket import create_connection

try:
    from openai import OpenAI
except Exception as e:
    print("Missing openai package. Install requirements.txt", file=sys.stderr)
    raise


def record_until_enter(samplerate=16000, channels=1):
    print("Press Enter to START recording...")
    input()
    print("Recording... Press Enter to STOP.")

    frames = []
    stream = sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16')
    stream.start()
    try:
        while True:
            # read ~100 ms
            data, _ = stream.read(int(0.1 * samplerate))
            frames.append(data.copy())
            if sys.stdin in select([sys.stdin], [], [], 0)[0]:
                _ = sys.stdin.readline()
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop(); stream.close()

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, channels), dtype=np.int16)
    return audio


def load_persona_and_facts(persona_path: str = None, facts_path: str = None):
    system_msg = "You are the voice of a Unitree G1 robot. Be concise and helpful."
    facts_list = []
    if persona_path and os.path.exists(persona_path):
        try:
            with open(persona_path, 'r') as f:
                system_msg = f.read().strip()
        except Exception:
            pass
    if facts_path and os.path.exists(facts_path):
        try:
            with open(facts_path, 'r') as f:
                facts_list = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            pass
    tools_msg = "Known facts (user-provided):\n" + "\n".join(f"- {x}" for x in facts_list) if facts_list else ""
    return system_msg, tools_msg


class VADListener:
    def __init__(self, samplerate=16000, frame_ms=20, aggressiveness=2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.samplerate = samplerate
        self.frame_len = int(samplerate * frame_ms / 1000)
        self.stream = None

    def _frames(self):
        while True:
            data, _ = self.stream.read(self.frame_len)
            yield data.copy()

    def record_utterance(self, max_silence_ms=800, max_len_s=15):
        # returns int16 mono of one utterance, using VAD start/stop
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, dtype='int16')
        self.stream.start()
        frames = []
        silence_ms = 0
        start_time = time.time()
        try:
            for frame in self._frames():
                raw = frame.tobytes()
                is_speech = self.vad.is_speech(raw, self.samplerate)
                frames.append(frame)
                if not is_speech:
                    silence_ms += int(1000 * len(frame) / self.samplerate)
                else:
                    silence_ms = 0
                if silence_ms >= max_silence_ms:
                    break
                if time.time() - start_time > max_len_s:
                    break
        finally:
            self.stream.stop(); self.stream.close(); self.stream = None
        return np.concatenate(frames, axis=0) if frames else np.zeros((0,1), dtype=np.int16)


class RobotMicSource:
    def __init__(self, iface_ip: str, group_ip: str = '239.168.123.161', port: int = 5555):
        self.iface_ip = iface_ip
        self.group_ip = group_ip
        self.port = port
        self.sock = None

    def open(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', self.port))
        mreq = socket.inet_aton(self.group_ip) + socket.inet_aton(self.iface_ip)
        s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        s.settimeout(0.1)
        self.sock = s

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def record_until_enter(self):
        if self.sock is None:
            self.open()
        print("Recording from robot mic... Press Enter to STOP.")
        bufs = []
        try:
            while True:
                try:
                    data = self.sock.recv(2048)
                    if data:
                        bufs.append(data)
                except socket.timeout:
                    pass
                if sys.stdin in select([sys.stdin], [], [], 0)[0]:
                    _ = sys.stdin.readline()
                    break
        except KeyboardInterrupt:
            pass
        pcm = b"".join(bufs)
        return np.frombuffer(pcm, dtype=np.int16).reshape(-1, 1)


class RobotVADListener:
    def __init__(self, iface_ip: str, samplerate=16000, frame_ms=20, aggressiveness=2,
                 group_ip: str = '239.168.123.161', port: int = 5555):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.samplerate = samplerate
        self.frame_bytes = int(samplerate * frame_ms / 1000) * 2  # 16-bit mono
        self.src = RobotMicSource(iface_ip, group_ip, port)
        self.src.open()

    def record_utterance(self, max_silence_ms=800, max_len_s=15):
        buf = bytearray()
        frames = []
        silence_ms = 0
        start = time.time()
        try:
            while True:
                try:
                    data = self.src.sock.recv(2048)
                    if data:
                        buf.extend(data)
                        while len(buf) >= self.frame_bytes:
                            chunk = bytes(buf[:self.frame_bytes])
                            del buf[:self.frame_bytes]
                            is_speech = self.vad.is_speech(chunk, self.samplerate)
                            frames.append(np.frombuffer(chunk, dtype=np.int16))
                            if is_speech:
                                silence_ms = 0
                            else:
                                silence_ms += int(1000 * (self.frame_bytes/2) / self.samplerate)
                            if silence_ms >= max_silence_ms:
                                raise StopIteration
                            if time.time() - start > max_len_s:
                                raise StopIteration
                except socket.timeout:
                    pass
        except StopIteration:
            pass
        if not frames:
            return np.zeros((0,1), dtype=np.int16)
        pcm = np.concatenate(frames).astype(np.int16)
        return pcm.reshape(-1,1)


def select(read_list, write_list, except_list, timeout):
    import select as _select
    return _select.select(read_list, write_list, except_list, timeout)


def wav_bytes_from_int16(audio_int16, samplerate=16000):
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(1 if audio_int16.ndim == 1 else audio_int16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_int16.tobytes())
    bio.seek(0)
    return bio.read()


def read_wav_bytes(data: bytes):
    with wave.open(io.BytesIO(data), 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError("Expected 16-bit WAV from TTS")
    pcm = np.frombuffer(frames, dtype=np.int16)
    if ch == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return pcm, sr


def resample_int16(pcm: np.ndarray, src_sr: int, dst_sr: int):
    if src_sr == dst_sr:
        return pcm
    x = pcm.astype(np.float32)
    duration = x.shape[0] / src_sr
    new_len = int(round(duration * dst_sr))
    if new_len <= 1:
        return np.zeros((0,), dtype=np.int16)
    t_old = np.linspace(0, duration, num=x.shape[0], endpoint=False)
    t_new = np.linspace(0, duration, num=new_len, endpoint=False)
    y = np.interp(t_new, t_old, x)
    y = np.clip(y, -32768, 32767).astype(np.int16)
    return y


def stream_pcm_to_bridge(pcm_int16: np.ndarray, host='127.0.0.1', port=5002):
    s = socket.socket()
    s.connect((host, port))
    # send in ~1s chunks (32000 bytes)
    chunk_bytes = 32000
    b = pcm_int16.tobytes()
    for i in range(0, len(b), chunk_bytes):
        s.sendall(b[i:i+chunk_bytes])
        time.sleep(1.0)
    s.close()


class PCMStreamSender:
    def __init__(self, host='127.0.0.1', port=5002):
        self.host = host
        self.port = port
        self.sock = None

    def __enter__(self):
        self.sock = socket.socket()
        self.sock.connect((self.host, self.port))
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None

    def send(self, b: bytes):
        if not b or not self.sock:
            return
        self.sock.sendall(b)


def ensure_pcm16k_from_audio_bytes(b: bytes) -> np.ndarray:
    # If WAV, parse directly
    if len(b) >= 4 and b[:4] == b'RIFF':
        pcm, sr = read_wav_bytes(b)
        return resample_int16(pcm, sr, 16000)
    # Otherwise, try ffmpeg to convert unknown format to s16le 16k mono
    try:
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fi:
            fi.write(b)
            fi.flush()
            in_path = fi.name
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
            '-i', in_path,
            '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1'
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True)
        raw = proc.stdout
        os.unlink(in_path)
        return np.frombuffer(raw, dtype=np.int16)
    except Exception as e:
        raise RuntimeError(f"TTS bytes are not WAV and ffmpeg failed: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge-host', default='127.0.0.1')
    parser.add_argument('--bridge-port', type=int, default=5002)
    parser.add_argument('--mic-rate', type=int, default=16000)
    parser.add_argument('--model-chat', default='gpt-4o-mini')
    parser.add_argument('--model-asr', default='whisper-1')
    parser.add_argument('--model-tts', default='gpt-4o-mini-tts')
    parser.add_argument('--model-realtime', default='gpt-4o-realtime-preview-2024-12-17')
    parser.add_argument('--tts-voice', default='alloy')
    parser.add_argument('--persona', help='Path to a text file describing personality', default=None)
    parser.add_argument('--facts', help='Path to a text file with one fact per line', default=None)
    parser.add_argument('--continuous', action='store_true', help='Enable continuous VAD mode (Whisper)')
    parser.add_argument('--continuous-realtime', action='store_true', help='Enable continuous mode via Agents SDK/WebRTC (ASR+LLM). TTS remains local.')
    parser.add_argument('--agents-token-url', default='http://127.0.0.1:3456/token', help='Token endpoint from agents_service')
    parser.add_argument('--vad-aggr', type=int, default=2, help='VAD aggressiveness 0-3')
    parser.add_argument('--robot-mic-iface', default=None, help='Local NIC to join robot mic multicast. Accepts IPv4 (e.g., 192.168.123.51) or interface name (e.g., enp62s0). If unset, auto-detect 192.168.123.x; otherwise uses laptop mic.')
    args = parser.parse_args()

    if not os.environ.get('OPENAI_API_KEY'):
        print('Please set OPENAI_API_KEY', file=sys.stderr)
        sys.exit(1)

    client = OpenAI()

    system_msg, tools_msg = load_persona_and_facts(args.persona, args.facts)
    print("Ready. Modes: PTT (Enter/Enter), --continuous (Whisper), or --continuous-realtime (Realtime).")
    if not (args.continuous or args.continuous_realtime):
        print("Press Enter to start; Enter to stop recording; Ctrl+C to exit")

    history = [{"role":"system","content":system_msg + ("\n\n" + tools_msg if tools_msg else "")}]

    def _parse_ip_from_ip_cmd(line: str):
        # sample: '2: enp62s0    inet 192.168.123.51/24 brd 192.168.123.255 ...'
        parts = line.strip().split()
        for i, tok in enumerate(parts):
            if tok == 'inet' and i+1 < len(parts):
                return parts[i+1].split('/')[0]
        return None

    def _resolve_iface_to_ip(arg_iface: str):
        # If arg is dotted IPv4, return it
        import re
        if arg_iface and re.match(r'^\d+\.\d+\.\d+\.\d+$', arg_iface):
            return arg_iface
        # If arg is iface name, query ip
        if arg_iface:
            try:
                out = subprocess.check_output(['ip', '-4', '-o', 'addr', 'show', arg_iface], text=True)
                for line in out.splitlines():
                    ip = _parse_ip_from_ip_cmd(line)
                    if ip:
                        return ip
            except Exception:
                pass
        # Auto-detect 192.168.123.x
        try:
            out = subprocess.check_output(['ip', '-4', '-o', 'addr', 'show'], text=True)
            for line in out.splitlines():
                ip = _parse_ip_from_ip_cmd(line)
                if ip and ip.startswith('192.168.123.'):
                    return ip
        except Exception:
            pass
        return None

    robot_src = None
    if args.robot_mic_iface:
        ip_for_mcast = _resolve_iface_to_ip(args.robot_mic_iface)
        if not ip_for_mcast:
            print('Robot mic: failed to resolve interface/IP; falling back to laptop mic')
            vad_listener = VADListener(samplerate=args.mic_rate, aggressiveness=args.vad_aggr)
        else:
            print(f'Robot mic: using {ip_for_mcast} for multicast join')
            vad_listener = RobotVADListener(ip_for_mcast, samplerate=args.mic_rate, aggressiveness=args.vad_aggr)
            robot_src = RobotMicSource(ip_for_mcast)
    else:
        vad_listener = VADListener(samplerate=args.mic_rate, aggressiveness=args.vad_aggr)
    def agents_webrtc_loop(gen_bytes_iterable, on_text_complete):
        # This Python client will not directly run WebRTC; we rely on the browser Agents SDK for media I/O.
        # Here we simply print a hint and keep the bridge active. Optional: serve a minimal page later.
        print("[agents] Use the browser client with the token server at --agents-token-url for continuous mode.")
        print(f"[agents] Token endpoint: {args.agents_token_url}")
        print("[agents] Keep this bridge running to play audio.")

    # Fallback to avoid NameError when --continuous-realtime is used.
    def realtime_stream_loop(gen_bytes_iterable, on_text_complete):
        return agents_webrtc_loop(gen_bytes_iterable, on_text_complete)

    while True:
        try:
            if args.continuous_realtime:
                # Realtime continuous: stream audio continuously; server VAD & responses
                def gen_bytes():
                    # Prefer robot mic if available; else laptop mic stream
                    if isinstance(vad_listener, RobotVADListener):
                        # pull from UDP mic socket in small chunks
                        while True:
                            try:
                                data = vad_listener.src.sock.recv(2048)
                                yield data
                            except socket.timeout:
                                yield b""
                    else:
                        with sd.InputStream(samplerate=args.mic_rate, channels=1, dtype='int16') as st:
                            while True:
                                data, _ = st.read(int(0.02 * args.mic_rate))  # ~20ms
                                yield data.tobytes()

                def on_text(reply_text: str):
                    print(f"Bot: {reply_text}")
                    # TTS locally
                    speech = client.audio.speech.create(
                        model=args.model_tts,
                        voice=args.tts_voice,
                        input=reply_text
                    )
                    if hasattr(speech, 'content') and isinstance(speech.content, (bytes, bytearray)):
                        audio_bytes = speech.content
                    elif isinstance(speech, (bytes, bytearray)):
                        audio_bytes = speech
                    elif hasattr(speech, 'to_bytes'):
                        audio_bytes = speech.to_bytes()
                    else:
                        audio_bytes = bytes(speech)
                    pcm16k = ensure_pcm16k_from_audio_bytes(audio_bytes)
                    stream_pcm_to_bridge(pcm16k, host=args.bridge_host, port=args.bridge_port)

                realtime_stream_loop(gen_bytes, on_text)
                continue
            elif args.continuous:
                audio = vad_listener.record_utterance()
                if audio.size == 0:
                    continue
            else:
                if robot_src is not None:
                    print("Press Enter to START recording from robot mic...")
                    input()
                    audio = robot_src.record_until_enter()
                else:
                    audio = record_until_enter(samplerate=args.mic_rate, channels=1)
            if audio.size == 0:
                print("No audio captured.")
                continue

            wav_bytes = wav_bytes_from_int16(audio, samplerate=args.mic_rate)
            # ASR
            tr = client.audio.transcriptions.create(
                model=args.model_asr,
                file=("audio.wav", wav_bytes)
            )
            user_text = tr.text if hasattr(tr, 'text') else str(tr)
            print(f"You: {user_text}")
            history.append({"role":"user","content":user_text})

            # Chat
            chat = client.chat.completions.create(
                model=args.model_chat,
                messages=history,
                temperature=0.6
            )
            reply = chat.choices[0].message.content
            print(f"Bot: {reply}")
            history.append({"role":"assistant","content":reply})

            # TTS -> audio bytes (may be WAV or another container)
            speech = client.audio.speech.create(
                model=args.model_tts,
                voice=args.tts_voice,
                input=reply
            )
            # Extract WAV bytes from SDK response
            if hasattr(speech, 'content') and isinstance(speech.content, (bytes, bytearray)):
                audio_bytes = speech.content
            elif isinstance(speech, (bytes, bytearray)):
                audio_bytes = speech
            elif hasattr(speech, 'to_bytes'):
                audio_bytes = speech.to_bytes()
            else:
                audio_bytes = bytes(speech)

            pcm16k = ensure_pcm16k_from_audio_bytes(audio_bytes)
            stream_pcm_to_bridge(pcm16k, host=args.bridge_host, port=args.bridge_port)

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()


