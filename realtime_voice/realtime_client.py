import asyncio
import base64
import json
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional, List
import errno

import numpy as np
import datetime
import threading
import time

import termios
import tty

try:
	import sounddevice as sd
except Exception as exc:  # pragma: no cover
	sd = None
	print("sounddevice import failed. Install system PortAudio (sudo apt install libportaudio2) and reinstall sounddevice.", file=sys.stderr)
	print(str(exc), file=sys.stderr)

import websockets

try:
	import soundfile as sf
except Exception as exc:  # pragma: no cover
	sf = None
if sf is None:
	print("soundfile not available; recording/playback WAV disabled until installed (pip install soundfile).", file=sys.stderr)


OPENAI_REALTIME_URL = os.environ.get(
	"OPENAI_REALTIME_URL",
	"wss://api.openai.com/v1/realtime?model=gpt-realtime",
)
DEBUG = os.environ.get("DEBUG", "0") == "1"
SERVER_SAMPLE_RATE_HZ = int(os.environ.get("SERVER_SAMPLE_RATE_HZ", "24000"))
METER_EVERY = int(os.environ.get("METER_EVERY", "500"))  # debug prints every N callbacks


class AudioDevices:
	"""Helpers to locate input/output devices by name substrings."""

	def __init__(self) -> None:
		if sd is None:
			raise RuntimeError("sounddevice not available")
		self.devices = sd.query_devices()

	def _find_by_name_substring(self, substrings: List[str], require_input: bool = False, require_output: bool = False) -> Optional[int]:
		for idx, dev in enumerate(self.devices):
			if require_input and dev.get("max_input_channels", 0) <= 0:
				continue
			if require_output and dev.get("max_output_channels", 0) <= 0:
				continue
			name = (dev.get("name") or "").lower()
			for s in substrings:
				if s and s.lower() in name:
					return idx
		return None

	def find_input_device_index(self, preferred_names: List[str]) -> Optional[int]:
		# Env override first (e.g., ec_source or pulse)
		override = os.environ.get("PREFERRED_INPUT", "").strip()
		idx = None
		if override:
			idx = self._find_by_name_substring([override], require_input=True)
			if idx is not None:
				return idx
		# Pulse echo-cancel source if present
		idx = self._find_by_name_substring(["ec_source", "echo-cancel", "pulse"], require_input=True)
		if idx is not None:
			return idx
		for idx, dev in enumerate(self.devices):
			if dev.get("max_input_channels", 0) <= 0:
				continue
			name = (dev.get("name") or "").lower()
			for p in preferred_names:
				if p in name:
					return idx
		return sd.default.device[0]

	def find_output_device_index(self, preferred_names: List[str]) -> Optional[int]:
		# Env override first (e.g., ec_sink or pulse)
		override = os.environ.get("PREFERRED_OUTPUT", "").strip()
		idx = None
		if override:
			idx = self._find_by_name_substring([override], require_output=True)
			if idx is not None:
				return idx
		# Pulse echo-cancel sink if present
		idx = self._find_by_name_substring(["ec_sink", "echo-cancel", "pulse"], require_output=True)
		if idx is not None:
			return idx
		for idx, dev in enumerate(self.devices):
			if dev.get("max_output_channels", 0) <= 0:
				continue
			name = (dev.get("name") or "").lower()
			for p in preferred_names:
				if p in name:
					return idx
		return sd.default.device[1]


@asynccontextmanager
async def connect_realtime(api_key: str):
	headers = {
		"Authorization": f"Bearer {api_key}",
		"OpenAI-Beta": "realtime=v1",
	}
	if DEBUG:
		print(f"[DEBUG] Connecting to {OPENAI_REALTIME_URL}")
	async with websockets.connect(OPENAI_REALTIME_URL, extra_headers=headers, max_size=None) as ws:
		if DEBUG:
			print("[DEBUG] WebSocket connected")
		yield ws


def pcm_int16_bytes_from_float(audio_block: np.ndarray) -> bytes:
	audio_block = np.clip(audio_block, -1.0, 1.0)
	int16 = (audio_block * 32767.0).astype(np.int16)
	return int16.tobytes()


def float_from_pcm_int16_bytes(pcm_bytes: bytes) -> np.ndarray:
	int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
	return (int16.astype(np.float32) / 32767.0).reshape(-1, 1)


def resample_mono_float(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
	if src_rate == dst_rate or len(audio) == 0:
		return audio
	ratio = float(dst_rate) / float(src_rate)
	src_indices = np.arange(audio.shape[0], dtype=np.float32)
	dst_length = int(round(audio.shape[0] * ratio))
	dst_indices = np.linspace(0, max(len(audio) - 1, 1), dst_length, dtype=np.float32)
	resampled = np.interp(dst_indices, src_indices, audio[:, 0]).astype(np.float32)
	return resampled.reshape(-1, 1)


def resample_int16_bytes(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
	floats = float_from_pcm_int16_bytes(pcm_bytes)
	resampled = resample_mono_float(floats, src_rate, dst_rate)
	return pcm_int16_bytes_from_float(resampled)


class RealtimeVoiceClient:
	def __init__(
		self,
		api_key: str,
		input_device_index: Optional[int],
		output_device_index: Optional[int],
		sample_rate_hz: int = 24000,
		block_size: int = 1024,
		voice: str = "marin",
	) -> None:
		self.api_key = api_key
		self.input_device_index = input_device_index
		self.output_device_index = output_device_index
		self.sample_rate_hz = sample_rate_hz
		self.block_size = block_size
		self.voice = voice

		self._ws = None
		self._in_queue: Optional[asyncio.Queue] = None
		self._stop: Optional[asyncio.Event] = None
		self._loop: Optional[asyncio.AbstractEventLoop] = None
		self._output_channels: int = 1
		self._play_buffer = bytearray()
		self._play_lock = threading.Lock()
		self._assistant_active: bool = False
		self._playback_tail_until_ms: float = 0.0
		self._rms_gate: float = float(os.environ.get("INPUT_RMS_GATE", "0.010"))
		self._hangover_ms: float = float(os.environ.get("HANGOVER_MS", "200"))
		self._vad_speaking: bool = False
		self._vad_silence_ms: float = float(os.environ.get("VAD_SILENCE_MS", "900"))
		self._silence_accum_ms: float = 0.0
		self._awaiting_response: bool = False
		# FIFO-based control channel (for external web UI)
		self._control_fifo_path: str = os.environ.get(
			"VOICE_CONTROL_FIFO",
			os.path.join(os.path.dirname(__file__), "control.fifo"),
		)
		self._fifo_fd: Optional[int] = None
		self._fifo_wfd: Optional[int] = None  # keep a writer open to avoid EOF when no external writers
		self._fifo_buf: bytes = b""
		# Hysteresis and minimum speech duration
		self._rms_start_gate: float = float(os.environ.get("VAD_START_GATE", str(max(self._rms_gate * 1.6, self._rms_gate + 0.004))))
		self._rms_stop_gate: float = float(os.environ.get("VAD_STOP_GATE", str(self._rms_gate)))
		self._min_speech_ms: float = float(os.environ.get("VAD_MIN_SPEECH_MS", "1200"))
		self._speech_accum_ms: float = 0.0
		# Barge-in (interruptible) control
		self._barge_in: bool = os.environ.get("BARGE_IN", "0") == "1"
		self._last_cancel_ms: float = 0.0
		self._cancel_cooldown_ms: float = float(os.environ.get("BARGE_CANCEL_COOLDOWN_MS", "250"))
		# Interactive mic mute toggle
		self._muted: bool = False
		# Recording state
		self._recording: bool = False
		self._record_buffer: List[np.ndarray] = []
		self._record_start_time_ms: float = 0.0
		self._record_dir: str = os.environ.get(
			"RECORDINGS_DIR",
			os.path.join(os.path.dirname(__file__), "recordings"),
		)
		self._record_path: Optional[str] = None
		# Config paths for persona and facts
		self._persona_path: str = os.environ.get(
			"PERSONA_PATH",
			os.path.join(os.path.dirname(__file__), "config", "persona.json"),
		)
		self._facts_path: str = os.environ.get(
			"FACTS_PATH",
			os.path.join(os.path.dirname(__file__), "config", "facts.json"),
		)
		# Output volume percent (0-100). Default 100. Supports OUTPUT_VOLUME_PCT; fallback to OUTPUT_VOLUME (0.0-2.0 scale).
		try:
			if "OUTPUT_VOLUME_PCT" in os.environ:
				self._volume_pct: int = int(float(os.environ.get("OUTPUT_VOLUME_PCT", "100")))
				self._volume_pct = max(0, min(100, self._volume_pct))
			else:
				legacy = float(os.environ.get("OUTPUT_VOLUME", "1.0"))
				self._volume_pct = int(max(0.0, min(2.0, legacy)) * 100)
		except Exception:
			self._volume_pct = 100

		# Ensure recordings directory exists
		try:
			os.makedirs(self._record_dir, exist_ok=True)
		except Exception:
			pass

	def _load_persona(self) -> dict:
		try:
			with open(self._persona_path, "r", encoding="utf-8") as f:
				return json.load(f) or {}
		except Exception:
			return {}

	def _load_facts(self) -> list:
		try:
			with open(self._facts_path, "r", encoding="utf-8") as f:
				data = json.load(f)
				if isinstance(data, list):
					return [str(x) for x in data]
				return []
		except Exception:
			return []

	def _build_instructions(self) -> str:
		persona = self._load_persona()
		facts = self._load_facts()
		intro = persona.get("intro", "") if isinstance(persona, dict) else ""
		style = persona.get("style", "") if isinstance(persona, dict) else ""
		behaviors = persona.get("behaviors", []) if isinstance(persona, dict) else []
		lines = []
		if intro:
			lines.append(f"Introduction: {intro}")
		if style:
			lines.append(f"Speaking style: {style}")
		if behaviors:
			lines.append("Behavior guidelines:")
			for b in behaviors:
				lines.append(f"- {b}")
		if facts:
			lines.append("Facts you should know and can rely on when asked:")
			for fact in facts:
				lines.append(f"- {fact}")
		return "\n".join(lines).strip()

	async def run(self) -> None:
		if sd is None:
			raise RuntimeError("sounddevice not available")

		# Capture loop and create per-loop primitives
		self._loop = asyncio.get_running_loop()
		self._in_queue = asyncio.Queue(maxsize=4096)
		self._stop = asyncio.Event()
		try:
			self._loop.add_signal_handler(signal.SIGINT, self._stop.set)
			self._loop.add_signal_handler(signal.SIGTERM, self._stop.set)
		except NotImplementedError:
			pass
		print("Press 'q' then Enter to quit; 'm' then Enter to mute/unmute mic; 's' then Enter to stop speaking; 'r' then Enter to reload; '+'/'-' then Enter to change volume (0-100 in steps of 10).")

		# Negotiate a working sample rate for both output and input
		negotiated_rate = self._negotiate_sample_rate()
		if negotiated_rate and negotiated_rate != self.sample_rate_hz:
			self.sample_rate_hz = negotiated_rate

		# Install FIFO control reader (for web UI commands)
		self._install_fifo_reader()

		# Determine output channels; prefer 2 if available
		try:
			devs = sd.query_devices()
			dev = devs[self.output_device_index] if self.output_device_index is not None else None
			max_out = int(dev.get("max_output_channels", 1)) if dev else 1
			self._output_channels = 2 if max_out >= 2 else 1
		except Exception:
			self._output_channels = 1

		if DEBUG:
			print(f"[DEBUG] Using device sample rate: {self.sample_rate_hz} Hz (server {SERVER_SAMPLE_RATE_HZ} Hz)")
			print(f"[DEBUG] Output channels: {self._output_channels}")

		# Log devices
		if DEBUG:
			devices = sd.query_devices()
			inp = devices[self.input_device_index] if self.input_device_index is not None else None
			outp = devices[self.output_device_index] if self.output_device_index is not None else None
			print(f"[DEBUG] Input device idx={self.input_device_index} name={inp['name'] if inp else None}")
			print(f"[DEBUG] Output device idx={self.output_device_index} name={outp['name'] if outp else None}")

		# Open raw output stream with callback-driven ring buffer playback
		bytes_per_sample = 2  # int16
		bytes_per_frame = bytes_per_sample * self._output_channels

		def on_audio_out(outdata, frames, time_info, status):  # type: ignore[unused-argument]
			needed = frames * bytes_per_frame
			consumed = 0
			with self._play_lock:
				buflen = len(self._play_buffer)
				if buflen >= needed:
					chunk = self._play_buffer[:needed]
					del self._play_buffer[:needed]
					consumed = needed
				else:
					chunk = bytes(buflen) + bytes(needed - buflen)
					consumed = buflen
					self._play_buffer.clear()
			# If we played any buffered audio, extend hangover window
			if consumed > 0:
				self._playback_tail_until_ms = time.monotonic() * 1000.0 + self._hangover_ms
			# Apply output volume scaling using percent
			if self._volume_pct != 100 and len(chunk) > 0:
				scale = float(self._volume_pct) / 100.0
				arr = np.frombuffer(chunk, dtype=np.int16).astype(np.int32)
				arr = np.clip((arr * scale), -32768, 32767).astype(np.int16)
				chunk = arr.tobytes()
			# Fill output bytes
			outdata[:] = chunk

		output_stream = sd.RawOutputStream(
			samplerate=self.sample_rate_hz,
			channels=self._output_channels,
			dtype="int16",
			blocksize=self.block_size,
			device=self.output_device_index,
			callback=on_audio_out,
		)
		output_stream.start()

		# Input stream callback enqueues PCM bytes
		def _try_put(data_bytes: bytes) -> None:
			try:
				assert self._in_queue is not None
				self._in_queue.put_nowait(data_bytes)
			except asyncio.QueueFull:
				pass

		def _schedule_json(obj: dict) -> None:
			if self._loop is not None:
				self._loop.call_soon_threadsafe(asyncio.create_task, self._send_json(obj))

		_meter_samples = 0
		def on_audio_in(data: np.ndarray, frames: int, time_info, status) -> None:  # type: ignore[override]
			if status and DEBUG:
				print(f"[DEBUG] Input status: {status}")
			try:
				# Append raw device-rate mono float32 frames to recording buffer if active
				if self._recording:
					# Copy to avoid referencing the underlying buffer
					self._record_buffer.append(np.array(data, dtype=np.float32))
				# Hard block mic while assistant speaking or during tail (always; ignore barge-in)
				now_ms = time.monotonic() * 1000.0
				if self._assistant_active or now_ms < self._playback_tail_until_ms:
					return
				# Mic mute toggle
				if self._muted:
					return
				# Compute RMS on device-rate audio
				lvl_dev = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
				frame_ms = 1000.0 * frames / float(self.sample_rate_hz)
				# Hysteresis: start/continue speaking on higher threshold; stop on lower
				if lvl_dev >= self._rms_start_gate:
					if not self._vad_speaking:
						self._speech_accum_ms = 0.0
					self._vad_speaking = True
					self._silence_accum_ms = 0.0
					self._speech_accum_ms += frame_ms
				elif lvl_dev <= self._rms_stop_gate:
					self._silence_accum_ms += frame_ms
					if self._vad_speaking and self._silence_accum_ms >= self._vad_silence_ms and not self._assistant_active and not self._awaiting_response:
						# Only end turn if we had enough speech
						if self._speech_accum_ms >= self._min_speech_ms:
							self._vad_speaking = False
							_schedule_json({"type": "input_audio_buffer.commit"})
							_schedule_json({
								"type": "response.create",
								"response": {"voice": self.voice}
							})
							self._awaiting_response = True
							self._speech_accum_ms = 0.0
							return
					else:
						# not enough speech; keep waiting or drop
						pass
				else:
					# Between thresholds: hold state; if speaking, accumulate speech
					if self._vad_speaking:
						self._speech_accum_ms += frame_ms
					self._silence_accum_ms = 0.0
				# Forward audio only if considered speaking
				if not self._vad_speaking:
					return
				if self.sample_rate_hz != SERVER_SAMPLE_RATE_HZ:
					data_rs = resample_mono_float(data, self.sample_rate_hz, SERVER_SAMPLE_RATE_HZ)
				else:
					data_rs = data
				pcm_bytes = pcm_int16_bytes_from_float(data_rs)
				if self._loop is not None:
					self._loop.call_soon_threadsafe(_try_put, pcm_bytes)
				if DEBUG:
					nonlocal _meter_samples
					_meter_samples += 1
					if _meter_samples % max(1, METER_EVERY) == 0:
						print(f"[DEBUG] Mic level ~ {lvl_dev:.3f}")
			except Exception as exc:
				if DEBUG:
					print(f"[DEBUG] Input callback error: {exc}")

		input_stream = sd.InputStream(
			samplerate=self.sample_rate_hz,
			channels=1,
			dtype="float32",
			blocksize=self.block_size,
			callback=on_audio_in,
			device=self.input_device_index,
		)
		input_stream.start()

		sender_task = None
		receiver_task = None
		stop_task = None
		stdin_task = None
		try:
			async with connect_realtime(self.api_key) as ws:
				self._ws = ws
				# Configure session: audio formats, voice, server VAD
				await self._send_json({
					"type": "session.update",
					"session": {
						"input_audio_format": "pcm16",
						"output_audio_format": "pcm16",
						"voice": self.voice,
						"turn_detection": {"type": "server_vad"},
						"instructions": self._build_instructions(),
					}
				})
				if DEBUG:
					print("[DEBUG] Sent session.update; starting continuous streaming")

				# Optional: prime an initial response (intro)
				if os.environ.get("SKIP_INTRO", "0") != "1":
					await self._send_json({
						"type": "response.create",
						"response": {"voice": self.voice}
					})
					if DEBUG:
						print("[DEBUG] Sent response.create")

				# Start tasks and wait for stop
				sender_task = asyncio.create_task(self._audio_sender())
				receiver_task = asyncio.create_task(self._receiver_loop(output_stream))
				stop_task = asyncio.create_task(self._stop.wait())
				stdin_task = asyncio.create_task(self._stdin_quit())
				await asyncio.wait([sender_task, receiver_task, stop_task, stdin_task], return_when=asyncio.FIRST_COMPLETED)
		finally:
			# Remove FIFO reader and close descriptors
			try:
				if self._loop is not None and self._fifo_fd is not None:
					self._loop.remove_reader(self._fifo_fd)
			except Exception:
				pass
			try:
				if self._fifo_fd is not None:
					os.close(self._fifo_fd)
			except Exception:
				pass
			try:
				if self._fifo_wfd is not None:
					os.close(self._fifo_wfd)
			except Exception:
				pass
			# Cancel tasks if alive
			for t in (sender_task, receiver_task, stop_task, stdin_task):
				if t and not t.done():
					t.cancel()
					try:
						await t
					except Exception:
						pass
			# Close WS
			try:
				if self._ws is not None:
					await self._ws.close()
			except Exception:
				pass
			# Close audio
			input_stream.stop()
			input_stream.close()
			output_stream.stop()
			output_stream.close()

	def _install_fifo_reader(self) -> None:
		"""Create and register a non-blocking FIFO reader for external commands."""
		try:
			if not os.path.exists(self._control_fifo_path):
				os.mkfifo(self._control_fifo_path)
			# Open read end non-blocking
			self._fifo_fd = os.open(self._control_fifo_path, os.O_RDONLY | os.O_NONBLOCK)
			# Open a dummy write end to keep FIFO from hitting EOF when no writers
			try:
				self._fifo_wfd = os.open(self._control_fifo_path, os.O_WRONLY | os.O_NONBLOCK)
			except Exception:
				self._fifo_wfd = None
			if self._loop is None:
				self._loop = asyncio.get_running_loop()
			self._loop.add_reader(self._fifo_fd, self._on_fifo_ready)
		except Exception as exc:
			if DEBUG:
				print(f"[DEBUG] FIFO setup failed: {exc}")

	def _on_fifo_ready(self) -> None:
		"""Handle readable FIFO: read complete newline-delimited JSON commands."""
		while True:
			try:
				chunk = os.read(self._fifo_fd, 4096) if self._fifo_fd is not None else b""
			except BlockingIOError:
				# No more data to read right now
				return
			except OSError as exc:
				if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
					return
				if DEBUG:
					print(f"[DEBUG] FIFO read error: {exc}")
				return
			if not chunk:
				# EOF or no data; stop for now
				return
			self._fifo_buf += chunk
			while b"\n" in self._fifo_buf:
				line, self._fifo_buf = self._fifo_buf.split(b"\n", 1)
				line_str = line.decode("utf-8", errors="ignore").strip()
				if not line_str:
					continue
				try:
					obj = json.loads(line_str)
					self._handle_control_command(obj)
				except Exception as exc:
					if DEBUG:
						print(f"[DEBUG] Bad control command: {line_str} ({exc})")

	def _handle_control_command(self, obj: dict) -> None:
		"""Apply control commands coming from web UI via FIFO."""
		cmd = str(obj.get("cmd", "")).lower()
		if not cmd:
			return
		if cmd == "mute":
			val = obj.get("value")
			if isinstance(val, bool):
				self._muted = val
				print(f"[INFO] Mic is now {'MUTED' if self._muted else 'UNMUTED'}")
		elif cmd == "toggle_mute":
			self._muted = not self._muted
			print(f"[INFO] Mic is now {'MUTED' if self._muted else 'UNMUTED'}")
		elif cmd == "stop":
			# Cancel assistant speech and clear buffer
			asyncio.create_task(self._send_json({"type": "response.cancel"}))
			with self._play_lock:
				self._play_buffer.clear()
			self._assistant_active = False
			self._awaiting_response = False
			self._playback_tail_until_ms = time.monotonic() * 1000.0 + self._hangover_ms
			print("[INFO] Stopped speaking.")
		elif cmd == "reload":
			print('[INFO] Reloading persona/facts and resetting conversation (full restart via web)...')
			try:
				asyncio.create_task(self._send_json({"type": "response.cancel"}))
			except Exception:
				pass
			with self._play_lock:
				self._play_buffer.clear()
			self._assistant_active = False
			self._awaiting_response = False
			try:
				os.execv(sys.executable, [sys.executable, __file__])
			except Exception as exc:
				print(f"[WARN] Hard reset failed: {exc}. Please restart the program.")
		elif cmd == "volume":
			try:
				val = int(obj.get("value"))
				self._volume_pct = max(0, min(100, val))
				print(f"[INFO] Volume: {self._volume_pct}%")
			except Exception:
				pass
		elif cmd == "volume_up":
			self._volume_pct = max(0, min(100, self._volume_pct + 10))
			print(f"[INFO] Volume: {self._volume_pct}%")
		elif cmd == "volume_down":
			self._volume_pct = max(0, min(100, self._volume_pct - 10))
			print(f"[INFO] Volume: {self._volume_pct}%")
		elif cmd == "barge_in":
			val = obj.get("value")
			if isinstance(val, bool):
				self._barge_in = val
				print(f"[INFO] Barge-in is now {'ENABLED' if self._barge_in else 'DISABLED'}")

		elif cmd == "record_start":
			if sf is None:
				print("[WARN] soundfile not installed; cannot start recording")
				return
			if self._recording:
				print("[INFO] Already recording")
				return
			name = str(obj.get("name") or "").strip()
			if not name:
				ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
				name = f"rec_{ts}.wav"
			if not name.lower().endswith(".wav"):
				name += ".wav"
			self._record_path = os.path.join(self._record_dir, os.path.basename(name))
			self._record_buffer = []
			self._record_start_time_ms = time.monotonic() * 1000.0
			self._recording = True
			print(f"[INFO] Recording started: {self._record_path}")
		elif cmd == "record_stop":
			if not self._recording:
				print("[INFO] Not recording")
				return
			self._recording = False
			try:
				self._save_recording()
			except Exception as exc:
				print(f"[WARN] Failed to save recording: {exc}")
			finally:
				self._record_buffer = []
				self._record_path = None
		elif cmd == "play":
			if sf is None:
				print("[WARN] soundfile not installed; cannot play WAV")
				return
			file_arg = obj.get("file") or ""
			if not file_arg:
				print("[WARN] Missing 'file' for play command")
				return
			path = file_arg
			if not os.path.isabs(path):
				path = os.path.join(self._record_dir, os.path.basename(path))
			if not os.path.exists(path):
				print(f"[WARN] File not found: {path}")
				return
			# Stop current speech and clear buffer, then queue file audio
			try:
				asyncio.create_task(self._send_json({"type": "response.cancel"}))
			except Exception:
				pass
			with self._play_lock:
				self._play_buffer.clear()
			self._assistant_active = False
			self._awaiting_response = False
			self._queue_wav_for_playback(path)
			print(f"[INFO] Playing: {os.path.basename(path)}")

	async def _stdin_quit(self) -> None:
		"""Interactive stdin: 'm' mute, 's' stop, 'r' reload, 'q' quit, '+'/'-' volume (0-100, step 10)."""
		loop = asyncio.get_running_loop()
		while self._stop is not None and not self._stop.is_set():
			line = await loop.run_in_executor(None, sys.stdin.readline)
			if not line:
				continue
			cmd = line.strip().lower()
			if cmd == 'q':
				self._stop.set()
				break
			if cmd in ('m', 'mute', 'unmute'):
				self._muted = (not self._muted) if cmd == 'm' else (cmd == 'mute')
				state = 'MUTED' if self._muted else 'UNMUTED'
				print(f"[INFO] Mic is now {state}")
			if cmd == 'r':
				print('[INFO] Reloading persona/facts and resetting conversation (full restart)...')
				try:
					# Stop any current speech
					try:
						await self._send_json({"type": "response.cancel"})
					except Exception:
						pass
					# Clear playback and flags
					with self._play_lock:
						self._play_buffer.clear()
					self._assistant_active = False
					self._awaiting_response = False
					# Exec a fresh process so server conversation/session is brand-new
					os.execv(sys.executable, [sys.executable, __file__])
				except Exception as exc:
					print(f"[WARN] Hard reset failed: {exc}. Please restart the program.")
			if cmd == 's':
				# Stop current assistant speech immediately
				try:
					await self._send_json({"type": "response.cancel"})
				except Exception:
					pass
				with self._play_lock:
					self._play_buffer.clear()
				self._assistant_active = False
				self._awaiting_response = False
				self._playback_tail_until_ms = time.monotonic() * 1000.0 + self._hangover_ms
				print('[INFO] Stopped speaking.')
			if cmd in ('+', 'up', 'louder'):
				self._volume_pct = max(0, min(100, self._volume_pct + 10))
				print(f"[INFO] Volume: {self._volume_pct}%")
			if cmd in ('-', 'down', 'softer'):
				self._volume_pct = max(0, min(100, self._volume_pct - 10))
				print(f"[INFO] Volume: {self._volume_pct}%")

	async def _send_json(self, obj: dict) -> None:
		if DEBUG:
			etype = obj.get("type")
			if etype != "input_audio_buffer.append":
				print(f"[DEBUG] -> {etype}")
		msg = json.dumps(obj)
		assert self._ws is not None
		# Avoid sending on closed socket
		if hasattr(self._ws, "closed") and self._ws.closed:
			return
		await self._ws.send(msg)

	async def _audio_sender(self) -> None:
		"""Continuously send input audio with server-side VAD enabled."""
		assert self._in_queue is not None
		count = 0
		while self._stop is not None and not self._stop.is_set():
			chunk = await self._in_queue.get()
			b64 = base64.b64encode(chunk).decode("ascii")
			try:
				await self._send_json({
					"type": "input_audio_buffer.append",
					"audio": b64,
				})
			except Exception:
				# Likely websocket closed; stop loop
				break
			count += 1
			if DEBUG and count % 200 == 0:
				print(f"[DEBUG] Sent {count} audio chunks")

	async def _receiver_loop(self, output_stream: sd.RawOutputStream) -> None:
		assert self._ws is not None
		async for message in self._ws:
			try:
				obj = json.loads(message)
			except Exception:
				continue

			event_type = obj.get("type")
			if DEBUG:
				print(f"[DEBUG] <- {event_type}")
			if event_type == "session.updated" and DEBUG:
				session = obj.get("session", {})
				ack_voice = session.get("voice")
				if ack_voice:
					print(f"[DEBUG] Server acknowledged voice: {ack_voice}")
			elif event_type == "response.output_text.delta":
				delta = obj.get("delta", "")
				if delta:
					print(delta, end="", flush=True)
			elif event_type == "response.output_text.done":
				print()
			elif event_type == "response.created":
				self._assistant_active = True
				self._awaiting_response = False
			elif event_type in ("response.audio.delta", "response.output_audio.delta"):
				audio_b64 = obj.get("delta") or obj.get("audio")
				if not audio_b64:
					continue
				pcm_bytes = base64.b64decode(audio_b64)
				if self.sample_rate_hz != SERVER_SAMPLE_RATE_HZ:
					pcm_bytes = resample_int16_bytes(pcm_bytes, SERVER_SAMPLE_RATE_HZ, self.sample_rate_hz)
				# Expand to stereo if needed
				if self._output_channels == 2:
					# Duplicate mono int16 samples to L/R
					mono = np.frombuffer(pcm_bytes, dtype=np.int16)
					stereo = np.repeat(mono, 2)
					pcm_bytes = stereo.tobytes()
				# Push into ring buffer for playback
				with self._play_lock:
					self._play_buffer.extend(pcm_bytes)
			elif event_type == "input_audio_buffer.committed":
				# Owned by local VAD; do not send response.create here
				pass
			elif event_type == "error":
				print(f"Realtime API error: {obj}")
			elif event_type == "response.done":
				self._assistant_active = False
				# Apply hangover before re-opening mic to avoid capturing tail audio
				self._playback_tail_until_ms = time.monotonic() * 1000.0 + self._hangover_ms

	def _save_recording(self) -> None:
		"""Write buffered recording to WAV file."""
		if sf is None:
			return
		if not self._record_path:
			return
		if not self._record_buffer:
			print("[INFO] Recording empty; nothing saved")
			return
		try:
			data = np.vstack(self._record_buffer).astype(np.float32)
			sf.write(self._record_path, data, self.sample_rate_hz, subtype='PCM_16')
			dur = data.shape[0] / float(self.sample_rate_hz)
			print(f"[INFO] Recording saved: {self._record_path} ({dur:.2f}s)")
		except Exception as exc:
			print(f"[WARN] Recording save failed: {exc}")

	def _queue_wav_for_playback(self, path: str) -> None:
		"""Read WAV file, resample/channels adapt, and append to playback buffer."""
		if sf is None:
			return
		try:
			data, rate = sf.read(path, dtype='int16', always_2d=False)
			# Ensure mono int16 ndarray
			if isinstance(data, tuple):
				# Some versions may return (data, samplerate); already handled
				pass
			arr = np.array(data, dtype=np.int16)
			if arr.ndim > 1:
				# Convert to mono by averaging channels
				arr = arr.mean(axis=1).astype(np.int16)
			pcm_bytes = arr.tobytes()
			if rate != self.sample_rate_hz:
				pcm_bytes = resample_int16_bytes(pcm_bytes, rate, self.sample_rate_hz)
			# Expand to stereo if output stream is stereo
			if self._output_channels == 2:
				mono = np.frombuffer(pcm_bytes, dtype=np.int16)
				stereo = np.repeat(mono, 2)
				pcm_bytes = stereo.tobytes()
			with self._play_lock:
				self._play_buffer.extend(pcm_bytes)
		except Exception as exc:
			print(f"[WARN] Failed to queue WAV for playback: {exc}")

	def _negotiate_sample_rate(self) -> int:
		"""Try common sample rates that both in/out devices accept. Returns chosen rate or existing."""
		common_rates = [48000, 44100, 32000, 24000, 22050, 16000]
		for rate in common_rates:
			out_ok = False
			in_ok = False
			try:
				s = sd.OutputStream(samplerate=rate, channels=1, dtype="float32", device=self.output_device_index)
				s.close()
				out_ok = True
			except Exception:
				out_ok = False
			try:
				s = sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=self.input_device_index)
				s.close()
				in_ok = True
			except Exception:
				in_ok = False
			if in_ok and out_ok:
				return rate
		try:
			return int(sd.default.samplerate) if sd.default.samplerate else self.sample_rate_hz
		except Exception:
			return self.sample_rate_hz


def main() -> int:
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		print("Please export OPENAI_API_KEY with your OpenAI API key.", file=sys.stderr)
		return 2

	if sd is None:
		return 3

	devices = AudioDevices()
	input_idx = devices.find_input_device_index(["gvaudio", "usb audio"])  # type: ignore[arg-type]
	output_idx = devices.find_output_device_index(["jbl", "go 4"])  # type: ignore[arg-type]

	client = RealtimeVoiceClient(
		api_key=api_key,
		input_device_index=input_idx,
		output_device_index=output_idx,
		sample_rate_hz=int(os.environ.get("AUDIO_SAMPLE_RATE", "24000")),
		block_size=int(os.environ.get("AUDIO_BLOCK_SIZE", "1024")),
		voice=os.environ.get("VOICE", "marin"),
	)

	try:
		asyncio.run(client.run())
	except (KeyboardInterrupt, asyncio.CancelledError):
		pass
	return 0


if __name__ == "__main__":
	raise SystemExit(main())