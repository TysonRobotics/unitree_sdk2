const express = require('express');
const fetch = require('node-fetch');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname + '/public'));

app.get('/token', async (req, res) => {
  try {
    const body = {
      session: {
        type: 'realtime',
        model: process.env.REALTIME_MODEL || 'gpt-realtime'
      }
    };
    const resp = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });
    const data = await resp.json();
    if (!resp.ok) {
      console.error('Token error:', data);
      return res.status(500).json({ error: 'Failed to mint token', detail: data });
    }
    res.json(data);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Exception minting token' });
  }
});

const port = process.env.PORT || 3456;
const server = app.listen(port, () => console.log(`Agents token server listening on :${port} (open http://127.0.0.1:${port}/ )`));

// Optional: WS endpoint for server-side Realtime bridging
const WebSocket = require('ws');
const wss = new WebSocket.Server({ server, path: '/realtime' });
wss.on('connection', (ws) => {
  ws.send(JSON.stringify({ type: 'hello', msg: 'WS up' }));
});


// Server-side: Robot mic (UDP multicast) -> OpenAI Realtime -> local TCP speaker bridge
if (process.env.ENABLE_SERVER_REALTIME === '1') {
  const dgram = require('dgram');
  const os = require('os');
  const net = require('net');
  const { spawn } = require('child_process');

  const GROUP_IP = process.env.ROBOT_MIC_GROUP || '239.168.123.161';
  const GROUP_PORT = Number(process.env.ROBOT_MIC_PORT || 5555);
  const BRIDGE_HOST = process.env.BRIDGE_HOST || '127.0.0.1';
  const BRIDGE_PORT = Number(process.env.BRIDGE_PORT || 5002);
  const REALTIME_MODEL = process.env.REALTIME_MODEL || 'gpt-realtime';
  const TTS_MODEL = process.env.TTS_MODEL || 'gpt-4o-mini-tts';
  const TTS_VOICE = process.env.TTS_VOICE || 'alloy';

  function findIfaceIp() {
    if (process.env.ROBOT_IFACE_IP) return process.env.ROBOT_IFACE_IP;
    const ifs = os.networkInterfaces();
    for (const name of Object.keys(ifs)) {
      for (const addr of ifs[name]) {
        if (addr.family === 'IPv4' && !addr.internal && addr.address.startsWith('192.168.123.')) {
          return addr.address;
        }
      }
    }
    return null;
  }

  function pcmToBase64(buf) {
    return Buffer.from(buf).toString('base64');
  }

  function streamPcmToBridge(pcm) {
    return new Promise((resolve, reject) => {
      const sock = new net.Socket();
      sock.connect(BRIDGE_PORT, BRIDGE_HOST, () => {
        const chunkBytes = 32000; // ~1s at 16kHz s16 mono
        let offset = 0;
        function sendChunk() {
          if (offset >= pcm.length) { sock.end(); return; }
          const end = Math.min(offset + chunkBytes, pcm.length);
          const slice = pcm.slice(offset, end);
          sock.write(slice, () => {
            offset = end;
            setTimeout(sendChunk, 1000);
          });
        }
        sendChunk();
      });
      sock.on('error', reject);
      sock.on('close', resolve);
    });
  }

  async function ttsToPcm16k(text) {
    // Call OpenAI TTS, then ffmpeg to s16le 16k mono
    const url = 'https://api.openai.com/v1/audio/speech';
    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ model: TTS_MODEL, voice: TTS_VOICE, input: text })
    });
    if (!resp.ok) {
      const err = await resp.text();
      throw new Error(`TTS failed: ${resp.status} ${err}`);
    }
    const audioBuf = Buffer.from(await resp.arrayBuffer());
    return new Promise((resolve, reject) => {
      const ff = spawn('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 's16le', 'pipe:1']);
      const out = [];
      ff.stdout.on('data', (d) => out.push(d));
      ff.on('error', reject);
      ff.on('close', (code) => {
        if (code === 0) return resolve(Buffer.concat(out));
        reject(new Error('ffmpeg exit ' + code));
      });
      ff.stdin.write(audioBuf);
      ff.stdin.end();
    });
  }

  function startRealtime() {
    const ifaceIp = findIfaceIp();
    if (!ifaceIp) {
      console.error('[server-realtime] No 192.168.123.x iface found. Set ROBOT_IFACE_IP.');
      return;
    }
    console.log('[server-realtime] Using interface IP', ifaceIp);

    const udp = dgram.createSocket({ type: 'udp4', reuseAddr: true });
    udp.bind(GROUP_PORT, () => {
      try { udp.addMembership(GROUP_IP, ifaceIp); } catch (e) { console.error('multicast join error', e); }
      console.log('[server-realtime] Joined', GROUP_IP, 'on', ifaceIp);
    });

    const openai = new WebSocket(`wss://api.openai.com/v1/realtime?model=${encodeURIComponent(REALTIME_MODEL)}`, {
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'OpenAI-Beta': 'realtime=v1'
      }
    });

    let pendingText = '';
    let wsReady = false;
    let responseActive = false;
    openai.on('open', () => {
      // Configure session per docs: nest audio config and rely on server VAD (no manual commits)
      try {
        openai.send(JSON.stringify({
          type: 'session.update',
          session: {
            type: 'realtime',
            model: REALTIME_MODEL,
            audio: {
              input: {
                format: 'pcm16',
                turn_detection: { type: 'semantic_vad', create_response: true }
              }
            }
          }
        }));
        wsReady = true;
      } catch (e) {
        console.error('[server-realtime] session.update send error', e);
      }
      console.log('[server-realtime] WS connected');
    });

    openai.on('message', async (data) => {
      try {
        const msg = JSON.parse(data.toString());
        // Log unexpected types for debugging
        if (!msg.type) {
          return;
        }
        if (msg.type === 'response.created' || msg.type === 'response.in_progress') {
          responseActive = true;
        }
        if (msg.type === 'response.text.delta' && msg.delta) {
          pendingText += msg.delta;
        } else if (msg.type === 'response.output_text.delta' && msg.delta) {
          pendingText += msg.delta;
        }
        if (msg.type === 'response.completed') {
          const text = pendingText.trim();
          pendingText = '';
          responseActive = false;
          if (text) {
            try {
              const pcm = await ttsToPcm16k(text);
              await streamPcmToBridge(pcm);
            } catch (e) {
              console.error('[server-realtime] TTS/bridge error', e);
            }
          }
        }
        if (msg.type === 'error' || msg.error) {
          console.error('[server-realtime] Realtime error', msg);
        }
      } catch (e) {
        // Not JSON event; ignore
      }
    });

    // Stream UDP PCM to Realtime; rely on server VAD to auto-respond
    let lastChunkTs = Date.now();
    let lastVoiceTs = 0;
    let hadSpeechThisTurn = false;
    let bufferOpen = false;
    const MIN_APPEND_BYTES = 16000 * 2 * 0.2; // batch ~200ms before append
    let batch = [];
    let batchBytes = 0;
    function appendAndMaybeCommit(buf) {
      if (!wsReady) return; // wait until WS is open
      if (buf && buf.length) {
        // simple RMS energy to detect speech presence
        let sumAbs = 0;
        for (let i = 0; i + 1 < buf.length; i += 2) {
          const s = buf.readInt16LE(i);
          sumAbs += Math.abs(s);
        }
        const samples = Math.max(1, Math.floor(buf.length / 2));
        const avgAbs = sumAbs / samples;
        const isVoice = avgAbs > 500; // heuristic threshold

        if (isVoice) {
          lastVoiceTs = Date.now();
          hadSpeechThisTurn = true;
        }

        // batch small UDP frames into ~200ms chunks before append
        batch.push(buf);
        batchBytes += buf.length;
        if (batchBytes >= MIN_APPEND_BYTES) {
          const b = Buffer.concat(batch, batchBytes);
          batch = [];
          batchBytes = 0;
          try {
            bufferOpen = true;
            openai.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: pcmToBase64(b) }));
          } catch (e) {
            console.error('[server-realtime] append send error', e);
            batch = [];
            batchBytes = 0;
            return;
          }
        }
        lastChunkTs = Date.now();
      }
      const now = Date.now();
      const idle = now - lastChunkTs;
      const silence = now - lastVoiceTs;
      // No manual commit/response.create when VAD is enabled; server will auto-respond
    }

    udp.on('message', (msg) => {
      appendAndMaybeCommit(msg);
    });

    udp.on('error', (e) => console.error('[server-realtime] UDP error', e));
    openai.on('close', () => { wsReady = false; console.log('[server-realtime] WS closed'); });
    openai.on('error', (e) => { wsReady = false; console.error('[server-realtime] WS error', e); });
  }

  startRealtime();
}


