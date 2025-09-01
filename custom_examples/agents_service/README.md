Agents SDK continuous mode

1) Start token server (Node):
```bash
cd /home/pim/Desktop/unitree_sdk2/custom_examples/agents_service
npm install
OPENAI_API_KEY=... npm start
```

2) Open browser client (use your favorite static host). Minimal snippet:
```html
<!doctype html>
<html>
  <body>
    <script type="module">
      import { RealtimeAgent, RealtimeSession } from "https://cdn.jsdelivr.net/npm/@openai/agents-realtime/+esm";
      async function main(){
        const tok = await fetch("http://127.0.0.1:3456/token").then(r=>r.json());
        const agent = new RealtimeAgent({ name: "G1", instructions: "Be concise and helpful." });
        const session = new RealtimeSession(agent, { model: 'gpt-realtime' });
        await session.connect({ apiKey: tok.client_secret.value });
        // Auto mic & speaker via WebRTC; on text, send to robot via local endpoint if you add one
        session.on('text', (text)=>console.log('Bot:', text));
      }
      main();
    </script>
  </body>
 </html>
```

3) Keep the C++ speaker bridge running on the dev host:
```bash
cd /home/pim/Desktop/unitree_sdk2/build
./bin/g1_audio_play_wav enp62s0
```

Notes:
- For streaming model audio to robot, either request server audio output and relay to TCP bridge, or keep text output and synthesize locally (current Python client).
- This setup uses browser Realtime Agents for continuous VAD/interruptions.

