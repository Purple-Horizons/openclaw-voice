# OpenClaw Voice

**Open-source browser-based voice interface for AI assistants.**

Talk to your AI like you talk to Alexa — but self-hosted, private, and connected to your own agent.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)

🌐 **Website:** [openclawvoice.com](https://openclawvoice.com)

## Features

| Feature | Description |
|---------|-------------|
| 🎤 **Local STT** | Whisper runs locally via faster-whisper. Your voice never leaves your machine. |
| 🔊 **Streaming TTS** | ElevenLabs with sentence-by-sentence streaming. Hear responses while they generate. |
| 🎯 **Voice Activity Detection** | Silero VAD filters background noise. Works in noisy environments. |
| 🧹 **Smart Text Cleaning** | Strips markdown, hashtags, URLs before TTS. No more "hash hash". |
| 🔌 **Any AI Backend** | OpenAI, Claude, or full OpenClaw agent with memory and tools. |
| 🌐 **Browser-Based** | No app install. Works on desktop and mobile. |
| 🚗 **Continuous Mode** | Hands-free conversation. Auto-listens after each response. |
| ☁️ **Server VAD (Aliyun Realtime)** | Uses qwen3-asr-flash-realtime VAD events (`speech_started`/`speech_stopped`) to auto-end turns without local torch/silero. |

## Quick Start

```bash
# Clone
git clone https://github.com/Purple-Horizons/openclaw-voice.git
cd openclaw-voice

# Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: Install VAD support (better noise handling)
pip install torch torchaudio

# Configure (create .env from example)
cp .env.example .env
# Edit .env with your API keys

# Run
PYTHONPATH=. python -m src.server.main

# Open http://localhost:8765
```

## Smoke Test (STT + OpenClaw roundtrip)

```bash
python scripts/smoke_ws_roundtrip.py \
  --ws ws://127.0.0.1:8765/ws \
  --audio /path/to/test.ogg
```

Expected: `SMOKE PASS` with non-empty transcript and non-empty `response_complete`.

## For OpenClaw Agents (Auto-Install)

One-liner for AI agents to self-install:

```bash
cd /tmp && \
git clone https://github.com/Purple-Horizons/openclaw-voice.git && \
cd openclaw-voice && \
python3 -m venv .venv && source .venv/bin/activate && \
pip install -r requirements.txt torch torchaudio && \
PYTHONPATH=. ELEVENLABS_API_KEY="$ELEVENLABS_API_KEY" OPENAI_API_KEY="$OPENAI_API_KEY" \
  nohup python -m src.server.main > /tmp/voice-server.log 2>&1 &
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ELEVENLABS_API_KEY` | Yes* | — | ElevenLabs API key for TTS |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key (if not using gateway) |
| `OPENCLAW_GATEWAY_URL` | No | — | OpenClaw gateway URL for full agent |
| `OPENCLAW_GATEWAY_TOKEN` | No | — | Gateway auth token |
| `OPENCLAW_PORT` | No | `8765` | Server port |
| `OPENCLAW_STT_PROVIDER` | No | `aliyun` | STT provider: `aliyun` or `whisper` |
| `OPENCLAW_ALIYUN_ASR_MODE` | No | `realtime` | Aliyun STT mode: `realtime` (server VAD) or `batch` |
| `OPENCLAW_ALIYUN_REALTIME_VAD_SILENCE_MS` | No | `900` | Server VAD silence threshold (ms) |
| `OPENCLAW_STT_MODEL` | No | `base` | Whisper model size (whisper only) |
| `OPENCLAW_STT_DEVICE` | No | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` (whisper only) |
| `OPENCLAW_REQUIRE_AUTH` | No | `false` | Require API keys for clients |

*One of `OPENAI_API_KEY` or `OPENCLAW_GATEWAY_URL` required.

### Whisper Model Sizes

| Model | Speed | Quality | VRAM | Best For |
|-------|-------|---------|------|----------|
| `tiny` | Fastest | Fair | ~400MB | Quick testing |
| `base` | Fast | Good | ~1GB | **Default. Good balance.** |
| `small` | Medium | Better | ~2GB | Clearer transcription |
| `medium` | Slower | Great | ~5GB | Accuracy priority |
| `large-v3-turbo` | Slow | Best | ~6GB | Maximum accuracy |

### TTS Options

| Backend | Type | Quality | Latency | Notes |
|---------|------|---------|---------|-------|
| **ElevenLabs** | Cloud | Excellent | ~500ms | Default. Streaming supported. |
| Chatterbox | Local | Very Good | ~1s | MIT license, voice cloning |
| XTTS-v2 | Local | Excellent | ~1s | Voice cloning supported |
| Mock | Local | None | 0ms | For testing (silence) |

ElevenLabs uses `eleven_turbo_v2_5` for fastest response.

## OpenClaw Gateway Integration

Connect to your full OpenClaw agent (same memory, tools, and persona as text chat):

```bash
# .env
OPENCLAW_GATEWAY_URL=http://localhost:18789
OPENCLAW_GATEWAY_TOKEN=your-token
ELEVENLABS_API_KEY=your-key
```

Add to your `openclaw.json`:

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  },
  "agents": {
    "list": [
      {
        "id": "voice",
        "workspace": "/path/to/workspace",
        "model": "anthropic/claude-sonnet-4-5"
      }
    ]
  }
}
```

## Architecture

```
┌─────────────┐   WebSocket   ┌─────────────────────────────────────┐
│   Browser   │◄────────────►│          Voice Server               │
│  (mic/spk)  │               │                                     │
└─────────────┘               │  ┌─────────┐  ┌─────┐  ┌─────────┐ │
                              │  │ Whisper │→│ AI  │→│ElevenLabs│ │
                              │  │  (STT)  │  │     │  │  (TTS)  │ │
                              │  └─────────┘  └─────┘  └─────────┘ │
                              │       ↑                     │      │
                              │    [VAD]              [streaming]  │
                              └─────────────────────────────────────┘
```

**Streaming Flow:**
1. User speaks → Whisper transcribes locally
2. AI responds (streamed) → buffer sentences
3. First sentence complete → TTS starts immediately
4. Audio streams to browser while AI continues
5. Result: ~50% faster perceived response

## HTTPS for Mobile

Mobile browsers require HTTPS for microphone access. Options:

**Tailscale Funnel (easiest):**
```bash
tailscale funnel 8765
# Access via https://your-machine.tailnet-name.ts.net
```

**nginx + Let's Encrypt:**
```nginx
server {
    listen 443 ssl;
    server_name voice.yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## API

### WebSocket Protocol

Connect to `ws://localhost:8765/ws`:

```javascript
// Start recording
{ "type": "start_listening" }

// Send audio (base64 PCM float32, 16kHz)
{ "type": "audio", "data": "base64..." }

// Stop recording
{ "type": "stop_listening" }

// Receive events:
{ "type": "transcript", "text": "...", "final": true }
{ "type": "response_chunk", "text": "..." }        // Streaming text
{ "type": "audio_chunk", "data": "...", "sample_rate": 16000 }  // Streaming audio (configurable via OPENCLAW_TTS_OUTPUT_SAMPLE_RATE)
{ "type": "response_complete", "text": "..." }     // Full response
{ "type": "vad_status", "speech_detected": true }  // VAD feedback
```

## Roadmap

- [x] WebSocket voice gateway
- [x] Whisper STT (local)
- [x] ElevenLabs TTS
- [x] Streaming TTS (sentence-by-sentence)
- [x] Voice Activity Detection (Silero)
- [x] Text cleaning (markdown/hashtags/URLs)
- [x] Continuous conversation mode
- [x] OpenClaw gateway integration
- [ ] WebRTC for lower latency
- [ ] Voice cloning UI
- [ ] Docker support

## License

MIT License — see [LICENSE](LICENSE).

## Credits

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — Local STT
- [ElevenLabs](https://elevenlabs.io) — Text-to-Speech
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice Activity Detection
- Built for [OpenClaw](https://openclaw.ai)

---

**Made with 🦞 by [Purple Horizons](https://purplehorizons.io)**
