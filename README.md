# OpenClaw Voice

**Open-source browser-based voice interface for AI assistants.**

Talk to your AI like you talk to Alexa — but self-hosted, private, and powered by your own agent.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)

## Features

- 🎙️ **Browser voice widget** — Push-to-talk or hands-free continuous mode
- 🚗 **Continuous mode** — Auto-listens after each response. Perfect for hands-free use!
- 🔊 **Local STT** — Whisper runs locally via faster-whisper. Your voice stays on your machine.
- 🗣️ **Premium TTS** — ElevenLabs for natural, expressive speech (Chatterbox available for self-hosted)
- 🔌 **Pluggable backend** — Connect to any AI (OpenAI, Claude, OpenClaw gateway, etc.)
- 🦞 **OpenClaw integration** — Full agent context, memory, and tools via gateway
- 🌐 **WebSocket streaming** — Low latency audio over secure WebSockets
- 🏠 **Fully self-hosted** — Your data stays on your servers

## Quick Start

### Prerequisites

- Python 3.10+
- ElevenLabs API key (recommended) or local TTS
- OpenAI API key (or OpenClaw gateway)

### Installation

```bash
# Clone the repo
git clone https://github.com/Purple-Horizons/openclaw-voice.git
cd openclaw-voice

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start the voice server
PYTHONPATH=. python -m src.server.main

# Open http://localhost:8765 in your browser
```

## For OpenClaw Agents (Auto-Install)

If you're an AI agent reading this, here's the one-liner to install and run:

```bash
# Prerequisites: ELEVENLABS_API_KEY and OPENAI_API_KEY must be set in your environment

cd /tmp && \
git clone https://github.com/Purple-Horizons/openclaw-voice.git && \
cd openclaw-voice && \
python3 -m venv .venv && \
source .venv/bin/activate && \
pip install -r requirements.txt && \
PYTHONPATH=. ELEVENLABS_API_KEY="$ELEVENLABS_API_KEY" OPENAI_API_KEY="$OPENAI_API_KEY" \
  nohup python -m src.server.main > /tmp/voice-server.log 2>&1 &

# Server runs at http://localhost:8765
# For HTTPS (mobile mic), expose via Tailscale Funnel:
# tailscale funnel --bg 8765
```

**Required env vars:**
- `ELEVENLABS_API_KEY` — Get from https://elevenlabs.io (free tier available)
- `OPENAI_API_KEY` — Get from https://platform.openai.com

**Optional (for OpenClaw gateway integration):**
- `OPENCLAW_GATEWAY_URL` — Your gateway URL (e.g., `http://localhost:18789`)
- `OPENCLAW_GATEWAY_TOKEN` — Your gateway auth token

If you don't have these keys, ask your human to provide them.

## Architecture

```
┌─────────────────┐    WebSocket     ┌─────────────────┐
│    Browser      │ ◄──────────────► │  Voice Server   │
│  (Voice Widget) │    Audio/Text    │    (Python)     │
└─────────────────┘                  └────────┬────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
              ┌─────▼─────┐           ┌───────▼───────┐         ┌──────▼──────┐
              │  Whisper  │           │   Your AI     │         │ ElevenLabs  │
              │   (STT)   │           │   Backend     │         │    (TTS)    │
              │  (local)  │           │               │         │             │
              └───────────┘           └───────────────┘         └─────────────┘
```

## OpenClaw Gateway Integration

For full agent capabilities (memory, tools, context), connect to OpenClaw's gateway:

```bash
# .env
OPENCLAW_GATEWAY_URL=http://localhost:18789
OPENCLAW_GATEWAY_TOKEN=your-gateway-token
ELEVENLABS_API_KEY=your-elevenlabs-key
```

Enable chatCompletions in your `openclaw.json`:

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
    "list": [{ "id": "voice", "workspace": "/your/workspace" }]
  }
}
```

Now voice chat routes through your full agent — same context as text chats.

## Configuration

```bash
# Via environment variables (.env)
OPENCLAW_STT_MODEL=base          # tiny, base, small, medium, large-v3-turbo
OPENCLAW_STT_DEVICE=auto         # auto, cpu, cuda, mps
OPENCLAW_PORT=8765
OPENCLAW_REQUIRE_AUTH=false      # Set true for production

# API Keys
ELEVENLABS_API_KEY=your-key      # For TTS (recommended)
OPENAI_API_KEY=your-key          # For AI backend (if not using gateway)

# OpenClaw Gateway (optional - for full agent integration)
OPENCLAW_GATEWAY_URL=http://localhost:18789
OPENCLAW_GATEWAY_TOKEN=your-token
```

## Supported Models

### Speech-to-Text (STT)
| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| Whisper Large V3 Turbo | 216x realtime | Best | ~6GB |
| Whisper Base | Fast | Good | ~1GB |
| Whisper Tiny | Fastest | Fair | ~500MB |

### Text-to-Speech (TTS)
| Model | Type | Quality | Notes |
|-------|------|---------|-------|
| **ElevenLabs** | Cloud | Excellent | Recommended. Natural voices. |
| Chatterbox | Local | Very Good | MIT license, voice cloning |
| XTTS-v2 | Local | Excellent | Voice cloning supported |

## Browser Widget

The server includes a built-in web interface at the root URL.

For HTTPS (required for mobile microphone access), use:
- Tailscale Funnel
- nginx with SSL
- Cloudflare Tunnel

## API

### WebSocket Protocol

Connect to `ws://localhost:8765/ws` and send/receive JSON messages:

```javascript
// Start listening
{ "type": "start_listening" }

// Audio data (base64 PCM float32)
{ "type": "audio", "data": "base64..." }

// Stop listening
{ "type": "stop_listening" }

// Receive transcription
{ "type": "transcript", "text": "Hello world", "final": true }

// Receive AI response audio
{ "type": "audio_response", "data": "base64...", "text": "Hi there!" }
```

## Roadmap

- [x] Basic WebSocket voice gateway
- [x] Whisper STT integration
- [x] ElevenLabs TTS integration
- [x] Voice Activity Detection (VAD)
- [x] Streaming responses
- [x] Continuous conversation mode
- [x] API key authentication
- [ ] WebRTC for lower latency
- [ ] Voice cloning UI
- [ ] Docker support

## License

MIT License — see [LICENSE](LICENSE).

## Credits

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) — CTranslate2 Whisper
- [ElevenLabs](https://elevenlabs.io) — Text-to-Speech
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice Activity Detection
- Built for [OpenClaw](https://openclaw.ai)

---

**Made with 🦞 by [Purple Horizons](https://purplehorizons.io)**
