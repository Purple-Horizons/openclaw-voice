#!/bin/bash
# Start OpenClaw Voice server
cd /Users/seneschal/openclaw-voice

# Kill any existing instance
ps aux | grep "src.server.main" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
sleep 1

# Read gateway token from config
TOKEN=$(python3 -c "import json; c=json.load(open('/Users/seneschal/.openclaw/openclaw.json')); print(c['gateway']['auth']['token'])")

# Start server
OPENCLAW_GATEWAY_URL=http://localhost:18789 \
OPENCLAW_GATEWAY_TOKEN="$TOKEN" \
PYTHONPATH=. \
nohup .venv/bin/python3.14 -c "
import uvicorn
uvicorn.run('src.server.main:app', host='0.0.0.0', port=8765, reload=False)
" >> /Users/seneschal/.openclaw/workspace/logs/openclaw-voice.log 2>&1 &

echo "OpenClaw Voice started (PID: $!)"
echo "Open http://localhost:8765 in your browser"
