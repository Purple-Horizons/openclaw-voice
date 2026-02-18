#!/usr/bin/env python3
"""
OpenClaw Voice WebSocket smoke test.

What it checks:
1) STT transcript is non-empty
2) AI response_complete text is non-empty

Usage:
  python scripts/smoke_ws_roundtrip.py \
    --ws ws://127.0.0.1:8765/ws \
    --audio /path/to/sample.ogg
"""

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
from pathlib import Path

import websockets


def to_f32le_16k_mono(audio_path: Path) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=True) as tmp:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            tmp.name,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return Path(tmp.name).read_bytes()


async def run(ws_url: str, audio_path: Path):
    raw = to_f32le_16k_mono(audio_path)
    audio_b64 = base64.b64encode(raw).decode()

    transcript_text = ""
    response_text = ""

    async with websockets.connect(ws_url, max_size=30 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "start_listening"}))
        await ws.recv()  # listening_started

        await ws.send(json.dumps({"type": "audio", "data": audio_b64}))

        # Optional VAD response(s)
        for _ in range(3):
            try:
                maybe = await asyncio.wait_for(ws.recv(), timeout=0.8)
                msg = json.loads(maybe)
                if msg.get("type") == "vad_status":
                    continue
                # put non-vad back into handling by capturing and breaking
                if msg.get("type") == "transcript":
                    transcript_text = msg.get("text", "")
                    break
            except Exception:
                break

        await ws.send(json.dumps({"type": "stop_listening"}))

        # Consume messages until response_complete
        for _ in range(80):
            msg_raw = await asyncio.wait_for(ws.recv(), timeout=45)
            msg = json.loads(msg_raw)
            t = msg.get("type")

            if t == "transcript" and not transcript_text:
                transcript_text = msg.get("text", "")
            elif t == "response_complete":
                response_text = msg.get("text", "")
                break

    if not transcript_text.strip():
        raise RuntimeError("SMOKE FAIL: transcript is empty")
    if not response_text.strip():
        raise RuntimeError("SMOKE FAIL: response_complete text is empty")

    print("SMOKE PASS")
    print(f"transcript: {transcript_text}")
    print(f"response: {response_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", default="ws://127.0.0.1:8765/ws", help="WebSocket endpoint")
    parser.add_argument("--audio", required=True, help="Input audio file path")
    args = parser.parse_args()

    audio_file = Path(args.audio)
    if not audio_file.exists():
        raise SystemExit(f"Audio not found: {audio_file}")

    asyncio.run(run(args.ws, audio_file))
