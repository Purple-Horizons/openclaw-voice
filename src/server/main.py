"""
OpenClaw Voice Server

WebSocket server that handles:
- Audio input from browser
- Speech-to-Text via Whisper
- AI backend communication
- Text-to-Speech via ElevenLabs
- Audio streaming back to browser
"""

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from pydantic_settings import BaseSettings

from .stt import WhisperSTT
from .tts import ChatterboxTTS
from .backend import AIBackend
from .vad import VoiceActivityDetector
from .auth import token_manager, load_keys_from_env, APIKey
from .text_utils import clean_for_speech


class Settings(BaseSettings):
    """Server configuration."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8765
    
    # Auth
    require_auth: bool = False  # Set True for production
    master_key: Optional[str] = None  # Admin key for full access
    
    # STT
    stt_model: str = "base"  # tiny, base, small, medium, large-v3-turbo
    stt_device: str = "auto"  # auto, cpu, cuda, mps
    
    # TTS
    tts_model: str = "chatterbox"
    tts_voice: Optional[str] = None  # Path to voice sample for cloning
    
    # AI Backend
    backend_type: str = "openai"  # openai, openclaw, custom
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    
    # OpenClaw Gateway (auto-detected from OPENCLAW_GATEWAY_URL + TOKEN)
    openclaw_gateway_url: Optional[str] = None
    openclaw_gateway_token: Optional[str] = None
    openclaw_session_key: Optional[str] = None  # Route to specific session (e.g. main)
    
    # Audio
    sample_rate: int = 16000
    
    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"


settings = Settings()
app = FastAPI(title="OpenClaw Voice", version="0.1.0")

# Global instances (initialized on startup)
stt: Optional[WhisperSTT] = None
tts: Optional[ChatterboxTTS] = None
backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None


@app.on_event("startup")
async def startup():
    """Initialize models on server start."""
    global stt, tts, backend, vad
    
    logger.info("Initializing OpenClaw Voice server...")
    
    # Load API keys
    load_keys_from_env()
    if settings.require_auth:
        logger.info("🔐 Authentication ENABLED")
    else:
        logger.warning("⚠️ Authentication DISABLED (dev mode)")
    
    # Initialize STT
    logger.info(f"Loading STT model: {settings.stt_model}")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        device=settings.stt_device,
    )
    
    # Initialize TTS
    logger.info(f"Loading TTS model: {settings.tts_model}")
    tts = ChatterboxTTS(
        voice_sample=settings.tts_voice,
    )
    
    # Initialize AI backend
    # Auto-detect OpenClaw gateway
    gateway_url = settings.openclaw_gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.openclaw_gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    
    if gateway_url and gateway_token:
        # Use OpenClaw gateway
        session_key = settings.openclaw_session_key or os.getenv("OPENCLAW_SESSION_KEY")
        logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url}")
        if session_key:
            logger.info(f"🔗 Session routing → {session_key}")
        backend = AIBackend(
            backend_type="openai",  # Gateway speaks OpenAI API
            url=f"{gateway_url}/v1",
            model="openclaw:voice",  # Maps to 'voice' agent in config
            api_key=gateway_token,
            session_key=session_key,
            system_prompt=(
                "This conversation is happening via real-time voice chat. "
                "Keep responses concise and conversational — a few sentences "
                "at most unless the topic genuinely needs depth. "
                "No markdown, bullet points, code blocks, or special formatting."
            ),
        )
    else:
        # Fallback to direct OpenAI
        logger.info(f"Connecting to backend: {settings.backend_type}")
        backend = AIBackend(
            backend_type=settings.backend_type,
            url=settings.backend_url,
            model=settings.backend_model,
            api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
        )
    
    # Initialize VAD
    logger.info("Loading VAD model")
    vad = VoiceActivityDetector()
    
    logger.info("✅ OpenClaw Voice server ready!")


@app.get("/")
@app.get("/voice")
@app.get("/voice/")
async def index():
    """Serve the demo page."""
    return FileResponse("src/client/index.html")


@app.post("/api/keys")
async def create_api_key(
    name: str,
    tier: str = "free",
    master_key: Optional[str] = None,
):
    """
    Create a new API key (requires master key).
    
    curl -X POST "http://localhost:8765/api/keys?name=myapp&tier=pro" \
         -H "x-master-key: YOUR_MASTER_KEY"
    """
    # Verify master key
    if settings.require_auth:
        if not master_key and not settings.master_key:
            return {"error": "Master key required"}
        
        provided_key = master_key or ""
        if provided_key != settings.master_key:
            # Also check if it's a valid master-tier key
            key = token_manager.validate_key(provided_key)
            if not key or key.tier != "enterprise":
                return {"error": "Invalid master key"}
    
    from .auth import PRICING_TIERS
    
    if tier not in PRICING_TIERS:
        return {"error": f"Invalid tier. Options: {list(PRICING_TIERS.keys())}"}
    
    tier_config = PRICING_TIERS[tier]
    
    plaintext_key, api_key = token_manager.generate_key(
        name=name,
        tier=tier,
        rate_limit=tier_config["rate_limit"],
        monthly_minutes=tier_config["monthly_minutes"],
    )
    
    return {
        "api_key": plaintext_key,  # Only shown once!
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier,
        "monthly_minutes": api_key.monthly_minutes,
        "rate_limit": api_key.rate_limit_per_minute,
    }


@app.get("/api/usage")
async def get_usage(api_key: str):
    """
    Get usage stats for an API key.
    
    curl "http://localhost:8765/api/usage?api_key=ocv_xxx"
    """
    key = token_manager.validate_key(api_key)
    if not key:
        return {"error": "Invalid API key"}
    
    return token_manager.get_usage(key)


@app.websocket("/ws")
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections."""
    # Check for API key in query params or headers
    api_key_str = websocket.query_params.get("api_key") or \
                  websocket.headers.get("x-api-key")
    
    api_key: Optional[APIKey] = None
    
    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return
        
        api_key = token_manager.validate_key(api_key_str)
        if not api_key:
            await websocket.close(code=4002, reason="Invalid API key")
            return
        
        if not token_manager.check_rate_limit(api_key):
            await websocket.close(code=4003, reason="Rate limit exceeded")
            return
        
        logger.info(f"Client connected: {api_key.name} (tier={api_key.tier})")
    else:
        # Dev mode - allow all
        if api_key_str:
            api_key = token_manager.validate_key(api_key_str)
        logger.info("Client connected (auth disabled)")
    
    await websocket.accept()
    
    audio_buffer = []
    client_sample_rate = 16000  # Default, updated by client
    is_listening = False
    session_start = None
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg["type"] == "start_listening":
                is_listening = True
                audio_buffer = []
                await websocket.send_json({"type": "listening_started"})
                logger.debug("Started listening")
                
            elif msg["type"] == "stop_listening":
                is_listening = False
                
                if audio_buffer:
                    # Combine audio chunks
                    audio_data = np.concatenate(audio_buffer)
                    duration = len(audio_data) / client_sample_rate
                    logger.info(f"Audio buffer: {len(audio_buffer)} chunks, {len(audio_data)} samples, {duration:.2f}s @ {client_sample_rate}Hz, max={np.max(np.abs(audio_data)):.4f}")
                    
                    # Resample to 16kHz if needed (Whisper expects 16kHz)
                    if client_sample_rate != 16000:
                        target_len = int(len(audio_data) * 16000 / client_sample_rate)
                        indices = np.linspace(0, len(audio_data) - 1, target_len)
                        audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)
                        logger.debug(f"Resampled {client_sample_rate}Hz -> 16kHz ({target_len} samples)")
                    
                    # Transcribe
                    logger.debug("Transcribing audio...")
                    transcript = await stt.transcribe(audio_data)
                    
                    await websocket.send_json({
                        "type": "transcript",
                        "text": transcript,
                        "final": True,
                    })
                    logger.info(f"Transcript: {transcript}")
                    
                    if transcript.strip():
                        # Stream AI response with progressive TTS
                        logger.debug("Streaming AI response...")
                        
                        full_response = ""
                        sentence_buffer = ""
                        audio_chunks = []
                        
                        # Stream response and synthesize sentences as they complete
                        async for chunk in backend.chat_stream(transcript):
                            full_response += chunk
                            sentence_buffer += chunk
                            
                            # Send text chunk for progressive display
                            await websocket.send_json({
                                "type": "response_chunk",
                                "text": chunk,
                            })
                            
                            # Check for sentence boundaries
                            while any(sep in sentence_buffer for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                                # Find first sentence boundary
                                earliest_idx = len(sentence_buffer)
                                for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                                    idx = sentence_buffer.find(sep)
                                    if idx != -1 and idx < earliest_idx:
                                        earliest_idx = idx + len(sep)
                                
                                if earliest_idx < len(sentence_buffer):
                                    sentence = sentence_buffer[:earliest_idx].strip()
                                    sentence_buffer = sentence_buffer[earliest_idx:]
                                    
                                    if sentence:
                                        # Clean and synthesize this sentence
                                        speech_text = clean_for_speech(sentence)
                                        if speech_text:
                                            logger.debug(f"Synthesizing: {speech_text[:50]}...")
                                            async for audio_chunk in tts.synthesize_stream(speech_text):
                                                audio_b64 = base64.b64encode(audio_chunk).decode()
                                                await websocket.send_json({
                                                    "type": "audio_chunk",
                                                    "data": audio_b64,
                                                    "sample_rate": 24000,
                                                })
                                else:
                                    break
                        
                        # Handle any remaining text
                        if sentence_buffer.strip():
                            speech_text = clean_for_speech(sentence_buffer.strip())
                            if speech_text:
                                async for audio_chunk in tts.synthesize_stream(speech_text):
                                    audio_b64 = base64.b64encode(audio_chunk).decode()
                                    await websocket.send_json({
                                        "type": "audio_chunk",
                                        "data": audio_b64,
                                        "sample_rate": 24000,
                                    })
                        
                        # Signal end of response
                        await websocket.send_json({
                            "type": "response_complete",
                            "text": full_response,
                        })
                        logger.info(f"Response complete: {full_response[:100]}...")
                
                audio_buffer = []
                await websocket.send_json({"type": "listening_stopped"})
                logger.debug("Stopped listening")
                
            elif msg["type"] == "audio" and is_listening:
                # Decode base64 audio
                audio_bytes = base64.b64decode(msg["data"])
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                audio_buffer.append(audio_np)
                # Track client sample rate (Safari uses 48kHz, Chrome may use 16kHz)
                if "sample_rate" in msg:
                    client_sample_rate = int(msg["sample_rate"])
                
                # VAD check - notify client if speech detected
                if vad and len(audio_np) > 0:
                    has_speech = vad.is_speech(audio_np)
                    await websocket.send_json({
                        "type": "vad_status",
                        "speech_detected": has_speech,
                    })
                
            elif msg["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Serve static files for client
client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
