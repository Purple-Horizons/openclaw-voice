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
    reload: bool = False
    
    # Auth
    require_auth: bool = False  # Set True for production
    master_key: Optional[str] = None  # Admin key for full access
    
    # STT
    stt_model: str = "base"  # tiny, base, small, medium, large-v3-turbo
    stt_device: str = "auto"  # auto, cpu, cuda, mps
    stt_provider: str = "whisper"  # whisper, aliyun
    stt_language: str = "zh"

    # Aliyun ASR mode: realtime (server VAD) | batch
    aliyun_asr_mode: str = "realtime"
    aliyun_realtime_vad_silence_ms: int = 900
    
    # TTS
    tts_model: str = "chatterbox"
    tts_voice: str = "Cherry"  # Aliyun voice
    tts_provider: str = "elevenlabs"  # elevenlabs, aliyun
    
    # Aliyun
    dashscope_api_key: Optional[str] = None
    
    # AI Backend
    backend_type: str = "openai"  # openai, openclaw, custom
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    
    # OpenClaw Gateway (from OPENCLAW_GATEWAY_URL / OPENCLAW_GATEWAY_TOKEN)
    gateway_url: Optional[str] = None
    gateway_token: Optional[str] = None
    gateway_model: str = "main"
    
    # Audio
    sample_rate: int = 16000  # input audio sample rate from client
    tts_output_sample_rate: int = 16000  # output audio sample rate to client
    
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


def _resample_pcm16(audio_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample PCM16 mono bytes with linear interpolation."""
    if src_rate == dst_rate or not audio_bytes:
        return audio_bytes

    audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio_i16.size == 0:
        return audio_bytes

    src = audio_i16.astype(np.float32)
    src_len = src.shape[0]
    dst_len = max(1, int(round(src_len * dst_rate / src_rate)))

    src_idx = np.linspace(0, src_len - 1, num=src_len, dtype=np.float32)
    dst_idx = np.linspace(0, src_len - 1, num=dst_len, dtype=np.float32)
    dst = np.interp(dst_idx, src_idx, src)

    out_i16 = np.clip(dst, -32768, 32767).astype(np.int16)
    return out_i16.tobytes()


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
    
    # Get API key
    api_key = settings.dashscope_api_key or os.environ.get("DASHSCOPE_API_KEY")
    
    # Initialize STT
    logger.info(f"Loading STT provider: {settings.stt_provider}")
    
    if settings.stt_provider == "aliyun":
        if not api_key:
            raise RuntimeError("Aliyun STT enabled but DASHSCOPE_API_KEY is missing")

        if settings.aliyun_asr_mode == "batch":
            from .aliyun_stt import AliyunASR
            stt = AliyunASR(api_key=api_key, language=settings.stt_language)
            logger.info("✅ Aliyun ASR ready (batch mode)")
        else:
            from .aliyun_stt import AliyunASRRealtimeFactory
            stt = AliyunASRRealtimeFactory(
                api_key=api_key,
                language=settings.stt_language,
                silence_duration_ms=settings.aliyun_realtime_vad_silence_ms,
            )
            logger.info("✅ Aliyun ASR realtime ready (server_vad)")
    else:
        logger.info(f"Loading STT model: {settings.stt_model}")
        stt = WhisperSTT(
            model_name=settings.stt_model,
            device=settings.stt_device,
        )
    
    # Initialize TTS
    logger.info(f"Loading TTS provider: {settings.tts_provider}")
    
    if settings.tts_provider == "aliyun":
        if not api_key:
            raise RuntimeError("Aliyun TTS enabled but DASHSCOPE_API_KEY is missing")
        from .aliyun_tts import AliyunTTSProxy
        tts = AliyunTTSProxy(api_key=api_key, voice=settings.tts_voice)
        logger.info("✅ Aliyun TTS ready")
    else:
        logger.info(f"Loading TTS model: {settings.tts_model}")
        tts = ChatterboxTTS(
            voice_sample=settings.tts_voice,
        )
    
    # Initialize AI backend
    gateway_url = settings.gateway_url or os.environ.get("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.gateway_token or os.environ.get("OPENCLAW_GATEWAY_TOKEN")

    if gateway_url and gateway_token:
        # Use OpenClaw gateway (OpenAI-compatible endpoint)
        gateway_base = gateway_url.rstrip("/")
        if not gateway_base.endswith("/v1"):
            gateway_base = f"{gateway_base}/v1"

        logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_base}")
        backend = AIBackend(
            backend_type="openai",
            url=gateway_base,
            model=settings.gateway_model,
            api_key=gateway_token,
            system_prompt=(
                "This conversation is happening via real-time voice chat. "
                "Keep responses concise and conversational — a few sentences "
                "at most unless the topic genuinely needs depth. "
                "No markdown, bullet points, code blocks, or special formatting."
            ),
        )
    else:
        # Fallback to direct OpenAI/custom backend
        logger.info(f"Connecting to backend: {settings.backend_type}")
        backend = AIBackend(
            backend_type=settings.backend_type,
            url=settings.backend_url,
            model=settings.backend_model,
            api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
        )
    
    # Initialize local VAD only when needed.
    if settings.stt_provider == "aliyun" and settings.aliyun_asr_mode != "batch":
        vad = None
        logger.info("Skip local VAD (using Aliyun realtime server_vad)")
    else:
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
    is_listening = False
    session_start = None
    asr_realtime_session = None
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            
            if msg["type"] == "start_listening":
                is_listening = True
                audio_buffer = []

                # Realtime ASR session (Aliyun server_vad mode)
                asr_realtime_session = None
                if hasattr(stt, "create_session"):
                    try:
                        asr_realtime_session = stt.create_session()
                        await asr_realtime_session.start()
                        logger.debug("Aliyun realtime ASR session started")
                    except Exception as e:
                        logger.warning(f"Failed to start realtime ASR session, fallback to batch: {e}")
                        asr_realtime_session = None

                await websocket.send_json({"type": "listening_started"})
                logger.debug("Started listening")
                
            elif msg["type"] == "stop_listening":
                is_listening = False

                # Tell client recording stopped immediately (UI can show processing state).
                await websocket.send_json({"type": "listening_stopped"})

                transcript = ""
                audio_data = None
                audio_energy = 0.0

                if audio_buffer:
                    # Keep local copy for fallback batch STT if realtime session has no final transcript.
                    audio_data = np.concatenate(audio_buffer)
                    if len(audio_data) > 0:
                        audio_energy = float(np.mean(np.abs(audio_data)))

                # Prefer realtime final transcript when available.
                if asr_realtime_session is not None:
                    logger.debug("Collecting realtime ASR transcript...")
                    try:
                        transcript = await asr_realtime_session.stop_and_get_transcript(timeout_sec=3.0)
                    except Exception as e:
                        logger.warning(f"Realtime ASR stop/get transcript failed: {e}")

                # Fallback to batch transcription only when speech was actually detected.
                # This avoids silence hallucinations (e.g. recognizing pure silence as "嗯").
                should_batch_fallback = (
                    audio_data is not None
                    and not transcript.strip()
                    and (
                        asr_realtime_session is None
                        or getattr(asr_realtime_session, "saw_speech_started", False)
                        # Realtime VAD may occasionally miss events on bursty input;
                        # use signal energy as secondary indicator for fallback.
                        or audio_energy > 0.008
                    )
                )
                if should_batch_fallback:
                    logger.debug("Transcribing audio (batch fallback)...")
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
                                            out_chunk = _resample_pcm16(
                                                audio_chunk,
                                                src_rate=24000,
                                                dst_rate=settings.tts_output_sample_rate,
                                            )
                                            audio_b64 = base64.b64encode(out_chunk).decode()
                                            await websocket.send_json({
                                                "type": "audio_chunk",
                                                "data": audio_b64,
                                                "sample_rate": settings.tts_output_sample_rate,
                                            })
                            else:
                                break

                    # Handle any remaining text
                    if sentence_buffer.strip():
                        speech_text = clean_for_speech(sentence_buffer.strip())
                        if speech_text:
                            async for audio_chunk in tts.synthesize_stream(speech_text):
                                out_chunk = _resample_pcm16(
                                    audio_chunk,
                                    src_rate=24000,
                                    dst_rate=settings.tts_output_sample_rate,
                                )
                                audio_b64 = base64.b64encode(out_chunk).decode()
                                await websocket.send_json({
                                    "type": "audio_chunk",
                                    "data": audio_b64,
                                    "sample_rate": settings.tts_output_sample_rate,
                                })

                    # Signal end of response
                    await websocket.send_json({
                        "type": "response_complete",
                        "text": full_response,
                    })
                    logger.info(f"Response complete: {full_response[:100]}...")
                else:
                    # Important for continuous mode UI state machine:
                    # always close this turn even when no transcript is recognized.
                    await websocket.send_json({
                        "type": "response_complete",
                        "text": "",
                    })
                    logger.info("No transcript recognized; sent empty response_complete")
                
                audio_buffer = []
                asr_realtime_session = None
                logger.debug("Stopped listening")
                
            elif msg["type"] == "audio" and is_listening:
                # Decode base64 audio
                audio_bytes = base64.b64decode(msg["data"])
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                audio_buffer.append(audio_np)

                # Realtime Aliyun ASR path: forward chunk + emit server_vad events.
                if asr_realtime_session is not None:
                    try:
                        await asr_realtime_session.append_audio_float32(audio_np)
                        for event in asr_realtime_session.drain_events():
                            et = event.get("type")
                            if et == "speech_started":
                                await websocket.send_json({
                                    "type": "vad_status",
                                    "speech_detected": True,
                                    "event": "speech_started",
                                })
                            elif et == "speech_stopped":
                                await websocket.send_json({
                                    "type": "vad_status",
                                    "speech_detected": False,
                                    "event": "speech_stopped",
                                })
                    except Exception as e:
                        logger.warning(f"Realtime ASR append/drain failed: {e}")

                # Local VAD fallback (whisper/batch modes)
                elif vad and len(audio_np) > 0:
                    has_speech = vad.is_speech(audio_np)
                    await websocket.send_json({
                        "type": "vad_status",
                        "speech_detected": has_speech,
                    })
                
            elif msg["type"] == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        if asr_realtime_session is not None:
            try:
                await asr_realtime_session.stop_and_get_transcript(timeout_sec=0.5)
            except Exception:
                pass
        logger.info("Client disconnected")
    except Exception as e:
        if asr_realtime_session is not None:
            try:
                await asr_realtime_session.stop_and_get_transcript(timeout_sec=0.5)
            except Exception:
                pass
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
        reload=settings.reload,
    )
