"""
Telnyx Text-to-Speech module.

Provides TTS via Telnyx AI Inference API (OpenAI-compatible).
This is a cost-effective alternative to ElevenLabs with streaming support.

Requirements:
    - TELNYX_API_KEY environment variable
    - aiohttp for async HTTP requests
"""

import asyncio
import os
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class TelnyxTTS:
    """
    Text-to-Speech using Telnyx AI Inference API.
    
    Telnyx provides OpenAI-compatible TTS with competitive pricing
    and low latency. Ideal for production voice applications.
    
    Usage:
        tts = TelnyxTTS()
        async for audio_chunk in tts.synthesize_stream("Hello world"):
            # Process PCM audio chunk
            pass
    """
    
    API_URL = "https://api.telnyx.com/v2/ai/speech"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "alloy",  # OpenAI-compatible voice names
        model: str = "tts-1-hd",  # tts-1 or tts-1-hd
        response_format: str = "pcm",  # pcm for streaming
    ):
        self.api_key = api_key or os.environ.get("TELNYX_API_KEY")
        self.voice = voice
        self.model = model
        self.response_format = response_format
        self._session: Optional[aiohttp.ClientSession] = None
        
        if not self.api_key:
            logger.warning("TELNYX_API_KEY not set - Telnyx TTS will not work")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio as numpy array (float32, 24kHz)
        """
        audio_bytes = b""
        async for chunk in self.synthesize_stream(text):
            audio_bytes += chunk
        
        if not audio_bytes:
            return np.zeros(16000, dtype=np.float32)
        
        # Convert PCM to float32
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_array.astype(np.float32) / 32768.0
    
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks.
        
        Yields:
            Raw PCM audio chunks (24kHz, 16-bit mono)
        """
        if not self.api_key:
            logger.error("Telnyx API key not configured")
            return
        
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not installed - run: pip install aiohttp")
            return
        
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/pcm",
        }
        
        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
        }
        
        try:
            async with session.post(
                self.API_URL,
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Telnyx TTS error: {response.status} - {error}")
                    return
                
                # Stream audio chunks
                async for chunk in response.content.iter_chunked(4096):
                    yield chunk
                    
        except aiohttp.ClientError as e:
            logger.error(f"Telnyx TTS request failed: {e}")
        except Exception as e:
            logger.error(f"Telnyx TTS unexpected error: {e}")
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
