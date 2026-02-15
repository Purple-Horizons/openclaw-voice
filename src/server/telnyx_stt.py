"""
Telnyx Speech-to-Text module.

Provides STT via Telnyx AI Inference API (OpenAI-compatible Whisper).
This is a cost-effective alternative to running Whisper locally.

Requirements:
    - TELNYX_API_KEY environment variable
    - aiohttp for async HTTP requests
"""

import asyncio
import os
from typing import Optional

import numpy as np
from loguru import logger

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class TelnyxSTT:
    """
    Speech-to-Text using Telnyx AI Inference API.
    
    Telnyx provides OpenAI-compatible Whisper transcription with
    competitive pricing and no local GPU required.
    
    Usage:
        stt = TelnyxSTT()
        text = await stt.transcribe(audio_array)
    """
    
    API_URL = "https://api.telnyx.com/v2/ai/audio/transcriptions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        language: Optional[str] = "en",
    ):
        self.api_key = api_key or os.environ.get("TELNYX_API_KEY")
        self.model = model
        self.language = language
        self._session: Optional[aiohttp.ClientSession] = None
        
        if not self.api_key:
            logger.warning("TELNYX_API_KEY not set - Telnyx STT will not work")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio as numpy array (float32, 16kHz expected)
            
        Returns:
            Transcribed text
        """
        if not self.api_key:
            logger.error("Telnyx API key not configured")
            return ""
        
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not installed - run: pip install aiohttp")
            return ""
        
        session = await self._get_session()
        
        # Convert float32 audio to WAV format for upload
        # Telnyx expects multipart/form-data with audio file
        wav_bytes = self._audio_to_wav(audio)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        # Build multipart form data
        form_data = aiohttp.FormData()
        form_data.add_field(
            "file",
            wav_bytes,
            filename="audio.wav",
            content_type="audio/wav",
        )
        form_data.add_field("model", self.model)
        if self.language:
            form_data.add_field("language", self.language)
        
        try:
            async with session.post(
                self.API_URL,
                headers=headers,
                data=form_data,
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Telnyx STT error: {response.status} - {error}")
                    return ""
                
                result = await response.json()
                return result.get("text", "")
                
        except aiohttp.ClientError as e:
            logger.error(f"Telnyx STT request failed: {e}")
            return ""
        except Exception as e:
            logger.error(f"Telnyx STT unexpected error: {e}")
            return ""
    
    def _audio_to_wav(self, audio: np.ndarray) -> bytes:
        """
        Convert numpy audio array to WAV bytes.
        
        Args:
            audio: Audio as numpy array (float32)
            
        Returns:
            WAV file as bytes
        """
        import io
        import wave
        
        # Ensure float32 is normalized to int16
        if audio.dtype == np.float32:
            # Clip to valid range
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)
        else:
            audio_int16 = audio.astype(np.int16)
        
        # Create WAV in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
