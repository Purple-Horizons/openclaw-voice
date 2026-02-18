"""
Alibaba Cloud qwen3-tts-flash TTS module.
Supports streaming audio synthesis.
"""

import asyncio
import base64
import json
import os
from typing import Optional, AsyncGenerator
import aiohttp

import numpy as np
from loguru import logger


class AliyunTTS:
    """Alibaba Cloud qwen3-tts-flash Text-to-Speech."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "Cherry",  # Popular Chinese voice
        language: str = "Chinese",
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for Aliyun TTS")
        self.voice = voice
        self.language = language
        self.sample_rate = 22050  # qwen3-tts-flash output rate
        
    async def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text
            
        Returns:
            Audio as numpy array (float32)
        """
        import dashscope
        dashscope.api_key = self.api_key
        
        try:
            response = dashscope.MultiModalConversation.call(
                model='qwen3-tts-flash',
                text=text,
                voice=self.voice,
                language_type=self.language,
                stream=False
            )
            
            if response.status_code == 200:
                audio_url = response.output.get('audio', {}).get('url', '')
                if audio_url:
                    # Download audio from URL
                    audio_data = await self._download_audio(audio_url)
                    return audio_data
                    
            logger.warning(f"TTS response: {response.code} - {response.message}")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            
        # Return silence on error
        return np.zeros(self.sample_rate, dtype=np.float32)
    
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks.
        
        Args:
            text: Input text
            
        Yields:
            Raw PCM audio chunks
        """
        import dashscope
        dashscope.api_key = self.api_key
        
        yielded = False

        try:
            # dashscope streaming returns an iterator/generator
            response_iter = dashscope.MultiModalConversation.call(
                model='qwen3-tts-flash',
                text=text,
                voice=self.voice,
                language_type=self.language,
                stream=True
            )

            for chunk in response_iter:
                status_code = getattr(chunk, "status_code", 200)
                if status_code != 200:
                    logger.warning(f"TTS streaming chunk status: {status_code}")
                    continue

                output = getattr(chunk, "output", None)
                if output and output.get('audio'):
                    audio_url = output['audio'].get('url', '')
                    if audio_url:
                        audio_data = await self._download_audio(audio_url)
                        if audio_data is not None:
                            # Normalize protocol with ElevenLabs path:
                            # send PCM int16 bytes for client playback queue.
                            audio_i16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
                            yielded = True
                            yield audio_i16.tobytes()

        except Exception as e:
            logger.error(f"TTS stream error: {e}")

        # Yield a short silence only if nothing was produced.
        if not yielded:
            yield np.zeros(1000, dtype=np.int16).tobytes()
    
    async def _download_audio(self, url: str) -> Optional[np.ndarray]:
        """Download audio from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        audio_bytes = await resp.read()
                        
                        # Try to convert to numpy
                        # qwen3-tts-flash returns WAV format
                        import io
                        import wave
                        
                        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
                            frames = wav.readframes(wav.getnframes())
                            audio = np.frombuffer(frames, dtype=np.int16)
                            # Convert to float32
                            return audio.astype(np.float32) / 32768.0
                            
        except Exception as e:
            logger.error(f"Audio download error: {e}")
            
        return None
    
    def synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.synthesize(text))


class AliyunTTSProxy:
    """
    Proxy class to match the interface expected by openclaw-voice.
    Wraps AliyunTTS to provide the expected interface.
    """
    
    def __init__(self, api_key: Optional[str] = None, voice: str = "Cherry"):
        self._tts = AliyunTTS(api_key=api_key, voice=voice)
        self._backend = "aliyun"
        
    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to speech."""
        return await self._tts.synthesize(text)
    
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream synthesis."""
        async for chunk in self._tts.synthesize_stream(text):
            yield chunk
