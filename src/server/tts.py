"""
Text-to-Speech module using ElevenLabs, Telnyx, Chatterbox, or fallbacks.

Supported providers:
    - ElevenLabs (cloud, high quality, streaming)
    - Telnyx (cloud, cost-effective, streaming via AI Inference API)
    - Chatterbox (local, voice cloning)
    - XTTS (local, voice cloning)
    - Mock (testing)
"""

import asyncio
import os
from typing import Optional, AsyncGenerator
from pathlib import Path

import numpy as np
from loguru import logger


class ChatterboxTTS:
    """Text-to-Speech using ElevenLabs, Chatterbox, or fallbacks."""
    
    def __init__(
        self,
        voice_sample: Optional[str] = None,
        device: str = "auto",
        voice_id: Optional[str] = None,  # ElevenLabs voice ID
    ):
        self.voice_sample = voice_sample
        self.device = device
        self.voice_id = voice_id or "cgSgspJ2msm6clMCkdW9"  # Jessica
        self.model = None
        self._backend = "mock"
        self._elevenlabs_client = None
        self._load_model()
    
    def _load_model(self):
        """Load the TTS model."""
        # Try ElevenLabs first (cloud, high quality)
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
        if elevenlabs_key:
            try:
                from elevenlabs import ElevenLabs
                self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
                self._backend = "elevenlabs"
                logger.info("✅ ElevenLabs TTS ready")
                return
            except ImportError:
                logger.warning("ElevenLabs SDK not installed, trying pip install...")
                try:
                    import subprocess
                    subprocess.check_call(["pip", "install", "elevenlabs", "-q"])
                    from elevenlabs import ElevenLabs
                    self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
                    self._backend = "elevenlabs"
                    logger.info("✅ ElevenLabs TTS ready (auto-installed)")
                    return
                except Exception as e:
                    logger.warning(f"ElevenLabs auto-install failed: {e}")
            except Exception as e:
                logger.warning(f"ElevenLabs failed: {e}")
        
        # Try Telnyx TTS (cloud, cost-effective)
        telnyx_key = os.environ.get("TELNYX_API_KEY")
        if telnyx_key:
            try:
                from .telnyx_tts import TelnyxTTS
                self._telnyx_tts = TelnyxTTS(api_key=telnyx_key)
                self._backend = "telnyx"
                logger.info("✅ Telnyx TTS ready")
                return
            except ImportError as e:
                logger.warning(f"Telnyx TTS module not available: {e}")
            except Exception as e:
                logger.warning(f"Telnyx TTS failed: {e}")
        
        # Try Chatterbox (self-hosted)
        try:
            from chatterbox.tts import ChatterboxTTS as CBModel
            logger.info("Loading Chatterbox TTS...")
            self.model = CBModel.from_pretrained(device=self._get_device())
            self._backend = "chatterbox"
            logger.info("✅ Chatterbox loaded")
            return
        except ImportError:
            logger.warning("Chatterbox not installed")
        except Exception as e:
            logger.warning(f"Chatterbox failed: {e}")
        
        # Try XTTS
        try:
            from TTS.api import TTS
            logger.info("Loading Coqui XTTS...")
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            self._backend = "xtts"
            logger.info("✅ XTTS loaded")
            return
        except ImportError:
            logger.warning("Coqui TTS not installed")
        except Exception as e:
            logger.warning(f"XTTS failed: {e}")
        
        # Mock mode
        logger.warning("⚠️ No TTS backend - using mock mode (silence)")
        self._backend = "mock"
    
    def _get_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    
    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)
    
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized audio chunks.
        
        Yields:
            Raw PCM audio chunks (24kHz, 16-bit)
        """
        if self._backend == "elevenlabs":
            try:
                # Use streaming API
                audio_generator = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    output_format="pcm_24000",
                )
                for chunk in audio_generator:
                    yield chunk
            except Exception as e:
                logger.error(f"ElevenLabs streaming error: {e}")
        elif self._backend == "telnyx":
            try:
                async for chunk in self._telnyx_tts.synthesize_stream(text):
                    yield chunk
            except Exception as e:
                logger.error(f"Telnyx TTS streaming error: {e}")
        else:
            # Non-streaming fallback
            audio = await self.synthesize(text)
            yield audio.tobytes()
    
    def _synthesize_sync(self, text: str) -> np.ndarray:
        """Synchronous synthesis."""
        if self._backend == "elevenlabs":
            try:
                # Generate audio with ElevenLabs (turbo model for speed)
                audio_generator = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5",  # Fastest model (~2x faster)
                    output_format="pcm_24000",  # 24kHz PCM (matches server expectation)
                )
                # Collect all chunks
                audio_bytes = b"".join(audio_generator)
                # Convert PCM bytes to numpy array
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                # Normalize to float32 [-1, 1]
                return audio_array.astype(np.float32) / 32768.0
            except Exception as e:
                logger.error(f"ElevenLabs TTS error: {e}")
                return np.zeros(16000, dtype=np.float32)  # 1 sec silence on error
        
        elif self._backend == "telnyx":
            # Telnyx uses async, so we need to run in event loop
            try:
                loop = asyncio.new_event_loop()
                audio_bytes = b""
                async def collect():
                    nonlocal audio_bytes
                    async for chunk in self._telnyx_tts.synthesize_stream(text):
                        audio_bytes += chunk
                loop.run_until_complete(collect())
                loop.close()
                if audio_bytes:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    return audio_array.astype(np.float32) / 32768.0
                return np.zeros(16000, dtype=np.float32)
            except Exception as e:
                logger.error(f"Telnyx TTS error: {e}")
                return np.zeros(16000, dtype=np.float32)
        
        elif self._backend == "chatterbox":
            if self.voice_sample:
                audio = self.model.generate(text, audio_prompt=self.voice_sample)
            else:
                audio = self.model.generate(text)
            return audio.cpu().numpy().astype(np.float32)
        
        elif self._backend == "xtts":
            if self.voice_sample:
                wav = self.model.tts(text=text, speaker_wav=self.voice_sample, language="en")
            else:
                wav = self.model.tts(text=text, language="en")
            return np.array(wav, dtype=np.float32)
        
        else:
            # Mock mode - return short silence
            logger.debug(f"Mock TTS: '{text[:50]}...'")
            # 0.5 seconds of silence at 24kHz
            return np.zeros(12000, dtype=np.float32)
