"""
Text-to-Speech module using ElevenLabs, Chatterbox, or fallbacks.
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
        
        # Try Piper (local, fast ONNX-based)
        try:
            from piper import PiperVoice
            # Look for model files in models/ directory
            models_dir = Path(__file__).parent.parent.parent / "models"
            onnx_files = list(models_dir.glob("*.onnx")) if models_dir.exists() else []
            if onnx_files:
                model_path = str(onnx_files[0])
                logger.info(f"Loading Piper TTS: {onnx_files[0].name}")
                self._piper_model = PiperVoice.load(model_path)
                self._piper_sample_rate = self._piper_model.config.sample_rate
                from piper.config import SynthesisConfig
                self._piper_synth_config = SynthesisConfig(length_scale=0.65)
                self._backend = "piper"
                logger.info(f"✅ Piper TTS loaded ({self._piper_sample_rate}Hz)")
                return
            else:
                logger.warning("Piper installed but no .onnx model found in models/")
        except ImportError:
            logger.warning("Piper not installed")
        except Exception as e:
            logger.warning(f"Piper failed: {e}")
        
        # Try Chatterbox (self-hosted, slow but high quality)
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
        else:
            # Non-streaming fallback
            audio = await self.synthesize(text)
            # Convert float32 [-1,1] to int16 PCM (what the browser client expects)
            audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            logger.info(f"TTS generated {len(audio_bytes)} bytes ({len(audio)/24000:.2f}s)")
            yield audio_bytes
    
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
        
        elif self._backend == "piper":
            import wave
            import io
            buf = io.BytesIO()
            wf = wave.open(buf, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._piper_sample_rate)
            for chunk in self._piper_model.synthesize(text, self._piper_synth_config):
                wf.writeframes(chunk.audio_int16_bytes)
            wf.close()
            buf.seek(44)  # Skip WAV header
            audio_array = np.frombuffer(buf.read(), dtype=np.int16)
            # Normalize to float32 [-1, 1] and resample to 24kHz if needed
            audio_float = audio_array.astype(np.float32) / 32768.0
            if self._piper_sample_rate != 24000:
                # Simple linear resample
                target_len = int(len(audio_float) * 24000 / self._piper_sample_rate)
                indices = np.linspace(0, len(audio_float) - 1, target_len)
                audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float)
            return audio_float
        
        elif self._backend == "chatterbox":
            if self.voice_sample:
                audio = self.model.generate(text, audio_prompt=self.voice_sample)
            else:
                audio = self.model.generate(text)
            return audio.cpu().numpy().flatten().astype(np.float32)
        
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
