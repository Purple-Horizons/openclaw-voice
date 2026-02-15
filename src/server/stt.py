"""
Speech-to-Text module using Whisper.

Supported providers:
    - faster-whisper (local, GPU accelerated)
    - openai-whisper (local)
    - Telnyx AI Inference API (cloud, OpenAI-compatible)
    - Mock (testing)
"""

import asyncio
import os
from typing import Optional

import numpy as np
from loguru import logger


class WhisperSTT:
    """Whisper-based Speech-to-Text."""
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        language: str = "en",
        use_telnyx: bool = False,  # Use Telnyx cloud STT
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.use_telnyx = use_telnyx or bool(os.environ.get("TELNYX_API_KEY") and not os.environ.get("OPENAI_API_KEY"))
        self.model = None
        self._backend = "mock"
        self._telnyx_stt = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        # Try Telnyx cloud STT first if requested or if no local resources
        if self.use_telnyx:
            try:
                from .telnyx_stt import TelnyxSTT
                self._telnyx_stt = TelnyxSTT(language=self.language)
                self._backend = "telnyx"
                logger.info("✅ Telnyx cloud STT ready")
                return
            except ImportError:
                logger.warning("Telnyx STT module not available")
            except Exception as e:
                logger.warning(f"Telnyx STT failed: {e}")
        
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            
            if self.device == "auto":
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    compute_type = "float16"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "cpu"
                    compute_type = "int8"
                else:
                    self.device = "cpu"
                    compute_type = "int8"
            elif self.device == "cuda":
                compute_type = "float16"
            else:
                compute_type = "int8"
            
            logger.info(f"Loading faster-whisper {self.model_name} on {self.device}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device if self.device != "mps" else "cpu",
                compute_type=compute_type,
            )
            self._backend = "faster-whisper"
            logger.info("✅ faster-whisper loaded")
            return
        except ImportError:
            logger.warning("faster-whisper not available")
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")
        
        # Try openai-whisper
        try:
            import whisper
            
            if self.device == "auto":
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading openai-whisper {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self._backend = "openai-whisper"
            logger.info("✅ openai-whisper loaded")
            return
        except ImportError:
            logger.warning("openai-whisper not available")
        except Exception as e:
            logger.warning(f"openai-whisper failed: {e}")
        
        # Mock mode for testing
        logger.warning("⚠️ No STT backend - using mock mode")
        self._backend = "mock"
    
    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)
    
    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription."""
        if self._backend == "telnyx":
            # Telnyx uses async, run in event loop
            try:
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(self._telnyx_stt.transcribe(audio))
                loop.close()
                return result
            except Exception as e:
                logger.error(f"Telnyx STT error: {e}")
                return ""
        
        elif self._backend == "faster-whisper":
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
            )
            return " ".join(segment.text for segment in segments).strip()
        
        elif self._backend == "openai-whisper":
            result = self.model.transcribe(audio, language=self.language)
            return result["text"].strip()
        
        else:
            # Mock mode - return placeholder
            logger.debug(f"Mock STT: received {len(audio)} samples")
            return "[Mock transcription - install whisper for real STT]"
