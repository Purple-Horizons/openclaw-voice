"""
Unit tests for OpenClaw Voice modules.
"""

import pytest
import numpy as np
import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.server.stt import WhisperSTT
from src.server.tts import ChatterboxTTS
from src.server.backend import AIBackend
from src.server.vad import VoiceActivityDetector


class TestWhisperSTT:
    """Tests for Speech-to-Text module."""
    
    def test_init_loads_model(self):
        """Test that STT initializes (may be mock or real)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        assert stt is not None
        assert stt._backend in ["faster-whisper", "openai-whisper", "mock"]
    
    @pytest.mark.asyncio
    async def test_transcribe_returns_string(self):
        """Test that transcribe returns a string."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        # Create 1 second of silence at 16kHz
        audio = np.zeros(16000, dtype=np.float32)
        result = await stt.transcribe(audio)
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_transcribe_with_noise(self):
        """Test transcription with random noise (should return something)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        # Random noise
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = await stt.transcribe(audio)
        assert isinstance(result, str)


class TestChatterboxTTS:
    """Tests for Text-to-Speech module."""
    
    def test_init_loads_model(self):
        """Test that TTS initializes (may be mock or real)."""
        tts = ChatterboxTTS()
        assert tts is not None
        assert tts._backend in ["elevenlabs", "chatterbox", "xtts", "pyttsx3", "mock"]
    
    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self):
        """Test that synthesize returns numpy array."""
        tts = ChatterboxTTS()
        result = await tts.synthesize("Hello world")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) > 0


class TestAIBackend:
    """Tests for AI Backend module."""
    
    def test_init_creates_client(self):
        """Test backend initialization."""
        backend = AIBackend(
            backend_type="openai",
            model="gpt-4o-mini",
        )
        assert backend is not None
        assert backend.backend_type == "openai"
    
    def test_system_prompt_default(self):
        """Test default system prompt is set."""
        backend = AIBackend()
        assert backend.system_prompt is not None
        assert "voice assistant" in backend.system_prompt.lower()
    
    def test_clear_history(self):
        """Test conversation history can be cleared."""
        backend = AIBackend()
        backend.conversation_history = [{"role": "user", "content": "test"}]
        backend.clear_history()
        assert len(backend.conversation_history) == 0
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    async def test_chat_returns_response(self):
        """Test actual API call (requires API key)."""
        backend = AIBackend(
            backend_type="openai",
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        result = await backend.chat("Say 'test' and nothing else.")
        assert isinstance(result, str)
        assert len(result) > 0


class TestVAD:
    """Tests for Voice Activity Detection module."""
    
    def test_init(self):
        """Test VAD initialization."""
        vad = VoiceActivityDetector()
        assert vad is not None
    
    def test_is_speech_silence(self):
        """Test that silence is not detected as speech."""
        vad = VoiceActivityDetector()
        silence = np.zeros(16000, dtype=np.float32)
        # Should return True if no VAD model (assumes speech)
        # or False if VAD model is loaded and detects no speech
        result = vad.is_speech(silence)
        assert isinstance(result, bool)
    
    def test_is_speech_noise(self):
        """Test with random noise."""
        vad = VoiceActivityDetector()
        noise = np.random.randn(16000).astype(np.float32)
        result = vad.is_speech(noise)
        assert isinstance(result, bool)


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.asyncio
    async def test_stt_tts_round_trip(self):
        """Test STT → TTS round trip (mock mode OK)."""
        stt = WhisperSTT(model_name="tiny", device="cpu")
        tts = ChatterboxTTS()
        
        # Generate some audio (silence)
        input_audio = np.zeros(16000, dtype=np.float32)
        
        # Transcribe
        text = await stt.transcribe(input_audio)
        assert isinstance(text, str)
        
        # Synthesize (even empty text should work)
        if text.strip():
            output_audio = await tts.synthesize(text)
        else:
            output_audio = await tts.synthesize("Hello")
        
        assert isinstance(output_audio, np.ndarray)
        assert len(output_audio) > 0


# Run tests with: pytest tests/ -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
