"""
Alibaba Cloud qwen3 ASR modules.

- AliyunASR: compatible-mode batch transcription (REST)
- AliyunASRRealtimeFactory/Session: realtime WebSocket transcription with server VAD
"""

import asyncio
import base64
import os
import queue
import time
from typing import Optional, AsyncGenerator, List, Dict, Any

from loguru import logger


class AliyunASR:
    """Alibaba Cloud qwen3-asr-flash batch STT via compatible-mode REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "zh",
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for Aliyun ASR")
        self.language = language

    async def transcribe_stream(self, audio_generator) -> AsyncGenerator[str, None]:
        """Accumulate stream then transcribe once (batch fallback)."""
        audio_chunks = []
        async for audio_chunk in audio_generator:
            audio_chunks.append(audio_chunk)

        if audio_chunks:
            audio_data = b"".join(audio_chunks)
            result = await self.transcribe(audio_data)
            if result:
                yield result

    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: raw audio bytes (float32 PCM or int16 PCM)
        """
        import aiohttp
        import numpy as np

        # If float32 input, convert to int16 PCM.
        if len(audio_data) % 4 == 0:
            try:
                float32_data = np.frombuffer(audio_data, dtype=np.float32)
                int16_data = (float32_data * 32767).astype(np.int16)
                audio_data = int16_data.tobytes()
            except Exception as e:
                logger.warning(f"Audio conversion failed: {e}")

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data_url = f"data:audio/wav;base64,{audio_b64}"

        payload = {
            "model": "qwen3-asr-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": data_url,
                            },
                        }
                    ],
                }
            ],
            "extra_body": {
                "asr_options": {
                    "language": self.language,
                }
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        text = (
                            result.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        logger.info(f"ASR result: {text}")
                        return text

                    error = await resp.text()
                    logger.error(f"ASR error: {error}")
                    return ""
        except Exception as e:
            logger.error(f"ASR exception: {e}")
            return ""


class _RealtimeCallback:
    """DashScope realtime callback bridge -> thread-safe event queue."""

    def __init__(self, event_queue: "queue.Queue[Dict[str, Any]]"):
        self.event_queue = event_queue

    def on_open(self):
        logger.debug("Aliyun realtime ASR websocket opened")

    def on_close(self, code, msg):
        logger.debug(f"Aliyun realtime ASR websocket closed: code={code}, msg={msg}")
        self.event_queue.put({"type": "closed", "code": code, "msg": msg})

    def on_event(self, response):
        try:
            msg = response if isinstance(response, dict) else dict(response)
        except Exception:
            logger.debug(f"Unexpected realtime response type: {type(response)}")
            return

        event_type = msg.get("type", "")

        if event_type == "input_audio_buffer.speech_started":
            self.event_queue.put({"type": "speech_started"})
            return

        if event_type == "input_audio_buffer.speech_stopped":
            self.event_queue.put({"type": "speech_stopped"})
            return

        if event_type == "conversation.item.input_audio_transcription.text":
            partial = msg.get("text") or msg.get("stash") or ""
            if partial:
                self.event_queue.put({"type": "partial", "text": partial})
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            final_text = msg.get("transcript") or msg.get("text") or ""
            self.event_queue.put({"type": "final", "text": final_text})
            return


class AliyunASRRealtimeSession:
    """One realtime ASR session (one utterance) with server-side VAD."""

    def __init__(
        self,
        api_key: str,
        language: str = "zh",
        model: str = "qwen3-asr-flash-realtime",
        ws_url: Optional[str] = None,
        silence_duration_ms: int = 900,
    ):
        self.api_key = api_key
        self.language = language
        self.model = model
        self.ws_url = ws_url or os.environ.get(
            "DASHSCOPE_REALTIME_URL",
            "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        )
        self.silence_duration_ms = silence_duration_ms

        self._event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._callback = _RealtimeCallback(self._event_queue)
        self._conversation = None

        self.final_transcript = ""
        self.partial_transcript = ""
        self.saw_speech_started = False

    async def start(self):
        """Open realtime WS and configure server-VAD transcription."""
        from dashscope.audio.qwen_omni import OmniRealtimeConversation
        from dashscope.audio.qwen_omni.omni_realtime import (
            AudioFormat,
            MultiModality,
            TranscriptionParams,
        )

        self._conversation = OmniRealtimeConversation(
            model=self.model,
            callback=self._callback,
            url=self.ws_url,
            api_key=self.api_key,
        )

        # connect / update_session are sync; run in thread to avoid blocking event loop.
        await asyncio.to_thread(self._conversation.connect)

        transcription_params = TranscriptionParams(
            language=self.language,
            sample_rate=16000,
            input_audio_format="pcm",
        )

        await asyncio.to_thread(
            self._conversation.update_session,
            output_modalities=[MultiModality.TEXT],
            input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            output_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
            enable_input_audio_transcription=True,
            input_audio_transcription_model=self.model,
            enable_turn_detection=True,
            turn_detection_type="server_vad",
            prefix_padding_ms=300,
            turn_detection_threshold=0.2,
            turn_detection_silence_duration_ms=self.silence_duration_ms,
            transcription_params=transcription_params,
        )

    async def append_audio_float32(self, audio_np):
        """Send float32 mono audio chunk (16k) to realtime ASR."""
        import numpy as np

        if self._conversation is None:
            return

        if audio_np is None or len(audio_np) == 0:
            return

        audio_i16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
        audio_b64 = base64.b64encode(audio_i16.tobytes()).decode("ascii")
        self._conversation.append_audio(audio_b64)

    def drain_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break

            events.append(event)
            if event.get("type") == "speech_started":
                self.saw_speech_started = True
            if event.get("type") == "final":
                self.final_transcript = (event.get("text") or "").strip()
            elif event.get("type") == "partial":
                self.partial_transcript = (event.get("text") or "").strip()

        return events

    async def stop_and_get_transcript(self, timeout_sec: float = 3.0) -> str:
        """Finish session and wait briefly for final transcript."""
        if self._conversation is not None:
            try:
                # server_vad should auto-commit; session.finish closes out pending results.
                await asyncio.to_thread(self._conversation.end_session, 8)
            except Exception as e:
                logger.warning(f"Realtime ASR end_session warning: {e}")
                try:
                    self._conversation.close()
                except Exception:
                    pass

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            self.drain_events()
            if self.final_transcript:
                break
            await asyncio.sleep(0.05)

        self.drain_events()
        return self.final_transcript or self.partial_transcript or ""


class AliyunASRRealtimeFactory:
    """Factory for per-utterance realtime ASR sessions, with batch fallback."""

    supports_realtime_vad = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "zh",
        model: str = "qwen3-asr-flash-realtime",
        silence_duration_ms: int = 900,
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for Aliyun ASR realtime")
        self.language = language
        self.model = model
        self.silence_duration_ms = silence_duration_ms
        self._batch_fallback = AliyunASR(api_key=self.api_key, language=self.language)

    def create_session(self) -> AliyunASRRealtimeSession:
        return AliyunASRRealtimeSession(
            api_key=self.api_key,
            language=self.language,
            model=self.model,
            silence_duration_ms=self.silence_duration_ms,
        )

    async def transcribe(self, audio_data: bytes) -> str:
        """Batch fallback if realtime transcript is empty."""
        return await self._batch_fallback.transcribe(audio_data)
