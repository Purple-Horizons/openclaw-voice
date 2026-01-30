"""
Streaming response utilities.

Enables lower perceived latency by:
1. Streaming AI responses sentence-by-sentence
2. Starting TTS while AI is still generating
3. Sending audio chunks as they're ready
"""

import asyncio
import re
from typing import AsyncGenerator, Optional
from loguru import logger


async def stream_sentences(text: str) -> AsyncGenerator[str, None]:
    """
    Split text into sentences for streaming.
    
    Yields sentences as they're "ready" (simulated for non-streaming backends).
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            yield sentence


async def stream_openai_response(
    client,
    messages: list,
    model: str = "gpt-4o-mini",
) -> AsyncGenerator[str, None]:
    """
    Stream OpenAI response chunk by chunk.
    
    Yields text as it arrives from the API.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            stream=True,
        )
        
        buffer = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                buffer += text
                
                # Yield complete sentences
                while True:
                    match = re.search(r'^(.*?[.!?])\s*', buffer)
                    if match:
                        sentence = match.group(1)
                        buffer = buffer[match.end():]
                        yield sentence
                    else:
                        break
        
        # Yield any remaining text
        if buffer.strip():
            yield buffer.strip()
            
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield "Sorry, I had trouble processing that."


class StreamingTTS:
    """
    Wrapper for TTS that supports streaming output.
    
    For models that don't support native streaming,
    we synthesize sentence-by-sentence.
    """
    
    def __init__(self, tts):
        self.tts = tts
    
    async def synthesize_streaming(
        self,
        text_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize audio from a stream of text chunks.
        
        Yields audio bytes as each chunk is ready.
        """
        async for sentence in text_stream:
            if sentence.strip():
                logger.debug(f"Synthesizing: {sentence[:50]}...")
                audio = await self.tts.synthesize(sentence)
                yield audio.tobytes()


async def process_with_streaming(
    transcript: str,
    backend,
    tts,
    websocket,
) -> None:
    """
    Process user input with streaming responses.
    
    Flow:
    1. Send transcript to AI (streaming)
    2. As sentences arrive, synthesize TTS
    3. Send audio chunks to client immediately
    """
    import base64
    import json
    
    full_response = ""
    
    # Check if backend supports streaming
    if hasattr(backend, '_client') and backend._client:
        # Stream from OpenAI
        messages = [
            {"role": "system", "content": backend.system_prompt},
            *backend.conversation_history[-10:],
            {"role": "user", "content": transcript},
        ]
        
        sentence_buffer = ""
        
        async for chunk in stream_openai_response(
            backend._client, 
            messages, 
            backend.model
        ):
            full_response += chunk + " "
            
            # Send text chunk to client
            await websocket.send_json({
                "type": "response_chunk",
                "text": chunk,
            })
            
            # Synthesize and send audio
            audio = await tts.synthesize(chunk)
            audio_b64 = base64.b64encode(audio.tobytes()).decode()
            
            await websocket.send_json({
                "type": "audio_chunk",
                "data": audio_b64,
                "sample_rate": 24000,
            })
        
        # Update conversation history
        backend.conversation_history.append({"role": "user", "content": transcript})
        backend.conversation_history.append({"role": "assistant", "content": full_response.strip()})
        
        # Send completion signal
        await websocket.send_json({
            "type": "response_complete",
            "text": full_response.strip(),
        })
    
    else:
        # Fallback to non-streaming
        response = await backend.chat(transcript)
        
        await websocket.send_json({
            "type": "response_text",
            "text": response,
        })
        
        audio = await tts.synthesize(response)
        audio_b64 = base64.b64encode(audio.tobytes()).decode()
        
        await websocket.send_json({
            "type": "audio_response",
            "data": audio_b64,
            "sample_rate": 24000,
            "text": response,
        })
