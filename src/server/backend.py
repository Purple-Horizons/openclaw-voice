"""
AI Backend module - connects to OpenAI, OpenClaw gateway, or custom backends.
"""

import asyncio
from typing import Optional, List, Dict, AsyncGenerator

from loguru import logger


class AIBackend:
    """AI backend for processing user messages."""
    
    def __init__(
        self,
        backend_type: str = "openai",
        url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        session_key: Optional[str] = None,
    ):
        self.backend_type = backend_type
        self.url = url
        self.model = model
        self.api_key = api_key
        self.session_key = session_key
        self.system_prompt = system_prompt or (
            "You are a helpful voice assistant. Keep responses concise and conversational. "
            "Aim for 1-2 sentences unless more detail is needed."
        )
        self.conversation_history: List[Dict] = []
        self._client = None
        self._setup_client()
    
    def _setup_client(self):
        """Set up the API client."""
        if self.backend_type == "openai":
            try:
                from openai import AsyncOpenAI
                extra_headers = {}
                if self.session_key:
                    extra_headers["x-openclaw-session-key"] = self.session_key
                    logger.info(f"🔗 Session routing: {self.session_key}")
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.url if self.url != "https://api.openai.com/v1" else None,
                    default_headers=extra_headers if extra_headers else None,
                )
                logger.info(f"✅ OpenAI client ready (model: {self.model})")
            except ImportError:
                logger.error("openai package not installed")
        elif self.backend_type == "openclaw":
            # OpenClaw gateway uses OpenAI-compatible API
            logger.info("OpenClaw gateway backend")
        else:
            logger.warning(f"Unknown backend type: {self.backend_type}")
    
    async def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: The user's transcribed speech
            
        Returns:
            AI response text
        """
        if self.backend_type == "openai" and self._client:
            return await self._chat_openai(user_message)
        else:
            # Fallback echo response
            return f"I heard you say: {user_message}"
    
    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Stream a response, yielding chunks as they arrive.
        
        Args:
            user_message: The user's transcribed speech
            
        Yields:
            Text chunks as they're generated
        """
        if self.backend_type == "openai" and self._client:
            async for chunk in self._chat_openai_stream(user_message):
                yield chunk
        else:
            yield f"I heard you say: {user_message}"
    
    async def _chat_openai(self, user_message: str) -> str:
        """Chat via OpenAI API."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])  # Last 10 turns
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,  # Allow longer for voice
                temperature=0.7,
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Sorry, I had trouble processing that. Could you try again?"
    
    async def _chat_openai_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream chat via OpenAI API."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])
        
        full_response = ""
        
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text
            
            # Add complete response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response,
            })
            
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield "Sorry, I had trouble processing that."
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
