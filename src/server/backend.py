"""
AI Backend module - connects to OpenAI, OpenClaw gateway, or custom backends.
"""

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
    ):
        self.backend_type = backend_type
        self.url = url
        self.model = model
        self.api_key = api_key
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

                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.url if self.url != "https://api.openai.com/v1" else None,
                )
                logger.info(f"✅ OpenAI client ready (model: {self.model})")
            except ImportError:
                logger.error("openai package not installed")
        elif self.backend_type == "openclaw":
            # OpenClaw gateway uses OpenAI-compatible API
            logger.info("OpenClaw gateway backend")
        else:
            logger.warning(f"Unknown backend type: {self.backend_type}")

    def _build_messages(self) -> List[Dict]:
        """Build request messages with system prompt + recent history."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])
        return messages

    async def _chat_openai_once(self, messages: List[Dict]) -> str:
        """Single non-stream completion call using pre-built messages."""
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"OpenAI non-stream API error: {e}")
            return ""

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
        """Chat via OpenAI API (non-streaming)."""
        self.conversation_history.append({"role": "user", "content": user_message})
        messages = self._build_messages()

        assistant_message = await self._chat_openai_once(messages)
        if not assistant_message:
            assistant_message = "Sorry, I had trouble processing that. Could you try again?"

        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    async def _chat_openai_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream chat via OpenAI API; fallback to non-stream if stream returns empty."""
        logger.info(f"[🔄 PIPELINE] 收到用户消息: {user_message}")
        logger.info(f"[🔄 PIPELINE] 目标模型: {self.model}, URL: {self.url}")

        self.conversation_history.append({"role": "user", "content": user_message})
        messages = self._build_messages()
        logger.info(f"[🔄 PIPELINE] 发送请求: model={self.model}, messages count={len(messages)}")

        full_response = ""

        try:
            logger.info("[🔄 PIPELINE] 调用 Gateway API (stream)...")
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
            )
            logger.info("[🔄 PIPELINE] Gateway 流式响应开始")

            async for chunk in stream:
                text = ""
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content

                if text:
                    full_response += text
                    logger.debug(f"[🔄 PIPELINE] 收到片段: {text[:30]}...")
                    yield text

            # Some OpenClaw gateways can return stream with no content chunks.
            if not full_response.strip():
                logger.warning("[🔄 PIPELINE] 流式响应为空，回退到非流式请求")
                fallback_text = await self._chat_openai_once(messages)
                if fallback_text:
                    full_response = fallback_text
                    yield fallback_text

            logger.info(f"[🔄 PIPELINE] AI 响应完成: {full_response[:80]}...")

        except Exception as e:
            logger.error(f"[🔄 PIPELINE] Gateway 流式调用错误: {e}")

            # Fallback to non-stream mode on stream failure.
            fallback_text = await self._chat_openai_once(messages)
            if fallback_text:
                full_response = fallback_text
                yield fallback_text
            else:
                full_response = "Sorry, I had trouble processing that."
                yield full_response

        self.conversation_history.append({"role": "assistant", "content": full_response})

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
