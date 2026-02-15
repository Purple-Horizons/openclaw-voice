"""
Telnyx WebRTC Phone Calling module.

Enables AI assistants to make and receive real phone calls via Telnyx WebRTC.
This integrates with ClawdTalk for voice-enabled AI phone interactions.

Requirements:
    - TELNYX_API_KEY environment variable
    - Telnyx SIP connection configured
    - aiohttp for async HTTP requests

Documentation:
    - https://developers.telnyx.com/docs/webrtc
    - https://github.com/team-telnyx/clawdtalk-client
"""

import asyncio
import json
import os
from typing import Optional, Callable, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

from loguru import logger

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class CallState(Enum):
    """Phone call states."""
    IDLE = "idle"
    CONNECTING = "connecting"
    RINGING = "ringing"
    ACTIVE = "active"
    HOLD = "hold"
    ENDED = "ended"
    ERROR = "error"


@dataclass
class CallInfo:
    """Information about a phone call."""
    call_id: str
    caller_id: Optional[str] = None
    callee_id: Optional[str] = None
    state: CallState = CallState.IDLE
    duration_seconds: float = 0.0
    direction: str = "outbound"  # inbound or outbound


class TelnyxWebRTCClient:
    """
    Telnyx WebRTC client for making and receiving phone calls.
    
    This client enables AI assistants to:
    - Make outbound phone calls to any number
    - Receive inbound calls
    - Stream audio bidirectionally
    - Use DTMF tones for IVR navigation
    
    Example usage:
        client = TelnyxWebRTCClient()
        await client.connect()
        
        call = await client.make_call("+15551234567")
        # Stream audio and handle events
        async for event in client.call_events():
            if event["type"] == "audio":
                # Process incoming audio
                pass
        
        await client.hangup()
    """
    
    TELNYX_API_URL = "https://api.telnyx.com/v2"
    TELNYX_WSS_URL = "wss://rtc.telnyx.com/ws"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        sip_username: Optional[str] = None,
        sip_password: Optional[str] = None,
        caller_id: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("TELNYX_API_KEY")
        self.sip_username = sip_username or os.environ.get("TELNYX_SIP_USERNAME")
        self.sip_password = sip_password or os.environ.get("TELNYX_SIP_PASSWORD")
        self.caller_id = caller_id or os.environ.get("TELNYX_CALLER_ID")
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._current_call: Optional[CallInfo] = None
        self._token: Optional[str] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._connected = False
        
        if not self.api_key:
            logger.warning("TELNYX_API_KEY not set - WebRTC calls will not work")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def get_webrtc_token(self) -> str:
        """
        Fetch a WebRTC token from Telnyx API.
        
        Returns:
            JWT token for WebRTC authentication
        """
        if not self.api_key:
            raise ValueError("TELNYX_API_KEY not configured")
        
        session = await self._get_session()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Request WebRTC token
        payload = {
            "ttl": 3600,  # 1 hour
        }
        
        async with session.post(
            f"{self.TELNYX_API_URL}/webrtc/tokens",
            headers=headers,
            json=payload,
        ) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"Failed to get WebRTC token: {error}")
            
            result = await response.json()
            self._token = result.get("data", {}).get("token")
            return self._token
    
    async def connect(self) -> bool:
        """
        Connect to Telnyx WebRTC gateway.
        
        Returns:
            True if connected successfully
        """
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not installed - run: pip install aiohttp")
            return False
        
        if not self._token:
            await self.get_webrtc_token()
        
        session = await self._get_session()
        
        try:
            self._ws = await session.ws_connect(
                f"{self.TELNYX_WSS_URL}?token={self._token}"
            )
            self._connected = True
            
            # Start message handler
            asyncio.create_task(self._message_handler())
            
            logger.info("Connected to Telnyx WebRTC gateway")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebRTC gateway: {e}")
            return False
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages."""
        if not self._ws:
            return
        
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_message(data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Audio data
                    await self._event_queue.put({
                        "type": "audio",
                        "data": msg.data,
                    })
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Message handler error: {e}")
    
    async def _handle_message(self, data: dict):
        """Handle parsed WebSocket message."""
        event_type = data.get("event")
        
        if event_type == "call.state":
            self._handle_call_state(data)
        elif event_type == "call.answered":
            if self._current_call:
                self._current_call.state = CallState.ACTIVE
            await self._event_queue.put({
                "type": "answered",
                "call_id": data.get("call_id"),
            })
        elif event_type == "call.hangup":
            if self._current_call:
                self._current_call.state = CallState.ENDED
            await self._event_queue.put({
                "type": "hangup",
                "call_id": data.get("call_id"),
            })
        elif event_type == "call.error":
            await self._event_queue.put({
                "type": "error",
                "message": data.get("error", {}).get("message"),
            })
    
    def _handle_call_state(self, data: dict):
        """Handle call state update."""
        if not self._current_call:
            return
        
        state_str = data.get("state", "")
        state_map = {
            "connecting": CallState.CONNECTING,
            "ringing": CallState.RINGING,
            "active": CallState.ACTIVE,
            "hold": CallState.HOLD,
            "ended": CallState.ENDED,
        }
        self._current_call.state = state_map.get(state_str, CallState.IDLE)
    
    async def make_call(
        self,
        destination: str,
        caller_id: Optional[str] = None,
    ) -> CallInfo:
        """
        Make an outbound phone call.
        
        Args:
            destination: Phone number in E.164 format (e.g., "+15551234567")
            caller_id: Optional caller ID (uses default if not set)
            
        Returns:
            CallInfo with call details
        """
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected to WebRTC gateway")
        
        caller_id = caller_id or self.caller_id
        
        # Send INVITE via WebRTC
        invite_msg = {
            "jsonrpc": "2.0",
            "method": "call",
            "params": {
                "destination_number": destination,
                "caller_id_name": caller_id or "AI Assistant",
                "caller_id_number": caller_id or "",
            },
        }
        
        await self._ws.send_json(invite_msg)
        
        # Create call info
        self._current_call = CallInfo(
            call_id="",  # Will be set on response
            caller_id=caller_id,
            callee_id=destination,
            state=CallState.CONNECTING,
            direction="outbound",
        )
        
        logger.info(f"Initiating call to {destination}")
        return self._current_call
    
    async def answer(self) -> bool:
        """Answer an incoming call."""
        if not self._ws or not self._current_call:
            return False
        
        answer_msg = {
            "jsonrpc": "2.0",
            "method": "answer",
            "params": {
                "call_id": self._current_call.call_id,
            },
        }
        
        await self._ws.send_json(answer_msg)
        self._current_call.state = CallState.ACTIVE
        return True
    
    async def hangup(self) -> bool:
        """Hang up the current call."""
        if not self._ws or not self._current_call:
            return False
        
        hangup_msg = {
            "jsonrpc": "2.0",
            "method": "hangup",
            "params": {
                "call_id": self._current_call.call_id,
            },
        }
        
        await self._ws.send_json(hangup_msg)
        self._current_call.state = CallState.ENDED
        return True
    
    async def send_audio(self, audio_data: bytes):
        """
        Send audio to the call.
        
        Args:
            audio_data: PCM audio data (16kHz, 16-bit mono)
        """
        if not self._ws or not self._connected:
            return
        
        if self._current_call and self._current_call.state == CallState.ACTIVE:
            await self._ws.send_bytes(audio_data)
    
    async def send_dtmf(self, digits: str):
        """
        Send DTMF tones during a call.
        
        Args:
            digits: DTMF digits to send (0-9, *, #)
        """
        if not self._ws or not self._current_call:
            return
        
        dtmf_msg = {
            "jsonrpc": "2.0",
            "method": "dtmf",
            "params": {
                "call_id": self._current_call.call_id,
                "digits": digits,
            },
        }
        
        await self._ws.send_json(dtmf_msg)
    
    async def call_events(self) -> AsyncGenerator[dict, None]:
        """
        Yield call events as they occur.
        
        Yields:
            Event dictionaries with 'type' and event-specific data
        """
        while self._connected:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )
                yield event
            except asyncio.TimeoutError:
                continue
    
    @property
    def current_call(self) -> Optional[CallInfo]:
        """Get the current call info."""
        return self._current_call
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to WebRTC gateway."""
        return self._connected
    
    @property
    def is_in_call(self) -> bool:
        """Check if currently in a call."""
        return self._current_call is not None and self._current_call.state == CallState.ACTIVE
    
    async def disconnect(self):
        """Disconnect from the WebRTC gateway."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False
    
    async def close(self):
        """Close all connections."""
        await self.disconnect()
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
