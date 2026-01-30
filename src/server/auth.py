"""
Authentication and API key management.

Token system like Telegram Bot API:
- Users get an API key to connect their voice widget
- Keys can be scoped (rate limits, features)
- Hosted version charges per minute or monthly
"""

import secrets
import hashlib
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class APIKey:
    """API key with metadata and limits."""
    key_id: str
    key_hash: str  # Store hash, not plaintext
    name: str
    created_at: datetime
    
    # Limits
    rate_limit_per_minute: int = 60  # requests per minute
    monthly_minutes: Optional[int] = None  # None = unlimited
    
    # Usage tracking
    minutes_used: float = 0.0
    last_request_at: Optional[datetime] = None
    request_count_this_minute: int = 0
    
    # Features
    features: Dict[str, bool] = field(default_factory=lambda: {
        "continuous_mode": True,
        "voice_cloning": False,
        "priority_queue": False,
    })
    
    # Status
    active: bool = True
    tier: str = "free"  # free, pro, enterprise


class TokenManager:
    """
    Manage API tokens for voice connections.
    
    In production, this would be backed by a database.
    For MVP, we use in-memory storage + env vars.
    """
    
    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        self._key_to_id: Dict[str, str] = {}  # hash -> key_id lookup
    
    def generate_key(
        self,
        name: str,
        tier: str = "free",
        rate_limit: int = 60,
        monthly_minutes: Optional[int] = None,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Returns:
            (plaintext_key, APIKey object)
            
        Note: Plaintext key is only returned once!
        """
        # Generate secure random key
        key_id = secrets.token_hex(8)
        plaintext_key = f"ocv_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(plaintext_key)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.now(tz=None),
            rate_limit_per_minute=rate_limit,
            monthly_minutes=monthly_minutes,
            tier=tier,
        )
        
        self._keys[key_id] = api_key
        self._key_to_id[key_hash] = key_id
        
        logger.info(f"Generated API key: {key_id} ({name}, tier={tier})")
        
        return plaintext_key, api_key
    
    def validate_key(self, plaintext_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return its metadata.
        
        Returns None if invalid.
        """
        if not plaintext_key or not plaintext_key.startswith("ocv_"):
            return None
        
        key_hash = self._hash_key(plaintext_key)
        key_id = self._key_to_id.get(key_hash)
        
        if not key_id:
            return None
        
        api_key = self._keys.get(key_id)
        
        if not api_key or not api_key.active:
            return None
        
        return api_key
    
    def check_rate_limit(self, api_key: APIKey) -> bool:
        """
        Check if request is within rate limits.
        
        Returns True if allowed, False if rate limited.
        """
        now = datetime.now(tz=None)
        
        # Reset counter if new minute
        if api_key.last_request_at:
            elapsed = (now - api_key.last_request_at).total_seconds()
            if elapsed >= 60:
                api_key.request_count_this_minute = 0
        
        # Check rate limit
        if api_key.request_count_this_minute >= api_key.rate_limit_per_minute:
            return False
        
        # Update counters
        api_key.request_count_this_minute += 1
        api_key.last_request_at = now
        
        return True
    
    def check_monthly_quota(self, api_key: APIKey, minutes: float = 0) -> bool:
        """
        Check if within monthly minute quota.
        
        Returns True if allowed, False if quota exceeded.
        """
        if api_key.monthly_minutes is None:
            return True  # Unlimited
        
        return (api_key.minutes_used + minutes) <= api_key.monthly_minutes
    
    def record_usage(self, api_key: APIKey, minutes: float):
        """Record minutes used for billing."""
        api_key.minutes_used += minutes
        logger.debug(f"Key {api_key.key_id}: used {minutes:.2f} min, total {api_key.minutes_used:.2f}")
    
    def get_usage(self, api_key: APIKey) -> Dict[str, Any]:
        """Get usage stats for an API key."""
        return {
            "key_id": api_key.key_id,
            "name": api_key.name,
            "tier": api_key.tier,
            "minutes_used": round(api_key.minutes_used, 2),
            "monthly_limit": api_key.monthly_minutes,
            "rate_limit": api_key.rate_limit_per_minute,
            "features": api_key.features,
        }
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].active = False
            logger.info(f"Revoked API key: {key_id}")
            return True
        return False
    
    def _hash_key(self, plaintext_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(plaintext_key.encode()).hexdigest()


# Global token manager instance
token_manager = TokenManager()


# Helper to load keys from environment
def load_keys_from_env():
    """
    Load API keys from environment variables.
    
    Format: OPENCLAW_API_KEY_<name>=<plaintext_key>
    
    For production, use a database instead.
    """
    import os
    
    # Check for master key (allows all access)
    master_key = os.getenv("OPENCLAW_MASTER_KEY")
    if master_key:
        # Register master key
        key_hash = token_manager._hash_key(master_key)
        api_key = APIKey(
            key_id="master",
            key_hash=key_hash,
            name="Master Key",
            created_at=datetime.now(tz=None),
            rate_limit_per_minute=1000,
            monthly_minutes=None,
            tier="enterprise",
        )
        api_key.features = {
            "continuous_mode": True,
            "voice_cloning": True,
            "priority_queue": True,
        }
        token_manager._keys["master"] = api_key
        token_manager._key_to_id[key_hash] = "master"
        logger.info("Loaded master API key from environment")


# Pricing tiers for hosted version
PRICING_TIERS = {
    "free": {
        "monthly_minutes": 60,
        "rate_limit": 30,
        "price": 0,
        "features": ["continuous_mode"],
    },
    "pro": {
        "monthly_minutes": 500,
        "rate_limit": 120,
        "price": 29,  # $/month
        "features": ["continuous_mode", "voice_cloning"],
    },
    "enterprise": {
        "monthly_minutes": None,  # Unlimited
        "rate_limit": 500,
        "price": 99,  # $/month
        "features": ["continuous_mode", "voice_cloning", "priority_queue"],
    },
}
