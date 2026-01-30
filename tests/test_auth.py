"""
Tests for authentication module.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.server.auth import TokenManager, APIKey, PRICING_TIERS


class TestTokenManager:
    """Tests for token management."""
    
    def test_generate_key(self):
        """Test API key generation."""
        tm = TokenManager()
        plaintext, api_key = tm.generate_key("test-app")
        
        assert plaintext.startswith("ocv_")
        assert len(plaintext) > 40
        assert api_key.name == "test-app"
        assert api_key.active
    
    def test_validate_key_success(self):
        """Test validating a valid key."""
        tm = TokenManager()
        plaintext, _ = tm.generate_key("test-app")
        
        result = tm.validate_key(plaintext)
        assert result is not None
        assert result.name == "test-app"
    
    def test_validate_key_invalid(self):
        """Test validating an invalid key."""
        tm = TokenManager()
        
        assert tm.validate_key("invalid") is None
        assert tm.validate_key("ocv_invalid") is None
        assert tm.validate_key("") is None
        assert tm.validate_key(None) is None
    
    def test_rate_limit(self):
        """Test rate limiting."""
        tm = TokenManager()
        _, api_key = tm.generate_key("test", rate_limit=5)
        
        # Should allow up to rate limit
        for i in range(5):
            assert tm.check_rate_limit(api_key) is True
        
        # Should block after limit
        assert tm.check_rate_limit(api_key) is False
    
    def test_monthly_quota(self):
        """Test monthly quota checking."""
        tm = TokenManager()
        _, api_key = tm.generate_key("test", monthly_minutes=10)
        
        # Should allow within quota
        assert tm.check_monthly_quota(api_key, 5) is True
        
        # Should block over quota
        assert tm.check_monthly_quota(api_key, 15) is False
        
        # Record usage
        tm.record_usage(api_key, 8)
        assert api_key.minutes_used == 8
        
        # Now over quota
        assert tm.check_monthly_quota(api_key, 3) is False
    
    def test_unlimited_quota(self):
        """Test unlimited quota (None)."""
        tm = TokenManager()
        _, api_key = tm.generate_key("test", monthly_minutes=None)
        
        # Should always allow
        assert tm.check_monthly_quota(api_key, 10000) is True
    
    def test_revoke_key(self):
        """Test key revocation."""
        tm = TokenManager()
        plaintext, api_key = tm.generate_key("test")
        
        # Key should be valid
        assert tm.validate_key(plaintext) is not None
        
        # Revoke it
        assert tm.revoke_key(api_key.key_id) is True
        
        # Now invalid
        assert tm.validate_key(plaintext) is None
    
    def test_get_usage(self):
        """Test usage stats retrieval."""
        tm = TokenManager()
        _, api_key = tm.generate_key("test", tier="pro")
        tm.record_usage(api_key, 5.5)
        
        usage = tm.get_usage(api_key)
        
        assert usage["name"] == "test"
        assert usage["tier"] == "pro"
        assert usage["minutes_used"] == 5.5
    
    def test_tiers(self):
        """Test different pricing tiers."""
        tm = TokenManager()
        
        # Free tier
        _, free_key = tm.generate_key("free-user", tier="free", 
                                       rate_limit=30, monthly_minutes=60)
        assert free_key.tier == "free"
        assert free_key.monthly_minutes == 60
        
        # Pro tier
        _, pro_key = tm.generate_key("pro-user", tier="pro",
                                      rate_limit=120, monthly_minutes=500)
        assert pro_key.tier == "pro"
        assert pro_key.monthly_minutes == 500


class TestPricingTiers:
    """Test pricing tier configuration."""
    
    def test_tiers_exist(self):
        """Test all expected tiers exist."""
        assert "free" in PRICING_TIERS
        assert "pro" in PRICING_TIERS
        assert "enterprise" in PRICING_TIERS
    
    def test_free_tier(self):
        """Test free tier config."""
        free = PRICING_TIERS["free"]
        assert free["price"] == 0
        assert free["monthly_minutes"] == 60
    
    def test_enterprise_unlimited(self):
        """Test enterprise has unlimited minutes."""
        enterprise = PRICING_TIERS["enterprise"]
        assert enterprise["monthly_minutes"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
