#!/usr/bin/env python3
"""
Generate a secure master API key for OpenClaw Voice.

Usage:
    python scripts/generate_master_key.py
    
Add the output to your .env file or deployment environment.
"""

import secrets


def generate_master_key() -> str:
    """Generate a secure master key."""
    return f"ocv_master_{secrets.token_urlsafe(32)}"


if __name__ == "__main__":
    key = generate_master_key()
    
    print("=" * 60)
    print("OpenClaw Voice Master Key")
    print("=" * 60)
    print()
    print(f"OPENCLAW_MASTER_KEY={key}")
    print()
    print("Add this to your .env file or deployment environment.")
    print("Keep this key SECRET - it has full admin access.")
    print("=" * 60)
