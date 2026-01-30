#!/usr/bin/env python3
"""
Download Whisper models for offline use.

Usage:
    python scripts/download_models.py [model_name]
    
Models:
    tiny    - 39M params, ~1GB VRAM (fastest)
    base    - 74M params, ~1GB VRAM
    small   - 244M params, ~2GB VRAM
    medium  - 769M params, ~5GB VRAM
    large-v3-turbo - 809M params, ~6GB VRAM (best quality/speed)
"""

import sys
import os


def download_model(model_name: str = "base"):
    """Download a Whisper model."""
    print(f"Downloading Whisper model: {model_name}")
    print("This may take a few minutes...")
    
    try:
        from faster_whisper import WhisperModel
        
        # This will download the model if not cached
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        
        print(f"✅ Model '{model_name}' downloaded successfully!")
        print(f"   Cached at: ~/.cache/huggingface/")
        
        # Test the model
        import numpy as np
        audio = np.zeros(16000, dtype=np.float32)
        segments, info = model.transcribe(audio)
        list(segments)  # Consume generator
        
        print(f"✅ Model tested successfully!")
        
    except ImportError:
        print("❌ faster-whisper not installed. Run:")
        print("   pip install faster-whisper")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def list_models():
    """List available models."""
    models = {
        "tiny": "39M params, ~1GB VRAM, fastest",
        "base": "74M params, ~1GB VRAM, good balance",
        "small": "244M params, ~2GB VRAM",
        "medium": "769M params, ~5GB VRAM",
        "large-v3": "1.5B params, ~10GB VRAM, best quality",
        "large-v3-turbo": "809M params, ~6GB VRAM, best quality/speed ratio",
    }
    
    print("Available Whisper models:")
    print()
    for name, desc in models.items():
        print(f"  {name:20} - {desc}")
    print()
    print("Recommended: large-v3-turbo (GPU) or base (CPU)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_models()
        print()
        model = input("Enter model name to download (or 'q' to quit): ").strip()
        if model.lower() == 'q':
            sys.exit(0)
    else:
        model = sys.argv[1]
        
        if model in ["-h", "--help", "help"]:
            list_models()
            sys.exit(0)
    
    download_model(model)
