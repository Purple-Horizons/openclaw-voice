#!/bin/bash
# Download default Piper TTS voice model
set -e
MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

MODEL="en_GB-alan-medium"
BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.2/en/en_GB/alan/medium"

if [ -f "$MODELS_DIR/${MODEL}.onnx" ]; then
    echo "Model already exists: ${MODEL}.onnx"
    exit 0
fi

echo "Downloading Piper voice model: ${MODEL}..."
curl -L -o "$MODELS_DIR/${MODEL}.onnx" "${BASE_URL}/${MODEL}.onnx"
curl -L -o "$MODELS_DIR/${MODEL}.onnx.json" "${BASE_URL}/${MODEL}.onnx.json"
echo "✅ Model downloaded to $MODELS_DIR/"
