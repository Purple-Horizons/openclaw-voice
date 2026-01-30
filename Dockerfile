# OpenClaw Voice - GPU-enabled Docker image
# Supports NVIDIA GPUs for fast Whisper + TTS inference

FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy requirements first for caching
COPY requirements.txt pyproject.toml ./

# Create venv and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e ".[stt]" && \
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY src/ ./src/
COPY .env.example ./.env.example

# Create directories for models (will be mounted or downloaded)
RUN mkdir -p /app/models /app/voices

# Environment variables
ENV OPENCLAW_HOST=0.0.0.0
ENV OPENCLAW_PORT=8765
ENV OPENCLAW_STT_MODEL=large-v3-turbo
ENV OPENCLAW_STT_DEVICE=cuda
ENV OPENCLAW_REQUIRE_AUTH=true

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8765/ || exit 1

# Run server
CMD [".venv/bin/python", "-m", "uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8765"]
