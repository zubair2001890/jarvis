# JARVIS - Self-Hosted Growth Equity Copilot
# Uses Faster-Whisper for local transcription (no data leaves your server)

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper model during build (faster startup)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu')"

# Copy application code
COPY backend/ ./backend/
COPY phone-app/ ./phone-app/
COPY laptop-ui/ ./laptop-ui/
COPY playbook/ ./playbook/

WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
