# ==============================================================================
# Algorithmic Crypto Volatility Forecaster - Container Image
# Multimodal Deep Learning (PyTorch + Hugging Face FinBERT) & FastAPI Dashboard
# ==============================================================================

FROM python:3.12-slim

# Avoid buffering and bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install minimal OS dependencies for compilation and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained checkpoints
COPY . .

# Expose FastAPI application port
EXPOSE 8000

# Container healthcheck against /api/status endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Default runtime command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
