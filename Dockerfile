# Production-ready multi-stage Dockerfile for Nomic Embedding API
# Stage 1: Build dependencies (using Debian for better multi-platform support)
FROM python:3.11-slim AS builder

# Build arguments for optional features
ARG INSTALL_PERFORMANCE=false
ARG INSTALL_NVIDIA=false

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Install performance dependencies
RUN if [ "$INSTALL_PERFORMANCE" = "true" ]; then \
    echo "Installing performance dependencies..." && \
    pip install --no-cache-dir "megablocks @ git+https://github.com/nomic-ai/megablocks.git" || echo "Performance deps failed"; \
    fi

# Optional: Install NVIDIA dependencies
RUN if [ "$INSTALL_NVIDIA" = "true" ]; then \
    echo "Installing NVIDIA dependencies..." && \
    pip install --no-cache-dir pynvml==11.5.3 || echo "NVIDIA deps failed"; \
    fi

# Stage 2: Production runtime (using slim for compatibility)
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Create directories with proper permissions
RUN mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py ./

# Switch to non-root user
USER appuser

# Set application environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=INFO
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV TORCH_HOME=/app/models
ENV EMBEDDING_MODEL=nomic-768
ENV ENABLE_CACHING=true
ENV MAX_CONCURRENT_REQUESTS=50
ENV MODEL_POOL_SIZE=0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "main.py"]