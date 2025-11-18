# Multi-stage Dockerfile for LlamaIndex application
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --system --no-cache -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/storage /app/.cache && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser api_server.py .

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DATA_DIR=/app/data
ENV STORAGE_DIR=/app/storage
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
