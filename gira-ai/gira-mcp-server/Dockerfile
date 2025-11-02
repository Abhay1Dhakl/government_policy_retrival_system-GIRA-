# MIRA Internal MCP Server Dockerfile - OPTIMIZED VERSION
# This version uses OpenAI embeddings instead of local models

FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable to prefer OpenAI embeddings
ENV USE_OPENAI_EMBEDDINGS=true \
    SKIP_LOCAL_EMBEDDINGS=true \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run WITHOUT model prefetch (using OpenAI API)
CMD ["python", "main.py"]
