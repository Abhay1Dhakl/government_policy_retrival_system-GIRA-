# GIRA Agent Dockerfile - OPTIMIZED VERSION
# This version skips local embedding model downloads
# and uses OpenAI API for embeddings instead

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (reduced from original)
RUN apt-get update && apt-get install -y \
    # Build essentials (needed for some Python packages)
    gcc \
    g++ \
    # PostgreSQL client
    postgresql-client \
    libpq-dev \
    # Camelot dependencies (table extraction from PDFs)
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libcairo2 \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    # Utilities
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies WITHOUT heavy ML libraries if using OpenAI embeddings
RUN pip install --no-cache-dir -r requirements.txt

# Install ONLY the small spaCy model (not the large one)
RUN python -m spacy download en_core_web_sm

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads/highlighted \
    && mkdir -p logs \
    && mkdir -p MCP_server_config

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_OPENAI_EMBEDDINGS=true \
    SKIP_LOCAL_EMBEDDINGS=true

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run the application WITHOUT model prefetch (using OpenAI API)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "1"]
