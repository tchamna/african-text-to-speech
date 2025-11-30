# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Install system dependencies for audio processing and ffmpeg (required by whisper)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY startup.txt .
COPY download_assets.py .

# Copy templates folder
COPY templates/ templates/

# Create directories for uploads, audio, static, and assets
RUN mkdir -p uploads audio static assets

# Note: assets/ will be downloaded from Azure Storage at runtime
# This keeps private data out of the git repository

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Startup: download assets, then run gunicorn
CMD python download_assets.py && gunicorn --bind 0.0.0.0:8000 --timeout 600 --workers 1 app:app
