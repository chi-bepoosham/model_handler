# Base image with explicit architecture targeting
FROM --platform=linux/amd64 python:3.10-slim-bullseye

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TEMP_IMAGES_DIR=/app/temp_images \
    MODEL_PATH=/app/data-models \
    PYTHONPATH=/app \
    OPENBLAS_CORETYPE=HASWELL

# System dependencies for x86_64 architecture
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff5-dev \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    libxvidcore4 \
    libx264-160 \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Application setup
WORKDIR /app
COPY . .

# Create required directories
RUN mkdir -p ${TEMP_IMAGES_DIR} && \
    mkdir -p ${MODEL_PATH}

# Python dependencies with architecture-aware installation
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --only-binary=:all: \
    numpy==1.24.3 \
    opencv-python-headless==4.8.0.76 \
    -e .

# Non-root user setup
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

USER appuser

# Startup command with architecture verification
CMD ["sh", "-c", "echo 'Running on x86_64 Architecture' && python api.py"]
