# Base image with Python 3.9 and explicit architecture targeting for x86_64
FROM --platform=linux/amd64 python:3.9-slim-bullseye

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TEMP_IMAGES_DIR=/app/temp_images \
    MODEL_PATH=/app/data-models \
    PYTHONPATH=/app \
    OPENBLAS_CORETYPE=HASWELL

# System dependencies
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

# Install Python dependencies using binary wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary .

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

USER appuser

# Run the application
CMD ["sh", "-c", "echo 'Running on x86_64 with Python 3.9' && python api.py"]
