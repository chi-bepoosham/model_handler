FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TEMP_IMAGES_DIR=temp_images \
    MODEL_PATH=/../data-models \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk3-module \
    libatlas-base-dev \
    gfortran \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/temp_images

# Copy requirements first to leverage Docker cache
COPY pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

# Copy application code
COPY . /app/

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Command to run the application
CMD ["python", "api.py"]
