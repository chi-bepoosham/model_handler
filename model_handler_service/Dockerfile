FROM nvidia/cuda:11.2.2-base-ubuntu20.04

WORKDIR /app
COPY . /app
ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /temp_images/

RUN apt-get update && apt-get install -y \
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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -e .

CMD ["python3", "api.py"]

