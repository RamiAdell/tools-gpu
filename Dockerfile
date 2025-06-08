# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    ffmpeg \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgtk-3-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements and install PyTorch with CUDA support first
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ONNX Runtime with GPU support for rembg
RUN pip install onnxruntime-gpu

# Pre-download u2net.onnx model and configure for GPU
RUN python3 -c "from rembg import new_session; new_session('u2net')"

# Copy application code
COPY . /app

# Create directories for file handling
RUN mkdir -p /tmp/uploads /tmp/processed

# Environment variables for CUDA
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 5550

# Gunicorn configuration with increased timeout for GPU processing
CMD ["gunicorn", "--bind", "0.0.0.0:5550", "--workers", "1", "--timeout", "1800", "--access-logfile", "-", "app:app"]