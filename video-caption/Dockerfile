# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    ffmpeg \
    build-essential \
    wget \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install PyTorch with CUDA support first (matching your CUDA 12.1)
RUN pip install --no-cache-dir torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenAI Whisper
RUN pip install --no-cache-dir openai-whisper

# Copy the Flask application code into the container
COPY . /app

# Create directories
RUN mkdir -p /app/fonts /app/.cache/torch /app/.cache/whisper

# Copy font files
COPY ./fonts/ /app/fonts/

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5003
ENV PYTHONUNBUFFERED=1
ENV FONT_FOLDER=/app/fonts

# GPU-specific environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
ENV NCCL_P2P_DISABLE=1
ENV TOKENIZERS_PARALLELISM=false

# Critical: Prevent CUDA forking issues
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CUDA_LAUNCH_BLOCKING=1

# Whisper model configuration
ENV WHISPER_MODEL_SIZE=small
ENV TORCH_HOME=/app/.cache/torch

# Set proper permissions
RUN chmod -R 755 /app

# Expose the port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5003/health || exit 1

# Use Gunicorn with spawn method to avoid CUDA forking issues
# Single worker to prevent multiprocessing issues with CUDA
CMD ["python", "-c", "import multiprocessing; multiprocessing.set_start_method('spawn', force=True); import app; app.app.run(host='0.0.0.0', port=5003, threaded=True)"]