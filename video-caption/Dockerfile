# Use NVIDIA CUDA base image with Python
# This provides CUDA runtime and development tools
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

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
    # Additional libraries that might be needed
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies with GPU support
# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir moviepy

# Install OpenAI Whisper with all dependencies
RUN pip install --no-cache-dir openai-whisper

# Copy the Flask application code into the container
COPY . /app

# --- Font Handling ---
# Create a directory for fonts if it doesn't exist
RUN mkdir -p /app/fonts

# Copy your font files into the /app/fonts directory in the container
# Ensure these paths are relative to the Dockerfile's location
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

# Whisper model configuration
ENV WHISPER_MODEL_SIZE=small
ENV TORCH_HOME=/app/.cache/torch

# Create cache directories
RUN mkdir -p /app/.cache/torch /app/.cache/whisper

# Set proper permissions
RUN chmod -R 755 /app

# Expose the port the app runs on
EXPOSE 5003

# Health check to ensure the service is running and GPU is accessible
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5003/health || exit 1

# Use Gunicorn for production with GPU-optimized settings
# Reduce worker count for GPU workloads to avoid memory issues
# Increase timeout for video processing tasks
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5003", \
     "--workers", "2", \
     "--threads", "1", \
     "--timeout", "3600", \
     "--keep-alive", "2", \
     "--max-requests", "50", \
     "--max-requests-jitter", "10", \
     "--preload", \
     "--worker-class", "sync", \
     "app:app"]

# Alternative: Use Flask development server (not recommended for production)
# CMD ["python", "app.py"]