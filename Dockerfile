# Use NVIDIA CUDA base image for proper GPU support
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python and system dependencies
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
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support first
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install ONNX Runtime with GPU support for rembg
RUN pip3 install onnxruntime-gpu

# Pre-download u2net.onnx model with GPU providers
RUN python3 -c "from rembg import new_session; session = new_session('u2net', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); print('Model downloaded successfully')"

# Copy application code
COPY . /app

# Create directories for file handling
RUN mkdir -p /tmp/uploads /tmp/processed

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 5550

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5550/health || exit 1

# Use gunicorn with proper logging and GPU-friendly settings
CMD ["gunicorn", "--bind", "0.0.0.0:5550", "--workers", "1", "--threads", "4", "--timeout", "1800", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "app:app"]