FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    git git-lfs wget curl ffmpeg \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    python3 python3-pip python-is-python3 \
    && apt-get clean

# Upgrade pip and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional: Pre-download model (or allow lazy download on first run)
RUN git lfs install && git clone https://huggingface.co/goofyai/3d_render_style_xl /app/3d_render_style_xl

# Copy source code
COPY . .

# Expose port
EXPOSE 5006

# Run using gunicorn with 1 worker and 4 threads (tune as needed)
CMD ["gunicorn", "--bind", "0.0.0.0:5006", "--workers=1", "--threads=4", "app:app"]
