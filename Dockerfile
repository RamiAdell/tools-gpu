# Use a minimal Python image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    ffmpeg \
    libgl1 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download u2net.onnx model
RUN python3 -c "from rembg import new_session; new_session('u2net')"

# Copy application code
COPY . /app

# Create directories for file handling
RUN mkdir -p /tmp/uploads /tmp/processed

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5550

# Gunicorn configuration
CMD ["gunicorn", "--bind", "0.0.0.0:5550", "--workers", "2", "--timeout", "1200", "--access-logfile", "-", "app:app"]