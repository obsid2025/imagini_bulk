FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for rembg/onnx
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download rembg model
RUN python -c "from rembg import remove; print('rembg ready')" || true

# Copy app files
COPY . .

# Create necessary directories
RUN mkdir -p uploads cache output fonts templates_img

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]
