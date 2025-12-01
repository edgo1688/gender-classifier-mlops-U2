FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create model directory
RUN mkdir -p /app/model

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/gender_googlenet.onnx
ENV MODEL_URL=https://samlopsicesiu2.blob.core.windows.net/models/gender_googlenet.onnx

# Download model at build time (alternative: download at runtime in app)
RUN python -c "import requests; import os; \
    os.makedirs(os.path.dirname(os.getenv('MODEL_PATH')), exist_ok=True); \
    response = requests.get(os.getenv('MODEL_URL'), stream=True); \
    response.raise_for_status(); \
    with open(os.getenv('MODEL_PATH'), 'wb') as f: \
        for chunk in response.iter_content(chunk_size=8192): \
            f.write(chunk); \
    print('Model downloaded successfully')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
