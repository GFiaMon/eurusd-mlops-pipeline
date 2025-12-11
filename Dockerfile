# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY api/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ /app/api/

# Copy models directory
COPY models/ /app/models/

# Copy data directory (for local storage option)
COPY data/ /app/data/

# Create necessary directories
RUN mkdir -p /app/models /app/data/processed /app/data/raw

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=api/app.py
ENV PORT=8080

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/api/stats')"

# Run the application
CMD ["python", "api/app.py"]
