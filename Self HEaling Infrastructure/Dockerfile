# SelfHealing Sentinel - Multi-stage Dockerfile
# This Dockerfile builds all components of the SelfHealing Sentinel system

# Base image with Python dependencies
FROM python:3.9-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/config /app/logs /app/models /app/python/api /app/python/notification /app/python/demo

# Common stage with shared files
FROM base AS common
COPY config/*.yaml /app/config/
COPY *.py /app/

# Anomaly Detector stage
FROM common AS anomaly-detector
COPY python/anomaly_detection/*.py /app/python/anomaly_detection/

EXPOSE 8000
CMD ["python", "/app/anomaly_detection.py", "--port", "8000"]

# Remediation Orchestrator stage
FROM common AS remediation-orchestrator
COPY python/remediation/*.py /app/python/remediation/

EXPOSE 8001
CMD ["python", "/app/remediation_orchestrator.py", "--port", "8001"]

# Sentinel API stage
FROM common AS sentinel-api
COPY python/api/*.py /app/python/api/

EXPOSE 5000
CMD ["python", "/app/python/api/sentinel_api.py", "--host", "0.0.0.0", "--port", "5000"]

# Notification Service stage
FROM common AS notification-service
COPY python/notification/*.py /app/python/notification/

CMD ["python", "/app/python/notification/send_alert.py"]

# Demo App stage
FROM common AS demo-app
COPY python/demo/*.py /app/python/demo/

EXPOSE 8080
CMD ["python", "/app/python/demo/demo_app.py", "--host", "0.0.0.0", "--port", "8080"]

# Model Trainer stage
FROM common AS model-trainer
COPY python/anomaly_detection/*.py /app/python/anomaly_detection/

CMD ["python", "/app/anomaly_detection.py", "--train-only"]

# Main stage - this is the default image that includes all components
FROM common AS main
COPY python/api/*.py /app/python/api/
COPY python/notification/*.py /app/python/notification/
COPY python/demo/*.py /app/python/demo/

EXPOSE 5000 8000 8001 8080
CMD ["python", "/app/sentinel_main.py"]