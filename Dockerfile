# Use an official Python runtime as a parent image
# python:3.12-slim is chosen to minimize image size and attack surface
FROM python:3.12-slim

# Create a non-root user for security (EKS / Kubernetes best practices)
RUN addgroup --system appgroup && adduser --system --group appuser

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc (equivalent to python -B)
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr (equivalent to python -u)
# PYTHONPATH: Ensures the root directory is in the python path for absolute imports
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# curl is added for the HEALTHCHECK instruction
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies from requirements.txt, and add missing dependencies explicitly
# We add gunicorn for production-ready serving instead of the default Flask development server
RUN pip install --no-cache-dir -r requirements.txt \
    flask \
    flask-cors \
    requests \
    openai \
    gunicorn

# Copy the rest of the application code
COPY . /app/

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5001

# Add a HEALTHCHECK for EKS / Jenkins / ECS compatibility
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5001/get-portfolio || exit 1

# Command to run the application using Gunicorn (production server)
# 4 workers, binding to 0.0.0.0:5001, loading 'app' from 'app/server.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "app.server:app"]
