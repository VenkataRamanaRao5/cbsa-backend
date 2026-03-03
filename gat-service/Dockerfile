# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Azure App Service sets PORT env variable; default to 8001
# WORKERS controls the number of uvicorn worker processes (default: 4)
ENV PORT=8001
ENV WORKERS=4

EXPOSE 8001

# Start the application (module name is 'main', not 'app.main')
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
