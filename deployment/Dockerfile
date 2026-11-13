# Dockerfile for Higgs Event Lab Dashboard
FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt pyproject.toml ./


# Copy application source, app, documentation, and configuration
COPY src/ ./src/
COPY app/ ./app/
COPY docs/ ./docs/
COPY team/ ./team/
COPY outputs/ ./outputs/
COPY streamlit_app.py README.md ./
RUN pip install --no-cache-dir ".[dashboard]"

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
