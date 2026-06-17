# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# Single image that can run either the API or the Streamlit frontend.
# Choose at runtime via the SERVICE env var: "api" (default) or "frontend".
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# System deps occasionally needed by PDF/onnx wheels; keep the layer slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cache the embedding model inside the image-friendly location.
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Persisted data + vector store live here; mount a volume to keep them.
RUN mkdir -p /app/data /app/chroma_db

EXPOSE 8000 8501

# Hosting platforms (Render/Railway/Fly) inject $PORT. Default to 8000 for the API.
ENV SERVICE=api \
    PORT=8000 \
    BACKEND_URL=http://localhost:8000

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
