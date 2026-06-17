#!/usr/bin/env bash
# Launch either the API or the Streamlit frontend from the same image.
# Controlled by the SERVICE env var.
set -euo pipefail

SERVICE="${SERVICE:-api}"
PORT="${PORT:-8000}"

if [ "$SERVICE" = "frontend" ]; then
  echo "Starting Streamlit frontend on :${PORT}"
  exec streamlit run frontend/app.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true
else
  echo "Starting FastAPI backend on :${PORT}"
  exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}"
fi
