# For platforms that read a Procfile (Railway, Heroku-style buildpacks).
# Railway/Fly typically inject $PORT; uvicorn binds to it.
web: uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
