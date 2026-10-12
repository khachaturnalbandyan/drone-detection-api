# ---- Runtime image for the ONNX backend (no PyTorch -> small image) ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Run as an unprivileged user.
RUN useradd --create-home appuser && mkdir -p /srv/models && chown -R appuser /srv
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# One worker process: the model and the micro-batcher live in this process.
# Scale out with more containers rather than more workers per container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
