FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY services/audio-service/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/install -r requirements.txt
COPY shared ./shared
RUN pip install --prefix=/install -e ./shared

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY services/audio-service/app ./app
COPY shared ./shared
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8003/internal/health || exit 1

EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
