FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-api.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    python -m pip install -r requirements-api.txt

COPY src ./src
COPY artifacts/ml/tfidf_sentiment_pipeline.joblib ./artifacts/ml/tfidf_sentiment_pipeline.joblib
COPY artifacts/dl/distilbert_sentiment ./artifacts/dl/distilbert_sentiment
RUN mkdir -p /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthcheck', timeout=3)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
