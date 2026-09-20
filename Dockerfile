# Python 3.12 slim image
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False \
    VECTORSTORE_DIR=/tmp/vectorstore

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock* ./
RUN uv pip install --system -e .

# Bake the ~80 MB all-MiniLM-L6-v2 ONNX model into the image at BUILD time.
# Without this the very first user click pays for the download, which is the
# single biggest cause of the "Find & Recommend Journals" button spinning forever.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]