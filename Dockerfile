FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces expects UID 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pre-download models as root (before switching to user)
# This ensures models are cached in the image at build time
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    print('Downloading embedding models...'); \
    SentenceTransformer('allenai/scibert_scivocab_uncased'); \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    print('Downloading cross-encoder...'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
    print('All models cached successfully!')"

RUN python -c "from transformers import pipeline; \
    print('Downloading summarization models...'); \
    pipeline('summarization', model='facebook/bart-large-cnn'); \
    pipeline('summarization', model='sshleifer/distilbart-cnn-12-6'); \
    print('Summarization models cached!')"

# Copy application code with proper ownership
COPY --chown=user:user . .

# Create /tmp directories for data persistence (HF Spaces requirement)
# /data won't persist, but /tmp will during session
RUN mkdir -p /tmp/papermind_data /tmp/huggingface && \
    chown -R user:user /tmp/papermind_data /tmp/huggingface

# Switch to non-root user
USER user

# Environment variables for HF Spaces
ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/tmp/papermind_data \
    HOME=/home/user \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface/sentence-transformers \
    TORCH_HOME=/tmp/huggingface/torch

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
