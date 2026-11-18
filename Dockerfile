FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Tesseract is needed by the OCR step (step 3); installed now so the image doesn't change later.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the exact cl100k_base encoding into the image (needs network at BUILD time only) and refuse to run
# without it: the 3,000-token corpus budget must never be enforced with an estimate.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken REQUIRE_EXACT_TOKENIZER=1
RUN mkdir -p /opt/tiktoken && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
COPY app ./app
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/logs && chown -R appuser /app /opt/tiktoken
USER appuser
EXPOSE 8000
# One worker: readiness state is in-process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
