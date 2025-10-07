FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY api/ ./api/
COPY models/ ./models/
COPY src/ ./src/

RUN mkdir -p data/incoming data/processed_batches results logs

ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PORT=8080

EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 api.app:app