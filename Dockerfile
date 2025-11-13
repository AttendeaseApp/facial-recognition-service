FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    make \
    libopenblas-dev \
    libboost-all-dev \
 && rm -rf /var/lib/apt/lists/*

COPY dev-requirements.txt .
RUN pip install --no-cache-dir -r dev-requirements.txt

COPY . .

EXPOSE 8001

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
