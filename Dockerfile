FROM python:3.9-slim

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    build-essential

COPY dev-requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r dev-requirements.txt

COPY . .

EXPOSE 8002
CMD ["uvicorn", "__main__:app", "--host", "0.0.0.0", "--port", "8002"]