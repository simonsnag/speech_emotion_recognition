FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    portaudio19-dev \
    libportaudio2 alsa-utils \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p files models
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
