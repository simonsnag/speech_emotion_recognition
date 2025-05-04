FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libportaudio2 alsa-utils \
    libasound2-dev \
    pulseaudio \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8009

CMD ["streamlit", "run", "main.py", "--browser.gatherUsageStats", "false"]