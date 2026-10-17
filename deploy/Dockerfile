# Образ для повторения замеров и запуска stt-streaming где угодно.
#   docker build -f deploy/Dockerfile -t stt-lab .
#   docker run --rm -p 8090:8090 -e STT_MAX_STREAMS=4 -v hf:/root/.cache/huggingface stt-lab
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 STT_MAX_STREAMS=4 STT_DECODER=greedy
EXPOSE 8090
CMD ["uvicorn", "service.app:main", "--factory", "--host", "0.0.0.0", "--port", "8090"]
