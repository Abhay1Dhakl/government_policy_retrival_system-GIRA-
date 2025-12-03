# GIRA Backend Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1
ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
EXPOSE 8082

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8082"]
