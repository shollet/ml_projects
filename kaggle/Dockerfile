FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Utiliser le montage de cache pour pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY . .
