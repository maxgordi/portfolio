FROM mcr.microsoft.com/playwright/python:v1.48.0-noble
ENV PYTHONUNBUFFERED=1 TZ=Europe/Kyiv
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
COPY app ./app
COPY tools ./tools
COPY config.yaml .
CMD ["python3", "-m", "app.main"]
