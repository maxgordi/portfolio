FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag rag
COPY knowledge_base knowledge_base
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
