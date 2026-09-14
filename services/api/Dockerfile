FROM python:3.12-slim

WORKDIR /app

# Build context must be the repo root: docker build -f services/api/Dockerfile .
COPY libs ./libs
RUN pip install --no-cache-dir ./libs

COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api/app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
