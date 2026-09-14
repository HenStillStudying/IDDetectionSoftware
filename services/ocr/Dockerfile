FROM python:3.12-slim

WORKDIR /app

# Build context must be the repo root: docker build -f services/ocr/Dockerfile .
COPY libs ./libs
RUN pip install --no-cache-dir ./libs

COPY services/ocr/ktp_ocr ./ktp_ocr
COPY services/ocr/pyproject.toml .
RUN pip install --no-cache-dir .

COPY services/ocr/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ocr/ocr_app ./ocr_app

EXPOSE 8001
CMD ["uvicorn", "ocr_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
