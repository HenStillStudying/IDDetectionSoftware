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

# System libs python:3.12-slim doesn't ship: libgl1/libglib2.0-0 for the
# full (GUI) opencv-contrib-python paddlex depends on (else `libGL.so.1:
# cannot open shared object file`), and libgomp1, the OpenMP runtime
# paddlepaddle itself links against (else `libgomp.so.1` missing — only
# surfaces when the engine is first constructed, since paddleocr imports
# paddle lazily). Placed after the pip layers so changing it doesn't
# invalidate them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/ocr/ocr_app ./ocr_app

EXPOSE 8001
CMD ["uvicorn", "ocr_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
