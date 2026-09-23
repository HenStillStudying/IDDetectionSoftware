FROM python:3.12-slim AS base

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

# The `onnx` extra adds onnxruntime (pinned in pyproject.toml). A separate
# layer from the plain `pip install .` above, so it reuses that cached
# PaddlePaddle install instead of re-downloading it.
RUN pip install --no-cache-dir ".[onnx]"


# Builder stage: download the native PP-OCRv6_small models and convert them
# to ONNX. This is the step that fails with a DLL error on Windows
# (paddle2onnx's native extension) — running it here, inside a Linux
# container, sidesteps that entirely. Only the resulting .onnx files are
# copied into the final image; paddle2onnx and the native model files
# never ship.
FROM base AS onnx-models
RUN pip install --no-cache-dir paddle2onnx==2.0.2rc3
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_detection_model_name='PP-OCRv6_small_det', text_recognition_model_name='PP-OCRv6_small_rec', enable_mkldnn=False)"
# mkdir first: paddle2onnx creates its output dir with os.mkdir, not
# makedirs, so a missing parent dir fails the conversion.
RUN mkdir -p /onnx_models \
    && for m in PP-OCRv6_small_det PP-OCRv6_small_rec; do \
         paddlex --paddle2onnx \
           --paddle_model_dir /root/.paddlex/official_models/$m \
           --onnx_model_dir /onnx_models/$m; \
       done \
    && test -f /onnx_models/PP-OCRv6_small_det/inference.onnx \
    && test -f /onnx_models/PP-OCRv6_small_rec/inference.onnx


FROM base
COPY --from=onnx-models /onnx_models /app/onnx_models

# ONNX Runtime by default: ~3-7x faster on CPU than native PaddlePaddle with
# identical output (see README). Set KTP_OCR_ENGINE=paddle to fall back to
# the native engine — which downloads its own models on first start.
ENV KTP_OCR_ENGINE=onnxruntime \
    KTP_OCR_ONNX_DET_DIR=/app/onnx_models/PP-OCRv6_small_det \
    KTP_OCR_ONNX_REC_DIR=/app/onnx_models/PP-OCRv6_small_rec

COPY services/ocr/ocr_app ./ocr_app

EXPOSE 8001
CMD ["uvicorn", "ocr_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
