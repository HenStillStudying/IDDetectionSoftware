# Build context must be the repo root: docker build -f services/ocr/Dockerfile .
#
# Layers go from least to most often changed: system libs, then every
# third-party dependency (installed from the dependency manifests alone),
# then this repo's own code last. The model-conversion stage branches off
# *before* any source code is copied in, so editing libs/ or services/ocr/
# doesn't re-download and re-convert the models either — only the final
# few small layers rebuild.
FROM python:3.12-slim AS deps

WORKDIR /app

# System libs python:3.12-slim doesn't ship: libgl1/libglib2.0-0 for the
# full (GUI) opencv-contrib-python paddlex depends on (else `libGL.so.1:
# cannot open shared object file`), and libgomp1, the OpenMP runtime
# paddlepaddle itself links against (else `libgomp.so.1` missing — only
# surfaces when the engine is first constructed, since paddleocr imports
# paddle lazily).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Every third-party dependency, read from the manifests only — no source
# code copied yet. Includes the `onnx` extra (onnxruntime, pinned in
# services/ocr/pyproject.toml).
COPY infra/pyproject_deps.py /tmp/deps/
COPY libs/pyproject.toml /tmp/deps/libs.toml
COPY services/ocr/pyproject.toml /tmp/deps/ocr.toml
COPY services/ocr/requirements.txt /tmp/deps/
RUN python /tmp/deps/pyproject_deps.py /tmp/deps/libs.toml /tmp/deps/ocr.toml > /tmp/deps/pyproject.txt \
    && pip install --no-cache-dir -r /tmp/deps/pyproject.txt -r /tmp/deps/requirements.txt \
    && rm -rf /tmp/deps


# Builder stage: download the native PP-OCRv6_small models and convert them
# to ONNX. This is the step that fails with a DLL error on Windows
# (paddle2onnx's native extension) — running it here, inside a Linux
# container, sidesteps that entirely. Only the resulting .onnx files are
# copied into the final image; paddle2onnx and the native model files
# never ship.
FROM deps AS onnx-models
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


FROM deps
COPY --from=onnx-models /onnx_models /app/onnx_models

# ONNX Runtime by default: ~3-7x faster on CPU than native PaddlePaddle with
# identical output (see README). Set KTP_OCR_ENGINE=paddle to fall back to
# the native engine — which downloads its own models on first start.
ENV KTP_OCR_ENGINE=onnxruntime \
    KTP_OCR_ONNX_DET_DIR=/app/onnx_models/PP-OCRv6_small_det \
    KTP_OCR_ONNX_REC_DIR=/app/onnx_models/PP-OCRv6_small_rec

# This repo's own packages — --no-deps, since everything they depend on is
# already installed above.
COPY libs ./libs
RUN pip install --no-cache-dir --no-deps ./libs
COPY services/ocr/ktp_ocr ./ktp_ocr
COPY services/ocr/pyproject.toml .
RUN pip install --no-cache-dir --no-deps .

COPY services/ocr/ocr_app ./ocr_app

EXPOSE 8001
CMD ["uvicorn", "ocr_app.main:app", "--host", "0.0.0.0", "--port", "8001"]
