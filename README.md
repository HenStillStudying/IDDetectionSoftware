# KTP Identification

Detects an Indonesian KTP (national ID card) in a photo and extracts its
fields (NIK, nama, tanggal lahir, alamat, etc.) via a detection + OCR
pipeline, exposed as a backend API service.

## Layout

```
libs/ktp_schema/      Shared contracts: field schema, NIK validation, reference data, API result schema
libs/ktp_interfaces/  DetectionService/OcrService contracts (implemented by services/detection and services/ocr)
services/api/         FastAPI gateway: /v1/ktp/extract (sync) and /v1/ktp/jobs (async)
services/detection/   YOLOv8 card detection + contour-based deskew (YoloDetectionService)
services/ocr/         PaddleOCR field extraction via label-matching (PaddleOcrService)
training/             Synthetic KTP dataset generator + YOLO training script
infra/                Dockerfiles, deployment configs
```

The API falls back to stub detection/OCR implementations
(`services/api/app/stub_models.py`) whenever the real ones aren't configured
(no trained weights, or `KTP_OCR_ENABLED` unset) — this keeps the service
bootable in dev/CI without either heavy dependency. **Never trust stub
output against real card data** — it returns placeholder/zero-confidence
fields by design.

## Local setup

```bash
pip install -e libs
pip install -r services/api/requirements-dev.txt

# optional — real models instead of stubs
pip install -e services/detection
pip install -e services/ocr
```

## Run

```bash
cd services/api
# add these to use real models instead of stubs:
#   KTP_DETECTION_WEIGHTS=../../training/runs/ktp_detector/weights/best.pt
#   KTP_OCR_ENABLED=true
uvicorn app.main:app --reload
```

## Train the detection model

```bash
cd training
python ktp_dataset_generator.py --count 300 --output ./dataset --aug-factor 2
python train_detector.py --data ./dataset/ktp.yaml --epochs 40
```

## Test

```bash
python -m pytest libs/tests services/api/tests
```

## API

- `POST /v1/ktp/extract` — multipart image upload, returns extraction result synchronously
- `POST /v1/ktp/jobs` — multipart image upload, returns a job id
- `GET /v1/ktp/jobs/{id}` — poll job status/result
- `GET /health` — liveness check

Set `KTP_API_KEY` to require an `X-API-Key` header on the `/v1/*` routes
(unset in dev, required in any shared environment).

## Status

End-to-end pipeline works: YOLOv8n detector (trained on synthetic data,
mAP50 0.995 — validates the plumbing, not real-world accuracy) → contour-based
deskew → PaddleOCR label-matching field extraction. Verified against
generated test images: 17/17 fields correct on a clean/lightly-rotated card;
graceful degradation (fields return `None`/0-confidence rather than silently
wrong values) on heavily perspective-warped, shadow-augmented ones.

**Known gaps to address before any real-world use:**
- Trained purely on synthetic data — hasn't seen a real photo yet. Needs
  real (or far more realistic) KTP images before it's trustworthy.
- Detection only corrects in-plane rotation, not true perspective distortion
  (a steeply-angled photo needs a 4-corner keypoint model).
- OCR latency is ~12s/request on CPU with MKL-DNN disabled (down from ~15s
  after skipping the doc-orientation/unwarping/textline-orientation stages,
  which are redundant since we deskew upstream — see
  `services/ocr/ktp_ocr/paddle_ocr_service.py`). MKL-DNN is disabled as a
  workaround for a PaddlePaddle/oneDNN crash on this dev machine. GPU was
  tried (`paddlepaddle-gpu`) and does work in isolation, but crashes with
  Windows DLL conflicts when PaddlePaddle and PyTorch (used by
  `ktp_detection`) are loaded in the same process — a real blocker for the
  current in-process wiring in `services/api/app/main.py`, not something to
  paper over. Reverted to CPU for reliability. Next real levers: an older
  pre-3.x paddlepaddle that might restore working MKL-DNN, lighter
  "mobile" det/rec models, or actually splitting detection and OCR into
  separate processes (which the package structure already supports) so GPU
  PaddleOCR isn't sharing a process with GPU PyTorch. Not production-viable
  latency yet either way.
