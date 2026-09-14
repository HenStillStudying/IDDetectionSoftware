# KTP Identification

Detects an Indonesian KTP (national ID card) in a photo and extracts its
fields (NIK, nama, tanggal lahir, alamat, etc.) via a detection + OCR
pipeline, exposed as a backend API service.

## Layout

```
libs/ktp_schema/      Shared contracts: field schema, NIK validation, reference data, API result schema
libs/ktp_interfaces/  DetectionService/OcrService contracts (implemented by services/detection and services/ocr)
services/api/         FastAPI gateway: /v1/ktp/extract (sync) and /v1/ktp/jobs (async)
services/detection/   YOLOv8 card detection + contour-based deskew (YoloDetectionService), in-process
services/ocr/         Standalone PaddleOCR microservice — its own FastAPI app, its own process
training/              Synthetic KTP dataset generator + YOLO training script
infra/                 Dockerfiles, deployment configs
```

**Detection runs in-process** inside the API service (it's cheap — a YOLOv8n
forward pass is milliseconds). **OCR runs as a separate service** that the
API calls over HTTP (`RemoteOcrService` → `services/ocr`'s
`POST /v1/ocr/extract`), not imported directly — see "Why OCR is a separate
service" below for why that split is load-bearing, not just tidiness.

The API falls back to stub detection/OCR (`services/api/app/stub_models.py`)
whenever the real ones aren't configured (no trained weights, or
`KTP_OCR_SERVICE_URL` unset) — this keeps it bootable in dev/CI without
either heavy dependency running. **Never trust stub output against real card
data** — it returns placeholder/zero-confidence fields by design.

## Local setup

```bash
pip install -e libs
pip install -r services/api/requirements-dev.txt
pip install -e services/detection          # optional — real detection instead of stub
pip install -e services/ocr                # optional — needed to run the OCR service at all
pip install -r services/ocr/requirements-dev.txt
```

## Run (two processes)

```bash
# terminal 1 — OCR service
cd services/ocr
uvicorn ocr_app.main:app --port 8001

# terminal 2 — API, pointed at the OCR service and the trained detector
cd services/api
KTP_DETECTION_WEIGHTS=../../training/runs/ktp_detector/weights/best.pt \
KTP_OCR_SERVICE_URL=http://127.0.0.1:8001 \
uvicorn app.main:app --reload
```

Without those two env vars the API still runs, using stubs for whichever
one is missing.

## Train the detection model

```bash
cd training
python ktp_dataset_generator.py --count 300 --output ./dataset --aug-factor 2
python train_detector.py --data ./dataset/ktp.yaml --epochs 40
```

## Test

```bash
python -m pytest libs/tests services/api/tests services/ocr/tests
```

All three suites run against stubs/fakes — none load a real model, so the
full suite runs in under a second.

## API

**Main API** (`services/api`):
- `POST /v1/ktp/extract` — multipart image upload, returns extraction result synchronously
- `POST /v1/ktp/jobs` — multipart image upload, returns a job id
- `GET /v1/ktp/jobs/{id}` — poll job status/result
- `GET /health` — liveness check

Set `KTP_API_KEY` to require an `X-API-Key` header on the `/v1/*` routes
(unset in dev, required in any shared environment).

**OCR service** (`services/ocr`):
- `POST /v1/ocr/extract` — multipart image (an already-rectified card), returns `KtpFields` JSON
- `GET /health` — only returns `ok` once the OCR model has finished loading (warm-up runs at
  startup, not on the first request, so no caller eats a cold-start penalty)

## Why OCR is a separate service

This isn't just clean separation of concerns — it's a real Windows
constraint. `paddlepaddle-gpu` and PyTorch (used by the detector) each
bundle their own private CUDA/cuDNN DLLs under identical filenames, and
having both loaded in one process is unreliable (inconsistent `WinError
127` failures, not a clean crash every time). Running OCR as its own
process sidesteps that entirely, and as a side effect lets the GPU-bound
OCR stage scale independently of the cheap detection stage — which was
always the right shape for production regardless of the Windows DLL issue.

## Status

End-to-end pipeline works, verified through the real separated services (not
just in-process function calls): YOLOv8n detector (trained on synthetic
data, mAP50 0.995 — validates the plumbing, not real-world accuracy) →
contour-based deskew → HTTP call to the OCR service → PaddleOCR
label-matching field extraction. 17/17 fields correct on a clean/lightly
-rotated test card; graceful degradation (fields return `None`/0-confidence
rather than silently wrong values) on heavily perspective-warped,
shadow-augmented ones.

**Known gaps to address before any real-world use:**
- Trained purely on synthetic data — hasn't seen a real photo yet. Needs
  real (or far more realistic) KTP images before it's trustworthy.
- Detection only corrects in-plane rotation, not true perspective distortion
  (a steeply-angled photo needs a 4-corner keypoint model).
- **OCR latency is ~14s/request over HTTP, CPU only.** Skipping PaddleOCR's
  redundant doc-orientation/unwarping/textline-orientation stages (already
  deskewed upstream) cut it from ~15s to ~12s in-process; going through the
  OCR service's HTTP layer adds a bit more. GPU (`paddlepaddle-gpu`) was
  tried twice: once diagnosed as a same-process DLL collision with PyTorch
  (which motivated the service split above), then retried standalone in the
  now-isolated OCR service — and it *still* failed the same way
  (`WinError 127` loading `cudnn_engines_precompiled64_9.dll`), even with no
  PyTorch anywhere in that process. The real cause turned out to be the
  `zlibwapi.dll` NVIDIA's own docs point to for cuDNN-on-Windows: a decade-old
  zlib 1.2.3 build that's likely missing exports modern cuDNN 9.5 expects.
  Reverted to CPU for reliability rather than ship something that fails
  non-deterministically. Next real levers: a newer/correct `zlibwapi.dll`
  build, an older pre-3.x paddlepaddle that might restore working MKL-DNN on
  CPU, or lighter "mobile" det/rec models. Not production-viable latency yet
  either way — route through `/v1/ktp/jobs` (async), not the sync endpoint,
  for anything beyond manual testing.
