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
training/              Synthetic KTP dataset generator, YOLO training script, and a ground-truth pipeline evaluator
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

## Evaluate real pipeline accuracy (not eyeballed)

Unit tests check logic; this measures the actual detector+OCR pipeline
against known-correct field values, since a rendered card's true field
values normally aren't recoverable after the fact (the dataset generator
only keeps images + bbox labels for detection training, not per-field
ground truth):

```bash
cd training
python generate_eval_set.py --count 30 --output ./eval_set   # renders cards + saves the true field values alongside each image
python evaluate_pipeline.py --eval-dir ./eval_set --weights ./runs/ktp_detector/weights/best.pt
```

Runs the same `KtpExtractionPipeline` `services/api` uses (not a
reimplementation), compares every extracted field to ground truth, and
reports per-field accuracy plus a full per-image JSON report. This is what
actually caught the three bugs listed below — worth re-running after any
change to detection or OCR, not just once.

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
label-matching field extraction.

**Measured, not eyeballed**: `evaluate_pipeline.py` against 30
ground-truth-labeled synthetic cards (see "Evaluate real pipeline accuracy"
above) — card detected on 30/30. Per-field accuracy (exact match after
normalizing case/whitespace, so this is a strict floor, not credit for
"close enough"):

| Field | Acc. | Field | Acc. | Field | Acc. |
|---|---|---|---|---|---|
| nik | **100%** | jenis_kelamin | 100% | status_perkawinan | 100% |
| pekerjaan | 100% | agama | 97% | provinsi | 97% |
| kewarganegaraan | 90% | tanggal_lahir | 90% | kecamatan | 90% |
| nama | 73% | tempat_lahir | 73% | golongan_darah | 73% |
| kota_kabupaten | 73% | kelurahan_desa | 67% | berlaku_hingga | 70% |
| rt_rw | 60% | alamat | 53% | | |

NIK — the field that actually matters most — is perfect across all 30.
The weaker fields (`alamat`, `rt_rw`, `berlaku_hingga`) are, on inspection
of every failure, now single-character OCR noise (`0`/`Q`/`D` digit
confusion, dropped letters) rather than logic bugs — a direct, accepted
consequence of choosing the faster `PP-OCRv6_small` model tier. Building
this evaluator immediately paid for itself: it caught three real bugs a
handful of eyeballed test images had missed — a `find_value_line` geometry
off-by-one, a `find_berlaku_hingga` false-positive on short unrelated
words ("GG" and "TENGAH" both scored high enough against "hingga"), and a
`split_rt_rw_kelurahan` regex that broke when OCR dropped a separator —
each fixed and each covered by a regression test.

**Known gaps to address before any real-world use:**
- Trained purely on synthetic data — hasn't seen a real photo yet. Needs
  real (or far more realistic) KTP images before it's trustworthy.
- **Perspective correction, not just rotation, is now implemented** —
  `YoloDetectionService` finds the card's 4 corners (contour + `approxPolyDP`)
  and applies a proper `warpPerspective`, falling back to the old
  rotation-only correction when a clean quadrilateral can't be resolved.
  Verified: a clean/rotated test card now comes out perfectly flat (previously
  just reduced tilt); a heavily perspective-warped + blurred/noisy one is
  meaningfully improved but not perfectly flat — classical CV corner-finding
  gets less precise once other degradations stack on top, which is the known
  limit of this approach (a learned 4-corner keypoint model would be the next
  step up if this proves insufficient on real photos).
  Getting here surfaced and fixed a real dataset-generation bug along the
  way: `ktp_dataset_generator.py`'s `compose_scene` used to rotate the card
  with a solid black `fillcolor`, which baked an artificial axis-aligned
  black square around every tilted card — classical contour-detection was
  finding *that* square instead of the card's true edge. Fixed by rotating
  in RGBA and pasting with the alpha channel as the mask, so the real scene
  background shows through the tilted corners instead (also just makes the
  synthetic data more realistic). Dataset regenerated, detector retrained
  (mAP50 0.995, mAP50-95 0.983).
  Verification against the tighter crop this produced then surfaced and
  fixed three more real bugs: `find_value_line` required a strictly
  positive vertical gap between a label and its value, which incorrectly
  skipped a same-row value (OCR sometimes gives a short label and its
  taller value box the same y1) and matched the *next* field's label
  instead; `find_berlaku_hingga` required an exact `"hingga"` substring,
  missing it when OCR mangled it further (e.g. "Hingga" → "Hnga"), now
  fuzzy-matched per word; and the perspective warp was cropping a few
  pixels too tight at the card's edges, clipping the top header row
  (`provinsi`) entirely — fixed by expanding the detected quadrilateral
  outward by a small margin before warping. All covered by regression
  tests (`services/ocr/tests/test_field_labels.py`,
  `test_parsing.py`).
- **OCR latency is ~3-4s/request, CPU only** (down from ~12-14s). Two CPU-only
  optimizations got it there: skipping PaddleOCR's redundant doc-orientation/
  unwarping/textline-orientation stages (already deskewed upstream), and
  switching from `PP-OCRv6_medium_*` to `PP-OCRv6_small_*` models — chosen
  over the even-faster `tiny` tier because `tiny` dropped NIK entirely on a
  hard test image and started merging adjacent text lines into single
  detection boxes, undermining the position-based field matching this
  service relies on (`small` kept NIK correct across every test image tried).
  Switching tiers surfaced two real parsing regressions (a misread comma
  breaking the tempat/tanggal-lahir split, a missing colon leaking label
  text into `berlaku_hingga`), both now fixed and covered by regression
  tests in `services/ocr/tests/test_parsing.py`.
  GPU (`paddlepaddle-gpu`) was tried twice: once diagnosed as a same-process
  DLL collision with PyTorch (which motivated the service split above), then
  retried standalone in the now-isolated OCR service — and it *still* failed
  the same way (`WinError 127` loading `cudnn_engines_precompiled64_9.dll`),
  even with no PyTorch anywhere in that process. The real cause turned out
  to be the `zlibwapi.dll` NVIDIA's own docs point to for cuDNN-on-Windows: a
  decade-old zlib 1.2.3 build that's likely missing exports modern cuDNN 9.5
  expects — no modern replacement was found (conda-forge's zlib doesn't ship
  that DLL name/ABI at all). Reverted to CPU for reliability. Remaining
  levers if more speed is needed: an older pre-3.x paddlepaddle that might
  restore working MKL-DNN on CPU (real risk: probably forces a paddleocr 2.x
  downgrade too, a different API), or `PP-OCRv6_tiny_*` if its accuracy risks
  turn out to be acceptable for a given deployment. Still worth routing
  through `/v1/ktp/jobs` (async), not the sync endpoint, in production.
