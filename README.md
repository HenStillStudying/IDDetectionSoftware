# KTP Identification

Detects an Indonesian KTP (national ID card) in a photo and extracts its
fields (NIK, nama, tanggal lahir, alamat, etc.) via a detection + OCR
pipeline, exposed as a backend API service.

## Layout

```
libs/ktp_schema/     Shared contracts: field schema, NIK validation, API result schema
services/api/        FastAPI gateway: /v1/ktp/extract (sync) and /v1/ktp/jobs (async)
services/detection/  (not yet built) YOLO card detection + perspective rectification
services/ocr/        (not yet built) field-region OCR + parsing
training/            Synthetic KTP dataset generator, used to train the detection model
infra/               Dockerfiles, deployment configs
```

The API is currently wired to stub detection/OCR implementations
(`services/api/app/stub_models.py`) so the request/response contract, error
handling, and async job flow can be built and tested before real models
exist. **Do not use the stubs against real card data** — they return
placeholder/zero-confidence output.

## Local setup

```bash
pip install -e libs
pip install -r services/api/requirements-dev.txt
```

## Run

```bash
cd services/api
uvicorn app.main:app --reload
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

Architecture and API contract in place with stub models. Next: train the
detection model on `training/ktp_dataset_generator.py` output, then build
the OCR/field-parsing service against the same `DetectionService`/`OcrService`
interfaces (`services/api/app/interfaces.py`).
