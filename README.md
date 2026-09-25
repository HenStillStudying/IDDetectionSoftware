# KTP Identification

Detects an Indonesian KTP (national ID card) in a photo and extracts its fields
(NIK, nama, tanggal lahir, alamat, etc.) via a detection + OCR pipeline, exposed
as a backend API service.

> **This is data extraction, not identity verification.** A `status: ok`
> result means a KTP-looking card was found and read. It does not mean the card
> is genuine. See [Known issues](docs/KNOWN_ISSUES.md#trust-what-the-output-does-not-prove).

## How it works

```
photo ──► api (FastAPI) ──► YOLOv8n detector ──► deskew/crop ──► ocr service ──► fields
              │                (in-process)                     (PaddleOCR, ONNX)
              ├─ NIK format validation + NIK self-consistency (birth date, gender)
              ├─ EXIF forensics note
              └─ async jobs: Redis + arq worker
```

- **`services/api`** handles uploads, detection, validation, and the job queue.
  It also serves a `/demo` page.
- **`services/ocr`** runs PaddleOCR as its own service, using the ONNX Runtime
  engine in Docker. It is split out for dependency isolation: paddle and torch
  clash in one process.
- **`libs/`** holds the shared schema (the result contract, NIK rules) and the
  service interfaces.
- **`training/`** holds the synthetic KTP generator, the detector training, and
  the accuracy evaluator.

## Current state

Working end to end in docker-compose (redis, ocr, api, worker). Not deployed
anywhere yet.

**Pipeline and accuracy**

| | |
|---|---|
| Detection | 30/30 synthetic cards; detector mAP50 0.995 (synthetic data) |
| Field accuracy | **458/510 (89.8%)** on 30 synthetic cards; **NIK 30/30** |
| Real card | 17/17 fields on the one real KTP tested; a real SIM was correctly rejected |
| Latency | ~1.2–1.6 s per request on CPU (Docker, ONNX); the first request after startup is slower |

**Robustness**

- Handles EXIF-rotated phone photos and HEIC (iPhone) photos.
- Rejects oversized uploads cleanly (10 MB, 413).
- Returns 503 or 504, not a 500, when OCR is down or slow.

**Operations**

| | |
|---|---|
| Memory | ~0.6 GB idle, ~1.3 GB under load |
| Tests | 112 tests. CI (GitHub Actions) runs them plus a compose config check on every push. Dependabot keeps dependencies current. |

<details>
<summary>Per-field accuracy (30 synthetic cards)</summary>

| Field | Correct | Field | Correct | Field | Correct |
|---|---|---|---|---|---|
| nik | 30 | jenis_kelamin | 30 | status_perkawinan | 30 |
| pekerjaan | 30 | kecamatan | 29 | agama | 29 |
| tempat_lahir | 28 | tanggal_lahir | 28 | golongan_darah | 28 |
| provinsi | 28 | rt_rw | 27 | kelurahan_desa | 26 |
| nama | 25 | kewarganegaraan | 25 | berlaku_hingga | 25 |
| kota_kabupaten | 24 | alamat | 16 | | |

</details>

**Biggest open gaps:**

- The system has been validated on only one real card.
- There is no anti-forgery beyond a NIK self-consistency check.
- Steeply angled photos aren't detected.
- Two cards in one photo give an arbitrary pick.
- There is no HTTPS, monitoring, or compliance work yet.

The full list is in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

## Quick start

Requires Docker and trained detector weights at
`training/runs/ktp_detector/weights/best.pt`. See
[Development](docs/DEVELOPMENT.md#train-the-detection-model) to train them.

```bash
cp .env.example .env   # set REDIS_PASSWORD at minimum
docker compose up --build
curl -F "file=@training/sample_ktp.png" http://127.0.0.1:8000/v1/ktp/extract
```

Or open `http://127.0.0.1:8000/demo` in a browser. The demo only works with
`KTP_API_KEY` unset.

## API at a glance

| Endpoint | Purpose |
|---|---|
| `POST /v1/ktp/extract` | Upload an image and get the extraction result synchronously |
| `POST /v1/ktp/jobs` | Upload an image, queue it, and get a job id |
| `GET /v1/ktp/jobs/{id}` | Poll a job's status and result |
| `GET /health` | Liveness check |

Set `KTP_API_KEY` to require an `X-API-Key` header. The details, error codes,
and what each status means are in [`docs/API.md`](docs/API.md).

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) | Open limitations and known issues |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | Detailed log of every fix, change, and measurement, with evidence |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Layout, local setup, running without Docker, training, tests, CI, evaluation |
| [`docs/API.md`](docs/API.md) | Full API reference, including the `status: ok` caveat |
| [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) | Data handling audit and what UU PDP compliance would require |
| [`infra/DEPLOY.md`](infra/DEPLOY.md) | VPS deployment runbook |
