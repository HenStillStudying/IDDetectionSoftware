# API reference

**Main API** (`services/api`):
- `POST /v1/ktp/extract` — multipart image upload, returns extraction result synchronously.
  `504` if the OCR service times out (30s), `503` if it's unreachable or errors — the
  real cause is logged server-side only, never echoed back
- `POST /v1/ktp/jobs` — multipart image upload, enqueues the job onto Redis (via `arq`), returns a job id immediately
- `GET /v1/ktp/jobs/{id}` — poll job status (`queued`/`in_progress`/`complete`) and result once done
- `GET /health` — liveness check

`/v1/ktp/jobs` requires a separate worker process actually consuming the
queue — `arq app.worker.WorkerSettings` (see "Run" in [DEVELOPMENT.md](DEVELOPMENT.md)) — the API only
enqueues, it never runs the pipeline for async jobs itself. Job state lives
in Redis, not the API process, so it survives an API restart (verified: a
job submitted before an API restart is still pollable — with the same
result — after one).

Set `KTP_API_KEY` to require an `X-API-Key` header on the `/v1/*` routes
(unset in dev, required in any shared environment). Set `KTP_REDIS_URL` to
point at a non-default Redis (default `redis://localhost:6379`).

> **What `status: ok` does — and does not — mean.** It means *an image
> that looks like a KTP was found and its fields were read*. It does **not**
> mean the card is genuine or belongs to a real person. A fabricated card
> passes every check this system has: the synthetic cards produced by this
> project's own `training/ktp_dataset_generator.py` are detected 30/30
> (the detector was trained on nothing else), read at ~90% field accuracy,
> carry structurally valid NIKs (validation checks the NIK's *format*, not
> whether it's registered), and only draw a non-suspicious "no EXIF" note
> from the forensics layer. The same would very likely hold for a
> fabricated card that's printed and photographed.
>
> One cheap layer now catches *careless* fakes: `nik_consistency` checks
> that the birth date and gender encoded in the NIK match the ones printed
> on the same card (a genuine card can't disagree with itself). It flags
> every one of this project's own synthetic cards whose birth date OCR
> actually read — but anyone who knows the NIK format can make a fake agree
> with itself, so a pass means little on its own. Check `fields_checked`
> too: a pass on gender alone is a coin flip, not evidence. Treat the
> output as
> *extracted data to be verified*, never as *identity verification*.
> Closing that gap takes things single-image analysis can't provide: an
> authoritative registry lookup (Dukcapil), live camera capture instead of
> uploaded files, detection of the card's physical security features, and
> human review of uncertain cases — none of which exist here yet.

**OCR service** (`services/ocr`):
- `POST /v1/ocr/extract` — multipart image (an already-rectified card), returns `KtpFields` JSON
- `GET /health` — only returns `ok` once the OCR model has finished loading (warm-up runs at
  startup, not on the first request, so no caller eats a cold-start penalty)
