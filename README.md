# KTP Identification

Detects an Indonesian KTP (national ID card) in a photo and extracts its
fields (NIK, nama, tanggal lahir, alamat, etc.) via a detection + OCR
pipeline, exposed as a backend API service.

## Layout

```
libs/ktp_schema/      Shared contracts: field schema, NIK validation, reference data, API result schema
libs/ktp_interfaces/  DetectionService/OcrService contracts (implemented by services/detection and services/ocr)
services/api/         FastAPI gateway: /v1/ktp/extract (sync) and /v1/ktp/jobs (async, via Redis + app/worker.py)
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
pip install -e "services/ocr[onnx]"        # optional — only needed for KTP_OCR_ENGINE=onnxruntime
```

## Run (four processes)

```bash
# terminal 1 — Redis (backs the async job queue), password-protected —
# pick your own password, don't reuse this example anywhere real
docker run -d --name ktp-redis -p 6379:6379 redis:7-alpine \
  redis-server --requirepass "your-own-password-here"

# terminal 2 — OCR service
cd services/ocr
uvicorn ocr_app.main:app --port 8001

# terminal 3 — API, pointed at the OCR service and the trained detector
cd services/api
KTP_DETECTION_WEIGHTS=../../training/runs/ktp_detector/weights/best.pt \
KTP_OCR_SERVICE_URL=http://127.0.0.1:8001 \
KTP_REDIS_URL=redis://:your-own-password-here@localhost:6379 \
uvicorn app.main:app --reload

# terminal 4 — worker (processes /v1/ktp/jobs submissions)
cd services/api
KTP_DETECTION_WEIGHTS=../../training/runs/ktp_detector/weights/best.pt \
KTP_OCR_SERVICE_URL=http://127.0.0.1:8001 \
KTP_REDIS_URL=redis://:your-own-password-here@localhost:6379 \
arq app.worker.WorkerSettings
```

Without `KTP_DETECTION_WEIGHTS`/`KTP_OCR_SERVICE_URL` the API and worker
still run, using stubs for whichever one is missing. Without Redis running,
`/v1/ktp/extract` (sync) still works — only `/v1/ktp/jobs` (async) needs it.

`KTP_REDIS_URL` defaults to `redis://localhost:6379` (no password) if
unset — fine for a quick local check, but Redis transiently holds raw
uploaded KTP photo bytes as job data (see "Data handling & compliance"
above), so the password-protected form shown here is the one actually
meant to be used, not just an option.

Optionally, set `KTP_OCR_INTERNAL_KEY` to the same value on both terminal 2
(the OCR service) and terminal 3/4 (the API/worker, where it's read as
`KTP_OCR_INTERNAL_KEY` and passed to `RemoteOcrService`) to require a
shared secret between them. Off by default — nothing in this repo's
deployment config exposes the OCR service's port beyond localhost today —
but it has no auth of its own otherwise, so this is a free defense-in-depth
layer worth turning on once a shared network (e.g. docker-compose) is
introduced.

Optionally, add these to terminal 2 to use the ONNX Runtime OCR engine
instead of native PaddlePaddle — confirmed ~3-7x faster on CPU with zero
accuracy cost (see Status below), but opt-in since it needs a manual
conversion step first (`paddlex --install paddle2onnx` then
`paddlex --paddle2onnx --paddle_model_dir <downloaded model> --onnx_model_dir services/ocr/onnx_models/<model name>`
for each of `PP-OCRv6_small_det`/`PP-OCRv6_small_rec` — confirmed working
on Linux, fails with a DLL error on Windows, though running the
already-converted models works fine on both):
```bash
KTP_OCR_ENGINE=onnxruntime \
KTP_OCR_ONNX_DET_DIR=./onnx_models/PP-OCRv6_small_det \
KTP_OCR_ONNX_REC_DIR=./onnx_models/PP-OCRv6_small_rec \
uvicorn ocr_app.main:app --port 8001
```

### Or: docker-compose (built and verified end to end)

`docker-compose.yml` at the repo root runs all four processes as
containers — `redis`, `ocr`, `api`, and `worker` (the last two share one
image/Dockerfile, just a different command). Only `api` publishes a port
to the host; `redis` and `ocr` are reachable only from other containers on
the compose network, matching the audit's recommendation to not rely on
"nothing exposes this port" as the only control once a real network
topology exists.

```bash
cp .env.example .env   # fill in a real REDIS_PASSWORD at minimum
docker compose up --build
```

**Verified by actually running it, not just `docker compose config`:**
- Real sync extraction through the containerized API against
  `training/sample_ktp.png`: identical result to the non-Docker path
  (detection 0.962, 16/17 fields, NIK valid) — real detection + OCR, not
  stubs.
- The async path end to end (`POST /v1/ktp/jobs` → worker → result),
  which crosses every container boundary at once: Redis password auth,
  the worker, and the OCR internal key.
- Isolation checked directly rather than assumed: `ocr` (8001) and
  `redis` (6379) are unreachable from the host; from inside the network,
  `ocr` returns 401 without `X-Internal-Key` and Redis returns `NOAUTH`
  without the password.
- Images: `ktp-api:local` ~2.8 GB, `ktp-ocr:local` ~2.4 GB. The api image
  installs torch/torchvision from PyTorch's CPU-only index (pinned to the
  same versions used locally) — the default PyPI build would bundle
  several GB of CUDA libraries this CPU container never uses.
- **The ocr image runs the ONNX Runtime engine by default, with the
  models converted inside its own build** — no manual conversion step,
  no volume mount. A multi-stage Dockerfile downloads the native
  `PP-OCRv6_small` models and converts them with `paddle2onnx==2.0.2rc3`
  in a throwaway builder stage; only the two `.onnx` files are copied
  into the final image (confirmed: neither paddle2onnx nor the native
  model files ship in it). This also finally sidesteps the conversion's
  Windows-only DLL failure documented under Status: the container is
  Linux regardless of the host. Set `KTP_OCR_ENGINE: paddle` in
  `docker-compose.yml` to fall back to the native engine.
- **Latency: ~1.2-1.6s per sync request**, down from ~5.9s warm on the
  native engine in the same containers (~4x). No model download at
  container startup anymore either — the native engine fetches its
  models on first start, the baked-in ONNX models need nothing.
- **Output verified against the local ONNX path, not assumed**: the
  in-container conversion produced a recognition model 40 bytes different
  in size from the one converted earlier outside Docker (the detection
  model is byte-for-byte the same size), so rather than assume
  equivalence, the same sample card was run through both via the real API
  path. All 17 field values match exactly; 4 of 17 confidences differ by
  at most 1.2e-7 (floating-point noise, far below anything that could
  change a field or status); three container runs are identical to each
  other. (The comparison has to go through the API, not an in-process
  pipeline call: the API re-encodes the rectified card as JPEG before
  sending it to the OCR service, which alone shifts the overall
  confidence from 0.9243 to 0.9270 on this card — comparing an HTTP run
  against an in-process one would have been apples to oranges.)

- **Survives crashes and reboots**: every service has
  `restart: unless-stopped` (only an explicit `docker compose stop`/`down`
  keeps it down). Tested, not assumed: killing the api container's main
  process got it restarted by Docker with `/health` answering again within
  ~4s. Logs rotate at 3 × 10 MB per container — Docker's default
  `json-file` logging never rotates and would eventually fill a
  long-running host's disk.

**What the first real build caught** (none of it visible from
`docker compose config` alone — all four would have shipped broken):
- `services/api/Dockerfile` never installed `services/detection`, so the
  container could only ever run stub detection.
- Both images crashed on startup with missing system libraries that
  `python:3.12-slim` doesn't ship: the full (GUI) OpenCV builds pulled in
  transitively by `paddlex` and `ultralytics` need `libgl1`/
  `libglib2.0-0`, and `paddlepaddle` needs `libgomp1` (the same error
  this project hit once before in the WSL2/GPU container test). The
  `libgomp1` miss surfaced only after a first round of fixes, because
  `paddleocr` imports `paddle` lazily — a plain import check passed; only
  constructing the real engine exposed it.
- A blank `.env` entry is passed into the container as an empty string,
  and `config.py` read it with `os.getenv()` as-is — so leaving
  `KTP_API_KEY` blank (as `.env.example` ships it) *enabled* auth with an
  empty secret and 401'd every request. Fixed at the config boundary in
  both services (empty now means unset), with a regression test.
- `.dockerignore` patterns like `__pycache__/` only match at the repo root
  without a `**/` prefix — so a local test run's `__pycache__` inside
  `services/ocr/ktp_ocr/` got copied into the build, invalidated that
  `COPY` layer's cache (forcing every later pip layer to re-download), and
  would have shipped stray `.pyc` files. All patterns now use `**/`.

## Train the detection model

```bash
cd training
python ktp_dataset_generator.py --count 300 --output ./dataset --aug-factor 2
python train_detector.py --data ./dataset/ktp.yaml --epochs 40
```

`ktp_dataset_generator.py` also generates hard-negative examples by
default (`--neg-ratio 0.15`) — plain backgrounds and non-KTP,
card-shaped/text-bearing distractors, each labeled with an empty YOLO
label — so the detector learns to reject non-KTP objects, not just detect
KTPs (see "False-positive tested" under Status).

## Test

```bash
python -m pytest libs/tests services/api/tests services/ocr/tests
```

All three suites run against stubs/fakes — none load a real model, so the
full suite runs in under a second.

**CI** (`.github/workflows/ci.yml`) runs this same suite on every push to
`main` and every pull request, plus `docker compose config` to catch a
broken compose file. It deliberately installs only `libs` and the two
`requirements-dev.txt` files, not `services/ocr` or `services/detection`
as packages — those pull in PaddlePaddle and PyTorch (several GB), which
the tests never import. Verified before relying on it: the full suite
passes in a clean virtualenv with none of those libraries present. That
check also caught a real gap — `ktp_ocr/text_lines.py` imports numpy, which
had only ever arrived via the heavy `pip install -e services/ocr`, so
`services/ocr/requirements-dev.txt` now lists it explicitly.

**Dependabot** (`.github/dependabot.yml`) opens a weekly pull request for
new versions of the pinned Python dependencies, the Docker base images,
and the GitHub Actions used by CI — exact pins mean nothing updates on its
own otherwise, security fixes included. All Python updates land in one
grouped PR across every directory, since `pillow`, `numpy`, `fastapi` and
others are pinned in several files and must stay on the same version
everywhere. Two blind spots to know about:
- `torch`/`torchvision` (api Dockerfile) and `paddle2onnx` (ocr
  Dockerfile) are pinned inside `RUN pip install` lines, which Dependabot
  doesn't read — they have to be bumped by hand.
- CI never installs PaddlePaddle, PaddleOCR, PyTorch or ultralytics, so a
  green CI run on a PR that bumps one of them says nothing about it.
  Rebuild the images and rerun the docker smoke test before merging those.

Three updates are deliberately excluded, all found on Dependabot's very
first run:
- `numpy` 2.4+: `paddlex` (what `paddleocr` is built on), even at its
  latest 3.7.x, requires `numpy<2.4` — the grouped PR's `numpy` 2.4.6 bump
  made the ocr image uninstallable. CI passed on that PR anyway, since it
  never installs `paddlex`; only a real `docker build` caught it — exactly
  the blind spot described above.
- `redis` 6+: its first grouped PR bumped `redis` to 8.1.0 and failed CI's
  install step — `arq`, even at its latest release (0.28.0), requires
  `redis<6`. With grouping, that one impossible bump blocked the other
  five updates in the same PR too.
- Python base-image upgrades (it proposed `python:3.12-slim` →
  `3.14-slim`): CI passed on that PR, but only because CI runs its own
  Python 3.12 and never builds these images — PaddlePaddle and PyTorch may
  not ship wheels for 3.14 at the pinned versions. Moving Python versions
  is done by hand, together with CI's `python-version`, after a rebuild and
  smoke test.

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
- `POST /v1/ktp/jobs` — multipart image upload, enqueues the job onto Redis (via `arq`), returns a job id immediately
- `GET /v1/ktp/jobs/{id}` — poll job status (`queued`/`in_progress`/`complete`) and result once done
- `GET /health` — liveness check

`/v1/ktp/jobs` requires a separate worker process actually consuming the
queue — `arq app.worker.WorkerSettings` (see "Run" above) — the API only
enqueues, it never runs the pipeline for async jobs itself. Job state lives
in Redis, not the API process, so it survives an API restart (verified: a
job submitted before an API restart is still pollable — with the same
result — after one).

Set `KTP_API_KEY` to require an `X-API-Key` header on the `/v1/*` routes
(unset in dev, required in any shared environment). Set `KTP_REDIS_URL` to
point at a non-default Redis (default `redis://localhost:6379`).

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

## Data handling & compliance (scoped, not yet built)

This system's whole purpose is extracting sensitive personal data from a
government ID — nearly every field on a KTP counts as personal data under
Indonesia's UU PDP (Law No. 27/2022, Personal Data Protection Law), and
`agama` (religion) specifically falls into the stricter "specific personal
data" category most privacy laws single out for extra protection. Scoped
at the project's current stage — a demo/portfolio project, not yet
processing real users' own KTPs in production — rather than as a
full legal-compliance engineering effort, which would be premature before
there's a real deployment with real users to protect.

**Audited, not assumed — what the code actually does today:**
- No logging statement anywhere in `services/api` or `services/ocr` logs
  image content or extracted field values (checked directly, not
  assumed) — all logging is startup/config-level (model paths, service
  URLs). Uvicorn's own access logs record method/path/status only, not
  request bodies.
- The async job queue (`/v1/ktp/jobs`, backed by Redis via `arq`) stores
  the **raw uploaded photo bytes** as the job's arguments, not just the
  extracted result — `arq`'s defaults apply (`keep_result=3600`, i.e. the
  extracted-fields result auto-expires after 1 hour) but this project
  never explicitly configures or verifies how long the raw image itself
  persists as job data; it currently relies on `arq`'s library defaults
  rather than a deliberate, documented retention policy.
- No image or extracted data is ever written to disk by the pipeline
  itself — everything is in-memory per-request. (Any local files seen
  during development testing this session were scratchpad artifacts
  created by the testing process, not the application code, and were
  deleted afterward per the real-KTP handling protocol.)
- Auth is a single shared API key (`KTP_API_KEY`), or disabled entirely
  in dev — no per-user identity, no access audit trail. Fine for a
  demo/portfolio stage with one operator; not fine once other people's
  real ID data is involved.
- `.gitignore` already excludes `dataset/`, `eval_set/`, `eval_report*.json`,
  and model weights — no real or synthetic PII-shaped content has ever
  been committed to this repo (verified repeatedly this session).

**Baseline hygiene worth doing now** (cheap, no reason to wait):
- Set an explicit, deliberate TTL on the Redis job data (both args and
  result) instead of relying on `arq`'s defaults implicitly — make the
  retention policy a decision, not an accident.
- Document the above audit findings somewhere a future contributor (or
  future you) would actually see them before changing the job queue —
  this section is that, for now.
- Add a one-line note to the API's own docs (`/docs`, `/demo`) that
  uploaded photos are processed transiently and not intentionally
  retained, so anyone testing it (including future real users) isn't
  left guessing.

**Would need to happen before any real user's KTP flows through this
in production** (not built, deliberately deferred until there's an
actual production deployment to justify the effort):
- A real lawful basis + explicit consent flow for processing sensitive
  personal data under UU PDP — this API has no consent mechanism today;
  that's currently the calling application's responsibility, not this
  service's, and that boundary would need to be explicit and enforced.
- Per-user authentication/authorization and an access audit trail
  (who extracted what, when) — the current single-shared-key model
  doesn't support this at all.
- A deliberate, short, documented data retention policy enforced in code
  (not library defaults), plus encryption at rest for anything that does
  get persisted even transiently.
- Data subject rights support (access/rectification/erasure requests) —
  likely easy in practice given nothing is durably stored today, but
  needs to be an explicit, testable guarantee, not an assumption.
- A Data Protection Impact Assessment (DPIA) — UU PDP expects one for
  processing that's high-risk by nature (sensitive personal data, at
  scale), which this qualifies as once real users are involved.
- Cross-border data transfer safeguards if deployed outside Indonesia
  (relevant once a deployment platform/region is chosen — see the
  deployment planning above).
- Breach notification procedures (UU PDP requires notifying the data
  protection authority and affected individuals within specific
  timeframes) and a basic incident response plan.
- A security review of the actual deployed system (not just this
  session's code-level testing) before real ID photos go anywhere near it.

**Effort estimate**: the baseline hygiene items above are each
small (well under an hour of engineering each). The "before real
production" list is a genuinely different scale of work — legal review
(consent language, DPIA, retention policy sign-off) alongside the
engineering (auth overhaul, audit logging, encryption) — realistically
weeks, not hours, and involves decisions (what jurisdiction, what user
base, what legal review process) this project doesn't have answers to
yet. Revisit this list once a real production deployment is actually
planned, not before.

## Status

End-to-end pipeline works, verified through the real separated services (not
just in-process function calls): YOLOv8n detector (trained on synthetic
data, mAP50 0.995 — validates the plumbing, not real-world accuracy) →
contour-based deskew → HTTP call to the OCR service → PaddleOCR
label-matching field extraction.

The async job queue is now production-grade, not the earlier in-memory
placeholder: `/v1/ktp/jobs` enqueues onto Redis via `arq`, a separate
worker process (`app/worker.py`) consumes it, and job state lives in Redis
rather than API process memory — verified end-to-end across three real
processes (API, worker, Redis), including that a job submitted before an
API restart is still correctly pollable after one. `arq` (async-native)
was chosen over the more common RQ specifically because RQ's worker relies
on `os.fork()`, which doesn't exist on Windows — this dev machine needed
the worker to run natively here too, not just in a Linux container.

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

**Tested against one real KTP photo** (a real card, kept strictly local for
the test and never committed or persisted anywhere in this repo). Detection
worked essentially perfectly — 0.95 confidence, correctly rectified. OCR
extraction initially came back almost entirely wrong: a systematic
one-row-shifted misread across nearly every field. Root cause: the
synthetic generator stacks label above value on two lines; a real KTP
prints them as separate OCR boxes side-by-side on the same row (occasionally
merged into one line) — a layout our label-matching had never seen. Fixed
`field_labels.py`/`parsing.py` to try three strategies in order (same-line
merged, same-row separate box, stacked-below) instead of only the last one,
plus two guards found necessary along the way (an oversized garbled label
box wrongly excluding its own value; the card's background security
watermark texture OCR'ing as spurious text that contaminated an unrelated
field). Also found the `PROVINCES` reference list was an incomplete
12-province sample missing real (including newer, 2022-created) provinces
entirely — expanded to the full official 38, plus a missing religion
(Konghucu). Re-tested after each fix: **15 of 17 fields now correct** (a few with OCR
word-spacing noise — e.g. a multi-word value coming back with its spaces
dropped, content right but concatenated — a PaddleOCR text-recognition
artifact, not a matching bug). The one
remaining real gap: `kelurahan_desa` comes back empty because the real
card prints RT/RW and Kel/Desa as fully separate rows, while the synthetic
generator (and this schema) combine them into one — a structural,
schema-level difference deliberately left unfixed until proper research
into regional/historical KTP layout variation is done (see below), rather
than redesigning the generator off a single sample.
- **Re-tested the same real KTP end-to-end through the live system** (all
  four real processes — Redis, OCR service, API, worker — driven through
  the API's Swagger UI at `/docs`, not `evaluate_pipeline.py` or a direct
  in-process pipeline call, so this is the first time the deployed service
  itself was validated against a real photo rather than just its internal
  pipeline). This was after the detection hard-negative retraining rounds
  and the OCR fixes above. Detection: 0.91 confidence, correctly rectified.
  OCR: 15 of 17 fields correct again, NIK valid, overall confidence 0.90.
  One new failure mode found: `tanggal_lahir` came back empty because the
  date OCR'd with *both* separators degraded ("13-032007" — no separator at
  all between month and year, not just a dropped hyphen like the
  previously-fixed case), so `split_tempat_tanggal_lahir` couldn't locate
  the date pattern at all and left the whole unsplit string in
  `tempat_lahir` instead. The other two misses are the already-documented
  word-spacing artifact (a multi-word value losing its spaces, content
  still correct) — not a new issue.
  **Fixed**: `_TTL_SEPARATOR` in `parsing.py` required a hyphen-or-space
  separator between *each* date component (day, month, year); changed
  every internal separator to optional (`[-\s]?`) so it now recognizes a
  date with any mix of hyphen, space, or nothing between its parts — the
  original hyphenated form, the previously-fixed dropped-hyphen form, and
  this fully-concatenated form all now split correctly. Covered by a new
  regression test in `test_parsing.py`; full suite (42 tests) and
  `evaluate_pipeline.py` both still pass with no regression (this exact
  degradation pattern isn't present in the synthetic eval set, so its
  numbers are unchanged, not improved).
- Trained purely on synthetic data — has now seen exactly one real photo
  (see above), tested twice. Broader real-world validation (more samples,
  different regions/eras/lighting) is still needed before this is
  trustworthy.
- **Researched KTP-el layout variation, scoped to the current electronic-KTP
  era only (~2011-present) — the older pre-chip format is obsolete and out
  of scope.** Finding: KTP-el is governed by a single national specification
  (currently Permendagri No. 72/2022) with centrally-distributed
  enrollment/printing software, not a regionally redesigned template —
  reasonable evidence there's one layout nationally, not several. No source
  found directly confirms the specific visual detail we needed (label/value
  side-by-side vs. stacked) at that level of detail, though — that's
  inferred from "one national template," not independently verified beyond
  the one real card tested. Acted on it anyway since it's strictly better
  than the previous, already-proven-wrong assumption regardless: the
  generator now renders every field label and value side-by-side on one row
  (fixed value column) instead of stacked on two lines, and RT/RW and
  Kel/Desa as separate rows instead of combined — both matching the real
  card. `field_labels.py`'s `ROW_LABELS`/`_row_value` updated to look up
  `kelurahan_desa` as its own row (previously only extracted by splitting
  it out of a combined RT/RW value, which — now that the generator no
  longer combines them — went to 100% missing until fixed). Re-evaluated
  after the fix: `kelurahan_desa` 0% → 83.3% exact match on a fresh
  synthetic eval set. One regression surfaced at the time and left as-is:
  `golongan_darah` (blood type) dropped 73% → 50%, every failure a clean
  `None`, not a wrong value.
  **Fixed, root-caused via raw OCR geometry dump (not guessed):** the real
  cause was a systematic row-bleed affecting several fields, not just
  golongan_darah. Every value's OCR-detected box turned out to sit
  consistently ~10-17px *above* its own label's box — an all-caps value's
  ink reaches full cap-height while its (usually mixed-case) label's
  doesn't, so text drawn at the identical nominal y produces
  differently-positioned detected boxes. With only 17px between rows, that
  offset was large enough for `find_same_row_value`'s overlap check to also
  (wrongly) qualify the *next* row's value — confirmed directly: "Jenis
  Kelamin" was matching both its own value and "Alamat"'s, and the
  trailing non-letter text broke golongan_darah's end-anchored blood-type
  regex, while "Nama" separately matched its own value plus
  "Tempat/Tgl Lahir"'s date. Two changes: widened the generator's row
  spacing from 17px to 24px (`ktp_dataset_generator.py`, close to the max
  the 640×404 canvas allows before hitting the footer), and made
  `find_same_row_value` anchor on the best-matching candidate rather than
  accepting every candidate that merely clears the overlap threshold —
  preferring an at-or-above candidate over a below one (matching the
  observed offset direction), then discarding any other candidate not
  itself close to that anchor. Spacing alone wasn't sufficient (some row
  pairs' offsets are close to the row spacing itself even at 24px); the
  matching change was still needed. Re-evaluated on a fresh eval set:
  `golongan_darah` 50% → 86.7%, `jenis_kelamin` 86.7% → 100%,
  `tanggal_lahir` 80% → 90% — all fields the row-bleed had been silently
  touching. First version of the anchor logic (nearest-distance-to-label,
  no directional preference) regressed `nik` 100% → 93.3% by picking the
  wrong side of a near-tie in one case; fixed by preferring at-or-above
  candidates outright rather than nearest-by-distance, confirmed back to
  100%. All 44 tests still pass.
  **The flagged risk was real — re-testing against the real card
  (via the new `/demo` upload page) surfaced a regression the very
  next test.** `berlaku_hingga` came back mangled ("12-06 20203" instead of
  "SEUMUR HIDUP"), the exact issue-date-near-the-signature contamination a
  gap-based guard had already fixed once before. Root cause: the
  at-or-above anchor preference was filtering the *entire* candidate list
  by closeness to a single anchor, before the existing gap-based walk got
  a chance to run — on this row, the stray issuance date ("12-06-2024")
  happened to sit above the label while the true value sat almost exactly
  level with it, so the anchor rule picked the stray box and discarded the
  correct one outright, even though the gap-based logic (checking x-distance,
  not y-position) would have excluded that same stray box correctly on its
  own. Fixed by narrowing the anchor logic: only de-duplicate candidates
  that start at nearly the *same x* as each other (the synthetic
  generator's row-bleed signature — a fixed value column means a wrong
  next-row candidate lands at virtually the same x1 as the correct one),
  leaving genuinely far-apart candidates (the real card's stray-box
  signature) untouched for the original gap-based walk to handle as
  before. Re-verified: `berlaku_hingga` back to correct, golongan_darah
  gains held (confirmed via the synthetic eval set unchanged from the
  numbers above). Separately, while re-testing the real card end to end,
  found `golongan_darah` had also regressed to `None` — unrelated to any
  of the above, and present since well before this session: OCR read the
  blood-type letter "O" as the digit "0", and `split_jenis_kelamin_gol_darah`'s
  `[ABO]`-only pattern didn't recognize a trailing "0" as a blood type at
  all. Fixed by accepting a trailing "0" and normalizing it to "O" before
  snapping to the known `BLOODS` values (blood type is never actually a
  digit, so this is unambiguous). **Result: all 17 fields now correct on
  the real card, the best result yet (previously 15/17)**, overall
  confidence 0.96. 4 regression tests added across this fix (2 field-match
  cases with real-card coordinates, 1 for the digit/letter blood-type
  confusion, all passing); 50 tests total.
- **Digit-lookalike autocorrection for digit-only fields.** The
  golongan_darah fix above corrected one specific digit-standing-in-for-a-
  letter case (OCR read "O" as "0"). The mirror problem — a letter standing
  in for a digit (e.g. "13-03-2OO7" instead of "13-03-2007") — was still
  unhandled, and `only_digits()` (used for NIK) would silently *drop* any
  such letter entirely rather than correct it, shortening the NIK and
  breaking its length validation instead of just being noisy. Added
  `normalize_digit_lookalikes()` (`parsing.py`), swapping OCR's common
  digit-lookalike letters (O/o→0, I/l/i→1, S/s→5, B→8, Z/z→2, G→6) back to
  digits, and applied it to the three fields that are digit-only (or
  digit-heavy) by format: NIK, RT/RW, and the isolated date value (after
  splitting from tempat_lahir's free text). Deliberately *not* applied to
  `berlaku_hingga` (can legitimately be literal text, "SEUMUR HIDUP", so
  blind character substitution would corrupt it — confirmed by a test:
  naive substitution turns it into "5EUMUR H1DUP") or to `nama`/`alamat`
  (free text/proper nouns — a dictionary or language-model spell-check was
  considered and rejected for these: Indonesian names and street names are
  exactly the kind of non-standard proper nouns a spell-checker would
  "correct" into a plausible but wrong value, which is worse for identity
  data than a visibly-noisy OCR error a human can catch). The existing
  `snap_to_enum` fuzzy-match mechanism already covers every closed-
  vocabulary field this way (gender, blood type, religion, marital status,
  job, province) — this fix is the equivalent for the fields whose "closed
  vocabulary" is just "digits." 3 regression tests added; 53 tests total,
  no regression on `evaluate_pipeline.py` or the real card.
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
- **OCR latency is ~3-4s/request on small synthetic images; a real phone
  photo runs slower but the exact number turned out to be unreliable —
  history below, corrected after re-testing.** First measurement: one real
  photo through the live system came back at 13.1s, ~3x the 3.6-4.2s seen
  on 960×720 synthetic images, plausibly explained by real photos being
  higher-resolution (both YOLO and PaddleOCR scale with image area).
  Added `_downscale_if_needed` to `pipeline.py`: shrinks any upload wider
  or taller than 1600px (preserving aspect ratio) before detection, a
  no-op below that. Measured on a synthetic image upscaled to 3120×2340 to
  stand in for a real photo: 9.1s → 6.1s, ~33% faster, NIK unaffected, but
  `provinsi` came back empty with downscaling on that one run — flagged at
  the time as possibly an artifact of the test's artificial
  upscale-then-downscale methodology rather than a real accuracy cost.
  **Re-validated against the actual real photo when it was re-shared**:
  turned out to be exactly 1600×1200 — at the threshold, so the downscale
  fix is a no-op for it. More importantly, the original 13.1s didn't
  reproduce at all: four separate re-measurements of the *same* photo
  (direct in-process call, live system cold request, live system warm
  request, live system started with the exact `--reload` flag the original
  test used) all landed in a tight 7.0-8.0s band — including one run with
  no code changes and matching startup flags. `provinsi` extracted
  correctly (0.97-0.99 confidence) in every one of these real-photo runs,
  confirming that miss really was a test-methodology artifact, not a real
  accuracy cost. Conclusion: the original 13.1s was most likely a one-off
  anomaly (system load or similar), not a reproducible baseline, so the
  "33% faster" synthetic figure doesn't have a confirmed real-world problem
  to solve behind it yet. The downscale fix itself stays in (it's a
  reasonable no-op-when-unneeded safeguard for genuinely huge uploads,
  e.g. a raw 4000px camera photo not pre-compressed by something like
  WhatsApp), but its benefit is unproven on real data, and the threshold
  (1600px) is left unchanged rather than lowered to force it to engage on
  photos like this one — that would trade unproven accuracy risk for a
  speed problem that hasn't been confirmed to exist.

  **Profiled properly before optimizing further** (all detector/OCR fixes
  above measured fresh, current numbers — not assumed from before):
  detection (YOLO) ~29ms, deskew/perspective warp ~9ms, the HTTP hop to
  the separate OCR service ~130ms overhead, image decode ~23ms — all
  negligible. **PaddleOCR itself is ~5.5s, roughly 97% of the ~5.6-5.7s
  total** on the real card as currently measured (itself noticeably faster
  than the ~7-8s reported earlier in this same session — most likely
  because the detector retraining since then produces a tighter rectified
  crop, 1223×1188 vs. a larger ~1570×1069 one before, so PaddleOCR has
  fewer pixels to process; reported as freshly measured rather than
  assumed carried over). Conclusion: any further latency work has to
  target PaddleOCR specifically, nothing else is worth touching.
  Tried, then reverted: downscaling the *rectified card* (separate from
  the upload-level downscale above, which only touches the original
  photo and was already a no-op here) to at most 1000px before OCR. Real
  card: only 7% faster (5646ms → 5272ms) — far less than the ~55% pixel-
  area reduction would suggest, implying PaddleOCR's recognition cost is
  driven more by the number of text regions than by raw canvas area — and
  a real accuracy cost on the *only* data point that exercises this code
  path at all: `nama` dropped one character partway through the name, and
  `alamat` lost its punctuation. The synthetic eval set gave zero
  signal on this change either way (its crops are already smaller than
  1000px, so the code path never engaged) — another case of the synthetic
  harness being unable to validate a real-photo-scale change. Reverted
  given the poor return (small speedup, real accuracy risk, no synthetic
  validation coverage).
  **ONNX export: tried, blocked by a Windows-specific DLL failure, not a
  dead end.** Planned as a low-risk lever since PaddleX (which `paddleocr`
  is built on) turned out to have first-class, official support for this
  — a built-in `onnxruntime` inference engine, plus an official
  `paddlex --install paddle2onnx` / `paddlex --paddle2onnx` conversion
  pipeline purpose-built for PaddleX/PaddleOCR's own models, not a manual
  reimplementation of detection/recognition pre- and post-processing.
  Feasibility confirmed (the plugin installed cleanly, `onnxruntime` was
  already present), but the actual conversion attempt failed immediately:
  `paddle2onnx`'s own compiled C++ extension
  (`paddle2onnx_cpp2py_export`) fails to import with `ImportError: DLL
  load failed... The specified procedure could not be found` — a
  version-mismatched native dependency, not a missing one (`vcruntime140.dll`
  is present in both expected locations, ruling out the obvious fix).
  This is the *third* distinct Windows-specific native-extension failure
  in the PaddlePaddle ecosystem this session (alongside the cuDNN
  `zlibwapi.dll` issue and the PyTorch/PaddlePaddle same-process DLL
  conflict) — a real pattern on this machine, not a fluke. Stopped here
  per the planned fallback rather than debugging indefinitely: **nothing
  in the running code was touched**, `PaddleOcrService` (PaddleOCR-native,
  CPU) remains the only path, completely unaffected. Given every other
  Windows-native-extension issue this session turned out to not reproduce
  on Linux (confirmed directly for GPU, via a free Kaggle T4), the same
  move would plausibly resolve this too — untried, a candidate for
  revisiting alongside the GPU work rather than fighting further on
  Windows.
  **Follow-up: confirmed on the same free Kaggle instance used for the
  GPU test, exactly as predicted.** `paddle2onnx`'s Linux wheel (a
  genuinely different compiled artifact from the Windows one that failed
  — pip resolves a separate `manylinux` build) converted both
  `PP-OCRv6_small_det` and `PP-OCRv6_small_rec` cleanly, no DLL issue at
  all — confirming the failure really was Windows-specific, the third
  time this exact pattern has held this session. Getting the converted
  models actually *running* needed two more fixes along the way: PaddleX
  defaults to expecting its library-default model name when given a
  custom `_model_dir`, so `_model_name` must be passed alongside it or it
  errors with a name-mismatch; and pointing at an ONNX-only directory
  isn't sufficient by itself — `PaddleOCR(..., engine="onnxruntime")`
  must be passed explicitly, or PaddleX still looks for native Paddle
  format files and fails to find them.
  **Result, on a fair same-machine comparison (forcing both to CPU
  specifically, since the installed `onnxruntime` build was CPU-only —
  GPU would need the separate `onnxruntime-gpu` package, untried):
  ONNX Runtime 1065-1078ms vs. native PaddlePaddle 3641ms — roughly a
  3.4x CPU speedup, with byte-for-byte identical recognized text between
  the two engines (zero accuracy cost, as expected since it's the same
  model weights, just a different execution engine).** Getting the native
  CPU comparison running at all surfaced a bonus finding: the exact same
  oneDNN crash this project already works around locally
  (`enable_mkldnn=False` in `paddle_ocr_service.py`, `onednn_instruction.cc`)
  reproduces on Linux too, just via a different specific error message —
  confirming that particular bug is a genuine PaddlePaddle issue, not a
  Windows-only artifact like the others found this session.
  Quantization remains untried (real speed lever, real accuracy risk,
  needs calibration data and full re-validation).
  **Integrated into `services/ocr` behind an opt-in flag, and fully
  validated — including catching and correcting a self-inflicted false
  alarm along the way.** `PaddleOcrService` now accepts
  `engine="onnxruntime"` plus the two converted model directories (wired
  through `KTP_OCR_ENGINE=onnxruntime` /
  `KTP_OCR_ONNX_DET_DIR`/`KTP_OCR_ONNX_REC_DIR` env vars in the OCR
  service's `main.py`); the default (`engine="paddle"`, unset env var) is
  completely unchanged. Loading and running inference with the converted
  ONNX models works cleanly on Windows with no DLL issues at all — unlike
  every native Paddle-ecosystem extension this session, `onnxruntime`
  itself is a separate, more mature cross-platform library, not sharing
  that fragility.
  A first correctness check looked alarming — feeding a raw, undetected
  scene photo directly to `run_ocr()` (skipping detection/rectification
  entirely) produced garbled fields and value/label offsets nearly 3x
  larger than the geometry this project's field-matching logic was tuned
  around. That was a testing mistake, not a real engine difference: the
  OCR service expects an already-rectified card, not a full scene, and
  feeding it the wrong kind of input broke the geometry assumptions
  regardless of engine. The actual validation — `evaluate_pipeline.py`
  (now with `--onnx-det-dir`/`--onnx-rec-dir` flags) run properly through
  the full detection+OCR pipeline against the 30-image synthetic eval
  set — came back **field-for-field identical to the native-engine
  baseline, zero regression on any field**. The real card, run through
  the same full pipeline: **all 17 fields correct** (matching the best
  native result) at **819ms total pipeline time, vs. ~5.6-5.7s native on
  the same machine — roughly 7x faster**, a stronger result than the
  ~3.4x measured in isolation on Kaggle (this includes the full
  detection+OCR pipeline, on this machine's specific hardware, not just
  OCR alone on a synthetic image).
  Kept opt-in rather than made default: the ONNX model directories don't
  auto-download the way the native engine's models do (they're a manual
  `paddlex --paddle2onnx` conversion step, confirmed working on Linux
  only so far, then copied in — see `services/ocr/onnx_models/`,
  gitignored), so defaulting to it would silently break a fresh
  install/deployment that hasn't done that setup. Both GPU (native,
  ~162ms OCR-only) and ONNX-on-CPU (~1070ms OCR-only, ~819ms full
  pipeline on a real card) are now confirmed, real, and validated levers.
  GPU remains the intended primary path once cloud access exists;
  ONNX-on-CPU is available today, right now, with no billing dependency
  at all, and is a dramatically stronger fallback than plain CPU while
  GPU access is blocked.
  **Real end-to-end confirmation, the most honest number yet**: ran with
  the ONNX engine live through all the actual moving parts — the `/demo`
  page, a real browser upload, FastAPI's multipart parsing, the full
  detection+OCR pipeline — against the real card. **17/17 fields correct,
  ~2s processing time.** Higher than the 819ms measured by calling
  `pipeline.run()` directly (that number skips browser upload overhead,
  multipart parsing, and isn't guaranteed warm), but still a genuine
  ~2.8x improvement over the ~5.6-5.7s native baseline on the same
  request path — and the more trustworthy figure of the two, since it's
  what an actual user of this API would experience rather than a
  synthetic benchmark.

  The 3-4s synthetic-image figure is itself down from ~12-14s before the
  two CPU-only optimizations that got it there: skipping PaddleOCR's redundant doc-orientation/
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
  that DLL name/ABI at all). Reverted to CPU for reliability.
  **This is very likely a Windows-only problem, partially confirmed.** cuDNN
  links via standard versioned `.so` files on Linux, not Windows'
  `LoadLibrary`/`GetProcAddress` mechanism, so the deployment target (a Linux
  container) probably doesn't hit this at all. WSL2 + Docker Desktop with
  `--gpus all` was installed on this dev machine specifically to test this:
  GPU passthrough into a real Linux container works cleanly (`nvidia-smi`
  runs fine inside one), and the two missing-dependency errors hit along the
  way (`libgomp.so.1`, `setuptools`) were both mundane one-line apt/pip fixes
  — nothing resembling Windows' `zlibwapi.dll` ABI mismatch. That's
  meaningfully supportive of the hypothesis. What's still unconfirmed: a
  full `PaddleOCR()` instantiation with GPU never completed — the
  `paddlepaddle-gpu` wheel download (~700MB, from a Chinese CDN) was
  consistently very slow (as low as 341 kB/s, 1.5-hour ETAs) through Docker
  Desktop's WSL2 network path and corrupted mid-download twice, which also
  filled the host disk once (recovered, unrelated to this project) and
  burned significant time without a clean result. This looked like a
  network-path/environment problem specific to Docker-Desktop-on-Windows,
  not a new technical blocker, but the actual cuDNN-loads-on-Linux question
  stayed open until confirmed directly.
  **Confirmed: GPU inference works, and the speedup is transformative.**
  Tested on a free Kaggle notebook (T4 GPU, no billing setup required —
  the practical option while cloud billing access is a real barrier right
  now) rather than the blocked local WSL2 path: `paddlepaddle-gpu` (CUDA
  version matched to whatever the notebook's own `nvidia-smi` reported)
  and `paddleocr` installed cleanly and fast — datacenter networking made
  the exact download that failed for hours on Windows a non-issue.
  `PaddleOCR(device="gpu", ...)` with the same `PP-OCRv6_small_*` models
  this project uses instantiated and ran without any of Windows' DLL
  errors. Measured 50ms on a single-line test image, then 162ms on a
  synthetic image built to match a real KTP's field density (~17 lines of
  text) — the more representative number, since recognition cost scales
  with the number of text regions, not just image size (learned the hard
  way testing CPU-side downscaling earlier). **162ms vs. the ~5.5s CPU
  baseline on a real photo is roughly a 34x speedup.** This resolves the
  single biggest open question this project has had. Caveats before
  treating 162ms as the production number: this measured PaddleOCR alone
  on a notebook-drawn synthetic image, not the full pipeline (detection +
  deskew + OCR) against an actual detected/rectified real card, and not on
  whatever GPU a real deployment platform would actually provision (T4 is
  a reasonable baseline, but instance type varies by provider/cost tier).
  Re-validating against the real photo, on whatever GPU the eventual
  deployment platform provides, is the natural next step once cloud
  billing access exists — but the core technical risk (does this even
  work on Linux) is no longer a risk.

  **Decided architectural plan, pending that validation: GPU is the
  intended primary path, CPU is the fallback.** Composing this session's
  separate measurements (image decode 23ms + YOLO detection 29ms + deskew
  9ms + the ~130ms inter-service HTTP hop + ~162ms GPU OCR) gives a
  theoretical full-pipeline estimate of **~350ms** — comfortably clears
  the sub-2-3s interactive-latency target discussed earlier, by a wide
  margin. That's a big enough gap that it changes the plan, not just a
  nice-to-have: with CPU alone (~5.6-5.7s/request), the sync
  `/v1/ktp/extract` endpoint was a fallback and `/v1/ktp/jobs` (async) was
  the recommended primary flow, since ~5-8s feels broken for an
  interactive upload. At ~350ms, sync becomes viable as the *primary*
  flow instead — a real UX simplification (no polling/job-status UI
  needed) if it holds up.
  **This is a known limitation, not a shipped result: the ~350ms figure
  is composed from separate measurements on different hardware/images, not
  one real end-to-end request timed as a whole.** It has not been
  validated against the actual pipeline (detection → deskew → OCR in one
  run), the real card, or a real deployment GPU — all blocked on the same
  cloud billing access gap. Until that validation happens, CPU remains
  what's actually running, and the async-primary/sync-fallback framing
  from before still describes production reality today, not the GPU
  numbers above. Other remaining CPU-only levers, now lower priority
  given this result: an older pre-3.x paddlepaddle that might restore
  working MKL-DNN on CPU (real risk: probably forces a paddleocr 2.x
  downgrade too, a different API), `PP-OCRv6_tiny_*` if its accuracy risks
  turn out to be acceptable, or ONNX export.
- **False-positive tested: does the detector correctly reject non-KTP
  images?** Never validated before — all prior testing only checked
  detection on images that do contain a card. Ran the trained detector
  against a batch of real, definitely-card-free photos (Windows wallpapers)
  plus targeted synthetic distractors. 10/10 real photos were correctly
  rejected, but 2 targeted tests exposed a real gap: a generic
  business-card mockup (white rectangle, black border, unrelated text
  lines — nothing KTP-specific about it) triggered a false detection at
  0.92 confidence, *higher* than most true positives, and one wallpaper got
  boxed as a "card" covering nearly the entire image. Root cause: the
  training set was 100% positive examples — every synthetic training image
  contained a KTP, so the model had learned "bordered rectangle containing
  text lines" rather than any KTP-specific visual signature (blue header
  band, photo placeholder, particular field layout). Confirmed with an
  isolating test: a plain colored rectangle with no text was correctly
  ignored, but adding a border + text lines to that same rectangle made it
  fire. Fixed by adding hard-negative generation to
  `ktp_dataset_generator.py` (`render_distractor`/`compose_negative_scene`,
  new `--neg-ratio` flag, default 0.15): plain backgrounds plus
  deliberately card-shaped, text-bearing but non-KTP objects (a
  business-card-style mockup, a framed-photo panel, a receipt-like narrow
  strip of text), all labeled with an empty YOLO label file. Regenerated
  the dataset and retrained (same yolov8n/40-epoch settings) — mAP50 held
  at 0.995 (no positive-detection regression, confirmed via
  `evaluate_pipeline.py` giving identical per-field numbers to before), and
  all 12/12 false-positive tests now pass, including both that previously
  failed.
  A follow-up adversarial test then found the fix was shallower than it
  looked: a mockup deliberately reusing KTP's *exact* palette (header/footer
  blue, cream body, gold emblem square) but different title/labels/content
  still triggered a false detection at 0.97 confidence — the model had
  learned to key on color scheme + layout, not actual content. Added a
  second distractor kind (`colored_id_card` in `ktp_dataset_generator.py`)
  that reuses KTP's palette (jittered, plus an exact match) with a photo box
  and label/value rows, but non-KTP titles and field labels, and retrained
  again — mAP50/OCR accuracy again unchanged. Result is a genuine partial
  fix, not a full one: the realistic case (the look-alike card rotated and
  placed on a background, the way an actual phone photo would look) now
  correctly passes, but an unrealistic edge case — the same look-alike
  filling the entire frame edge-to-edge with zero rotation and no background
  margin — still fires, though confidence dropped meaningfully (0.97 →
  0.84). Likely cause: `compose_negative_scene` (and `compose_scene` for
  positives) always places its subject rotated with background margin, so
  neither positives nor negatives ever train the model on an edge-to-edge,
  unrotated frame — an out-of-distribution gap in how training scenes are
  composed, not specifically a color/content issue.
  A third round confirmed that hypothesis and closed most of it: two
  independent tests — a grayscale/photocopy-style render of a real
  synthetic KTP (checking the *inverse* risk, that color-reliance might
  cause false *negatives* on a real card; it didn't, all variants still
  detected at 0.95+) and two fictional "sibling" Indonesian documents (a
  SIM/driver's-license mockup sharing several KTP field labels and general
  styling, and a KK/family-card mockup with a very different tall-page
  layout) — showed the SIM look-alike hit the exact same
  edge-to-edge/no-rotation-only failure pattern (0.76 confidence) despite
  using a different blue than KTP's, while the KK look-alike (very
  different shape) was rejected cleanly in every form. That confirmed the
  gap was scene-composition (framing), not color or content. Fixed by
  adding `_compose_edge_to_edge` to `ktp_dataset_generator.py`: 15% of the
  time, both `compose_scene` (positives) and `compose_negative_scene`
  (negatives) now place their subject filling the entire frame edge-to-edge
  with near-zero rotation instead of always leaving background margin and
  a larger rotation range. Retrained again — mAP50/OCR accuracy unchanged
  — and both previously-failing edge-to-edge cases (blue-header mockup,
  SIM mockup) now correctly reject. This did surface one new, weaker false
  positive: the KK mockup, flat/edge-to-edge only, now fires at 0.50
  confidence (barely above the 0.4 detection threshold, boxing only the
  page's top portion) — versus 0.95+ for every real detection. Root cause:
  none of the distractor kinds are tall/portrait-shaped (KK is a full A4
  page, ~620×876), so edge-to-edge training never covered that aspect
  ratio. Net effect of this round: two confident false positives fixed,
  one much weaker one introduced. Left undone at the time rather than
  chasing a fourth round in the same session (diminishing returns /
  whack-a-mole risk).
  **Revisited, then reverted — a serious lesson about what the synthetic
  eval set can't catch.** Added a `portrait_document` distractor kind
  (`ktp_dataset_generator.py`) — A4-ish proportions (~0.65-0.75
  width:height, vs. every other distractor's landscape/card shape), a
  bordered title block, a few label/value lines, then a header row and
  several data rows, mimicking a household-register-style document.
  Retrained: mAP50 0.995 (unchanged), `evaluate_pipeline.py` gave identical
  per-field numbers to before (no regression), and the KK mockup that
  previously fired at 0.50 confidence now correctly rejected. By every
  measure this project had been using, this looked like a clean fix.
  **It wasn't** — re-tested against the one real KTP photo (the same
  discipline that caught the `berlaku_hingga`/`golongan_darah` regressions
  above) and the retrained detector found *zero* boxes on it at all, not
  even a low-confidence one, versus 0.91 confidence on the exact same
  photo with the previous weights. A real, complete detection failure that
  every synthetic and false-positive check had missed entirely. Reverted
  to the previous weights immediately (`ktp_detector_v4`) — confirmed
  detection restored (0.91 confidence, matching every prior real-card
  test). **Root-caused on a follow-up, once the real photo was available
  again**: `_compose_edge_to_edge` (shared by both `compose_scene` and
  `compose_negative_scene`) force-resized *every* subject into a fixed
  960×720 landscape frame regardless of its native proportions. For
  `portrait_document` specifically (~0.7 width:height) that's a severe
  horizontal stretch unlike any other distractor kind — plausibly enough
  visual distortion, appearing in 15% of negative training examples, to
  shift the model's general decision boundary and suppress recall on an
  already-somewhat-marginal real photo (0.91 confidence was never as
  confident as synthetic positives' typical 0.99+). Fixed at the source
  rather than patched around: the edge-to-edge frame now sizes itself to
  match the (rotated) subject's own dimensions instead of forcing a fixed
  size, since a real "photo cropped tight to a document's edges" would
  naturally have that document's own aspect ratio anyway. This meant
  updating the two label-normalization call sites in
  `generate_dataset` that had assumed every scene was exactly IMG_W×IMG_H
  to use the actual per-image scene size instead.
  Retrained (`ktp_detector_v6`) and — critically, in this order this
  time — **validated against the real photo *before* touching production
  weights**: 0.87 confidence (close to the original 0.91), then confirmed
  the KK false positive still correctly rejects, then confirmed
  `evaluate_pipeline.py` unchanged, only *then* promoted to production.
  All green. **Confirms the methodology lesson stands regardless of this
  fix**: any future retraining on this project must be validated against
  the real photo *before* being promoted, not just the synthetic eval set
  and false-positive battery — those two alone gave a false all-clear the
  first time, and only re-testing against real data caught it.
  While validating this fix end-to-end, a *second*, unrelated bug
  surfaced on the real card: `status_perkawinan` came back as
  `"PEKERJAAN"` — literally the next field's label text, not "BELUM
  KAWIN". Root cause via a fresh geometry dump: OCR had merged the label
  and value onto one line with *no colon at all* this time
  ("Status Perkawinarc BELUM KAWIN" as a single box) — a fourth
  label/value layout pattern (alongside colon-merged, same-row-separate-
  box, and stacked-below) that nothing in `field_labels.py` handled, so it
  fell through every strategy to the stacked-below fallback, which then
  matched the *next row's label* (sharing the same left margin every
  label uses) since there was nothing else at that exact row to compete
  with it. Fixed by adding `merged_label_prefix_value`: finds the
  word-count prefix of a colonless line that best fuzzy-matches a known
  label, and treats everything after it as the value. This first version
  caused its own regression — `tempat_lahir`/`tanggal_lahir` collapsed to
  0% because a genuine label-*only* line ("Tempat/Tgl Lahir", no value
  present at all) had a partial prefix ("Tempat/Tgl" alone) that already
  scored 0.78 against the full canonical label, high enough to clear the
  same 0.6 threshold `find_label_line` uses elsewhere — so "Lahir" (the
  label's own trailing word) got sliced off and mistaken for a value.
  Caught immediately by re-running `evaluate_pipeline.py` (not skipped
  this time) before treating the fix as done; fixed by giving this
  specific check its own, stricter threshold (0.85) — a genuine merged
  label+value split scores far higher (~0.91 for "Status Perkawinarc"
  against "Status Perkawinan") since the *whole* label, not a fragment,
  precedes the value there. Final result: **all 17 fields correct on the
  real card again**, `evaluate_pipeline.py` numbers unchanged from
  baseline, 57 tests passing.
- **Security audit — grounded in the actual code, not a generic
  checklist.** Found five real findings, two fixed immediately (fastest,
  lowest-risk to change), three left as documented, prioritized gaps
  since they matter most once this is reachable by anyone but the
  operator, not right now:
  - **Fixed**: `/v1/ktp/jobs/{job_id}` was returning a failed job's raw
    exception message (`str(exc)`) straight to the client — a real
    information-disclosure risk, since an internal error could contain
    paths or other implementation details. Now logs the real exception
    server-side (`logger.exception`) and returns a generic message.
  - **Fixed**: Pillow's own default decompression-bomb protection
    (`Image.MAX_IMAGE_PIXELS`, ~89M pixels) already prevents the real
    memory-exhaustion risk from a maliciously crafted "image bomb," but
    `DecompressionBombError` wasn't in `pipeline.py`'s
    `except (UnidentifiedImageError, OSError)` tuple (confirmed via its
    MRO: it's a direct `Exception` subclass, not related to either), so
    hitting it produced an unhandled 500 instead of the same clean
    `invalid_image` response any other bad upload gets. Not a
    vulnerability by itself — a robustness/error-handling gap.
  - Both covered by regression tests (mocking a failed job's `Job.result`
    for the first; monkeypatching `MAX_IMAGE_PIXELS` down for the second,
    to keep the test fast rather than constructing an actual huge image).
    62 tests passing.
  - **Redis authentication — fixed next, the most dangerous of the three
    findings.** Redis had no password (`redis://localhost:6379`) despite
    transiently holding raw uploaded KTP photo bytes as job data —
    unauthenticated Redis is one of the most commonly mass-scanned-and-
    exploited misconfigurations on the internet. No application code
    needed changing — `redis-py`/`arq` both already support credentials
    embedded in the connection URL — so this was purely an operational
    fix: start Redis with `--requirepass`, set `KTP_REDIS_URL` to match.
    Verified, not just documented: confirmed an unauthenticated connection
    is actually rejected (`AuthenticationError`) and the correct password
    is accepted, then ran the full async job flow end-to-end (enqueue via
    the API → picked up by the worker → polled to completion) against the
    password-protected Redis to confirm nothing else broke. The README's
    run instructions now show the authenticated form as the one meant to
    be used, not an option — `KTP_REDIS_URL`'s no-password default still
    exists for a quick local check, but real usage should always set a
    password given what Redis holds here.
  - **Dependency version pinning — fixed.** Every declared dependency
    across both services (`requirements.txt`/`requirements-dev.txt` in
    `services/api` and `services/ocr`) and all three internal packages'
    `pyproject.toml` files (`libs`, `services/detection`, `services/ocr`)
    used a bare `>=` with no upper bound — a future `pip install` could
    silently pull a newer, compromised, or simply breaking version across
    a large surface (fastapi, pillow, paddleocr, paddlepaddle,
    ultralytics, etc.). Pinned every one to its exact currently-installed
    version (via `pip freeze`, not guessed) — versions already proven to
    work together in this exact environment, not an upgrade or downgrade
    of anything. Kept shared dependencies (`pillow`, `numpy`) at the same
    exact version across every file that declares them, to avoid a
    resolver conflict between packages. Also found and fixed a related,
    adjacent gap while at it: `onnxruntime` (needed for the
    `KTP_OCR_ENGINE=onnxruntime` path integrated last session) was never
    declared as a dependency anywhere at all — added as a pinned optional
    extra (`pip install -e "services/ocr[onnx]"`) rather than a required
    one, matching that path's already-opt-in design.
    Verified rather than assumed: the full test suite still passes, and
    `pip install --dry-run` was run against every changed
    requirements/pyproject file (including the new `onnx` extra) to
    confirm they all still resolve cleanly with no version conflicts.
  - **Rate limiting — fixed, closing out all five findings from the
    original audit.** Added `slowapi` (15 requests/minute per IP) to the
    two expensive endpoints, `/v1/ktp/extract` and `POST /v1/ktp/jobs` —
    each does real ML inference. Deliberately left `GET
    /v1/ktp/jobs/{job_id}` (polling) unlimited: it's a cheap Redis lookup,
    not the work the limit exists to protect, and a client checking its
    own job's status repeatedly shouldn't share that budget. Limited by
    IP, not API key, since the current single-shared-key auth model means
    every caller would share one bucket anyway — per-key becomes the
    better fit once per-user auth exists (already a "before real
    production" PII item).
    One design decision changed during implementation, worth being
    explicit about: the plan called for Redis-backed limit storage, on
    the reasoning that this project "already runs multiple processes."
    That reasoning didn't survive contact with the actual architecture —
    only the API process ever handles HTTP requests (the worker doesn't),
    and this isn't horizontally scaled, so in-memory storage (the
    library's default) is correct here, not a shortcut; Redis-backed
    storage would've been solving a scaling problem this deployment
    doesn't have. Revisit if the API is ever run as multiple replicas
    behind a load balancer.
    Verified live, not just unit-tested: 15 requests against a running
    instance succeeded, the 16th and 17th both returned 429 with a clear
    message, and polling remained unaffected throughout. Also added a
    `client` fixture reset for the limiter's in-memory state between
    tests — without it, one test hitting the limit would silently poison
    every later test sharing the same test-client source IP. `slowapi`
    pinned like every other dependency now. 65 tests passing (3 new).
- **Direct code review of `services/api` and `services/ocr`, scoped to
  the actually-served code rather than the wider repo (training scripts,
  eval harnesses) — cheap to do directly rather than via a fresh
  subagent, since the context of what's important here was already
  built up over the whole session.** Found and fixed one significant bug:
  both services' `/v1/ktp/extract` and `/v1/ocr/extract` handlers called
  their real work (`pipeline.run()`, `ocr_service.extract_fields()`) —
  synchronous, CPU-bound, 2s-13s depending on engine — directly inside an
  `async def` handler with no `asyncio.to_thread`. Since uvicorn's
  default run mode is one process with one event loop, this blocked the
  *entire process* for the full duration of every request — not just
  that caller, but every other concurrent request the instance was
  serving, including its own `/health` check (a real risk: a load
  balancer's health check timing out mid-request could kill the instance
  while it's actually fine). `worker.py` already had the correct pattern
  (`await asyncio.to_thread(ctx["pipeline"].run, image_bytes)`); applied
  the same fix to both HTTP handlers.
  Proven, not just applied: wrote a concurrency regression test per
  service (a slow stub service, two requests fired concurrently via
  `ThreadPoolExecutor`, asserting they overlap rather than serialize),
  then deliberately reverted each fix and re-ran its test to confirm it
  actually fails without the fix (0.6s+, serialized) before trusting it
  passes because of the fix (under 0.5s, concurrent) — not just that the
  test happened to pass. That check caught a second, subtler bug in the
  process: `services/ocr/tests/test_ocr_app.py` built its `TestClient`
  as a bare module-level instance rather than via `with TestClient(app)`,
  which (confirmed empirically, not assumed) gives each request its own
  isolated event loop rather than sharing one — so the concurrency test
  passed regardless of whether the handler fix was even applied,
  silently testing thread-level parallelism instead of the actual
  shared-event-loop behavior a real deployed process has. Fixed by
  converting that test file to the same fixture-based
  `with TestClient(app) as client` pattern `test_api.py` already used.
  67 tests passing (2 new).
  One minor finding from the same review left undone: `redis_pool.py`'s
  lazy singleton (`if _pool is None: _pool = await create_pool(...)`) has
  an unsynchronized check-and-create race — two requests arriving before
  the pool exists could both create one, leaking the loser. Narrow
  window (only matters at cold start), low impact (a wasted connection
  pool, not a correctness bug) — noted here rather than fixed this round.
- **A formal multi-hunter security audit (5 independent parallel reviewers,
  each assigned a distinct attack-class angle) went deeper than the direct
  review above and found 3 further real issues, plus 2 smaller ones — full
  report in this run's output.** First and most significant, **fixed**:
  both `services/api` and `services/ocr`'s upload handlers checked the
  request body's size (`_read_and_validate_upload`'s
  `len(contents) > max_upload_bytes`) only *after* `await file.read()` had
  already fully received it — verified directly against the installed
  Starlette/FastAPI source that FastAPI parses the full multipart body
  before any `Depends()`-based check or the `@limiter.limit` rate limiter
  ever runs, and that Starlette's multipart parser applies no size cap to
  file parts at all (only non-file form fields get its `max_part_size`
  check) — a file part spools straight to disk past 1MB with no upper
  bound. Net effect: a client — including an unauthenticated one, since
  `services/ocr` has no auth at all — could send a request body far beyond
  the declared 10MB limit and have the server fully absorb it (memory +
  disk) before finally rejecting it. Fixed with a new ASGI-level
  `MaxBodySizeMiddleware` (added to both services, duplicated rather than
  shared since the two services already keep independent upload-size
  constants) that rejects an oversized body before FastAPI's own parsing
  ever touches it: via `Content-Length` up front for a client that
  declares one honestly, and via a running byte count over the raw ASGI
  receive channel for chunked/absent-`Content-Length` requests. Verified,
  not just applied: added regression tests asserting on the middleware's
  distinctly-worded rejection message (not just a 413 from somewhere), then
  deliberately disabled the middleware and confirmed those tests correctly
  failed (falling back to the pre-existing endpoint-level message) before
  re-enabling it. 75 tests passing (6 new). The other 4 findings from this
  audit (a rate-limit gap on failed-auth requests, an unbounded-by-design
  Redis job-data retention window, a missing `DecompressionBombError`
  handler in `services/ocr` mirroring one already fixed in `pipeline.py`,
  and a dead-code-today `ValueError` in `libs/ktp_schema`'s `parse_nik`)
  are documented in the audit report, not yet fixed as of this entry.
- **Second audit finding, fixed: the rate limit didn't apply to requests
  that failed auth.** `@limiter.limit("15/minute")` only ran its check when
  the decorated endpoint function's body actually executed — but FastAPI
  resolves `dependencies=[Depends(require_api_key)]` *before* calling that
  function, so a request with a missing/wrong API key raised 401 during
  dependency resolution and never reached the decorator's counter at all.
  Verified empirically before fixing: 30 wrong-key requests against
  `/v1/ktp/extract` returned 30×401, never once 429. Fixed by switching
  from per-route `@limiter.limit(...)` decorators to `default_limits=
  ["15/minute"]` on the `Limiter` plus `app.add_middleware(SlowAPIMiddleware)`
  — slowapi's middleware runs before FastAPI's routing/dependency
  resolution, so the limit now applies regardless of whether auth
  eventually succeeds or fails. Routes that should stay unlimited
  (`/health`, `/demo`, job polling) now opt out explicitly via
  `@limiter.exempt` instead of the old model where limiting only ever
  applied to routes that opted in. Verified the same way as the buffering
  fix: added a regression test asserting failed-auth requests eventually
  hit 429, confirmed it (and the two pre-existing rate-limit tests) fail
  when the middleware is disabled, then restored it. 76 tests passing
  (1 new).
- **Remaining audit findings, fixed — closing out all issues from the
  formal audit.**
  - `services/ocr` was missing the `Image.DecompressionBombError` handler
    already fixed once in `services/api/app/pipeline.py`. Fixed identically
    (same except clause, same 422 response).
  - `libs/ktp_schema`'s `parse_nik()` could raise a raw, unlabeled
    `datetime.date` `ValueError` on a NIK encoding Feb 29 that resolves to
    a non-leap year — `validate_nik()`'s day/month check is deliberately
    lenient about this since it can't know the real century from a 2-digit
    year. Currently dead code in the live request path (only
    `validate_nik` is called there), but `parse_nik` is public API. Fixed
    by wrapping the date construction and re-raising as `parse_nik`'s own
    labeled "Invalid NIK" error.
  - The API key comparison (`x_api_key != settings.api_key`) was a plain
    string comparison, not constant-time — switched to
    `hmac.compare_digest`. Also closed a real, related test gap the audit
    flagged: `require_api_key` had zero test coverage before this (exactly
    the code path the rate-limit-bypass bug above lived in undetected).
  - `NIK_PATTERN` used Python's default Unicode-aware `\d`, so a NIK built
    from Unicode digit look-alikes (e.g. a fullwidth zero) could pass
    validation and slip past the exact-string "00" region-code checks
    while still evaluating to a real zero under `int()`. Fixed by compiling
    with `re.ASCII`.
  - `services/ocr` had no auth of its own at all, relying entirely on
    network position — a real gap once any shared-network deployment
    exists, though not yet materialized (no docker-compose/orchestration
    config exists in this repo). Added an optional shared-secret check
    (`KTP_OCR_INTERNAL_KEY`, off by default) mirroring `require_api_key`,
    with `RemoteOcrService` sending it as `X-Internal-Key` when configured
    on the api side too.
  - All five verified the same way as every other fix this session: a
    regression test per fix, each confirmed to fail when the fix is
    reverted, then restored. 87 tests passing (11 new).
- **Third audit finding, fixed: raw KTP photo bytes no longer default to a
  24-hour Redis lifetime.** `redis.enqueue_job(...)` was called with no
  `_expires`, so the job's argument data — the raw uploaded photo bytes —
  inherited arq's default expiry, confirmed by reading arq's own installed
  source: `constants.expires_extra_ms = 86_400_000` (24 hours), a generic
  scheduling safety margin, not a deliberate PII retention decision. Real
  processing finishes in seconds. Fixed by passing an explicit
  `_expires=JOB_EXPIRES_SECONDS` (10 minutes — generous headroom for arq's
  own retry defaults: up to 5 tries, 300s timeout each — without leaving
  the raw photo around anywhere near a full day). Verified the same way as
  the other two fixes: a regression test spies on the actual
  `enqueue_job(...)` call and asserts `_expires` is set, confirmed it fails
  without the fix, restored. 77 tests passing (1 new).
- **Tier-1 tamper-detection heuristics added (`services/api/app/forensics.py`)
  — classical image forensics, not ML, and explicitly scoped down from the
  original plan.** Four techniques were discussed (Error Level Analysis,
  JPEG block-grid misalignment, EXIF inspection, per-field font
  consistency); only ELA and EXIF are actually implemented here. The other
  two were deliberately deferred — both need real design work to implement
  reliably (naive block-grid forensics especially can produce
  confident-looking nonsense), and shipping an unvalidated version of
  either would be worse than not having it. This is a warnings/scoring
  layer, not a hard reject: a `ForensicsResult` (`exif_present`,
  `editor_software_detected`, `error_level_anomaly_score`, `suspicious`,
  `warnings`) rides alongside the existing extraction result, the same
  "confidence plus warnings, not a verdict" shape OCR confidence already
  uses.
  Runs on the *original* uploaded bytes, never on anything the pipeline
  has already resized or perspective-warped — both resample pixels and
  destroy the compression-history artifacts ELA depends on, so it has to
  run before `_downscale_if_needed`/`detect_and_rectify` would touch the
  image, not after.
  ELA here specifically looks for a *regional* anomaly (the error map is
  split into a grid and each cell's mean is z-scored against the rest),
  not a single whole-image average — a small localized edit can be
  invisible in an image-wide number while still standing out clearly
  against its immediate surroundings, which is the entire point of this
  check existing.
  **Measured, not estimated, per this project's own rule**: earlier
  latency planning estimated 50-150ms for all four originally-scoped
  techniques; the two actually built here measured at **~12ms average**
  on a 640×404 test image (10-run average, `services/api/app/forensics.py`
  timed directly) — negligible next to the ~2s ONNX or ~5.6s native OCR
  cost either way.
  **An honest, not-yet-resolved caveat surfaced by that same measurement**:
  running this against `training/sample_ktp.png` (the synthetic,
  template-rendered test card used for quick smoke tests) returned
  `suspicious=True` — almost certainly a false positive, since that image
  is vector-rendered graphics with none of a real photo's natural sensor
  noise, not evidence the check is broken. This project's own hard-learned
  lesson (the detector-retraining regression that only the real KTP photo
  caught, see below) applies just as much here: `ELA_ANOMALY_Z_THRESHOLD`
  (currently `3.0`, a first-guess placeholder) has not been calibrated
  against any real photo's false-positive rate yet, and shouldn't be
  trusted for that until it is — exactly the kind of validation gap this
  project's eval-harness discipline exists to catch before something like
  this ships as a real signal rather than a demo.
- **That caveat played out immediately: tested against the real KTP photo,
  ELA false-positived, and was shelved.** The genuine, unedited real card
  scored 4.6 std-devs — above the 3.0 threshold, flagged `suspicious=True`
  on a photo with nothing wrong with it. Root cause, not just a bad
  threshold: a real photo's card body, text edges, photo backdrop, and
  background surface naturally carry very different amounts of
  high-frequency detail, which produces the same regional
  compression-response variance this check was looking for from tampering
  — the smooth-gradient synthetic test images used during development
  never exercised that at all. The file was also a WhatsApp download,
  whose own recompression pass is a well-documented real-world source of
  ELA false positives independent of any editing. Same lesson this project
  has already learned once with the detector (a synthetic-only test set
  hid a real-photo-only failure) — this time on the very first real-photo
  test.
  **Decision: shelved ELA, kept EXIF inspection only.** `ForensicsResult`
  no longer carries `error_level_anomaly_score`; `suspicious` is now driven
  solely by detected editor-software EXIF tags. Re-verified against the
  same real photo afterward: `suspicious: False`, just the (expected,
  non-suspicious) "no EXIF" note — WhatsApp strips it, unrelated to
  tampering. ELA isn't abandoned for good — revisiting it would mean
  isolating the analysis to just the card region (excluding the
  background) and validating against more than one real photo, not just
  tuning the threshold on this same single case. 89 tests passing (2 ELA
  tests removed with the code they tested).
- **Fixed the last documented gap: `redis_pool.py`'s lazy-singleton race.**
  `if _pool is None: _pool = await create_pool(...)` had an unsynchronized
  check-and-create — two requests arriving before the pool existed could
  both pass the `None` check, both call `create_pool()`, and whichever
  finished last would silently overwrite the other's pool, leaking the
  loser's connections. Fixed with an `asyncio.Lock` and a re-check inside
  it (the standard double-checked-locking shape): the first waiter creates
  the pool, every other waiter then sees it's no longer `None` and skips
  creating another. Verified, not just applied: a test forces 10 concurrent
  callers through the race window (by making a mocked `create_pool` slow
  enough that every caller observes `_pool` as `None` before any of them
  finishes), asserts `create_pool` was only actually called once and every
  caller got the same pool instance — reverting the fix makes the same 10
  concurrent callers create 10 separate pools, confirming the test actually
  catches the bug it's named for. 90 tests passing (1 new).
