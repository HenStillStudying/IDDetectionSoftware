# Development guide

Setting up, running, testing and evaluating the project locally. For deploying
to a server see [`infra/DEPLOY.md`](../infra/DEPLOY.md).

## Layout

```
libs/ktp_schema/      Shared contracts: field schema, NIK validation, reference data, API result schema
libs/ktp_interfaces/  DetectionService/OcrService contracts (implemented by services/detection and services/ocr)
services/api/         FastAPI gateway: /v1/ktp/extract (sync) and /v1/ktp/jobs (async, via Redis + app/worker.py)
services/detection/   YOLOv8 card detection + contour-based deskew (YoloDetectionService), in-process
services/ocr/         Standalone PaddleOCR microservice — its own FastAPI app, its own process
training/              Synthetic KTP dataset generator, YOLO training script, and a ground-truth pipeline evaluator
infra/                 Deployment runbook (DEPLOY.md) and Docker build helpers
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
accuracy cost (see [CHANGELOG](CHANGELOG.md)), but opt-in since it needs a manual
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
image/Dockerfile, just a different command). Only `api` publishes a port,
and only on `127.0.0.1:8000` — never all interfaces, because on a server
Docker's iptables rules bypass `ufw` entirely, so a bare `"8000:8000"` would
be public regardless of the firewall (verified locally: the host's other
addresses now refuse the connection). `redis` and `ocr` are reachable only
from other containers on the compose network, matching the audit's
recommendation to not rely on "nothing exposes this port" as the only
control once a real network topology exists.

**Deploying to a VPS:** see [`infra/DEPLOY.md`](../infra/DEPLOY.md) — a
step-by-step runbook (SSH hardening, firewall, Docker install, secrets,
verification over an SSH tunnel, day-to-day operations), with server
sizing based on measured memory use.

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
  Windows-only DLL failure documented in [CHANGELOG](CHANGELOG.md): the container is
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

- **Fast rebuilds after code changes**: both Dockerfiles install every
  third-party dependency from the dependency manifests alone (read out of
  the `pyproject.toml` files by `infra/pyproject_deps.py`, so the pins
  still live in one place), before copying in any of this repo's code;
  our own packages are installed last with `--no-deps`. The ocr image's
  model-conversion stage also branches off before any source is copied.
  Measured: a one-line edit in `libs/` used to rebuild everything after it
  (~6 min, re-downloading PyTorch/PaddlePaddle); it now rebuilds in ~13s,
  with every dependency layer and the model conversion reused from cache.
  The reorder didn't change what's installed — `pip freeze` is identical
  to before in both images (67 and 81 packages), and `pip check` is clean.
- **Survives crashes and reboots**: every service has
  `restart: unless-stopped` (only an explicit `docker compose stop`/`down`
  keeps it down). Tested, not assumed: killing the api container's main
  process got it restarted by Docker with `/health` answering again within
  ~4s. Logs rotate at 3 × 10 MB per container — Docker's default
  `json-file` logging never rotates and would eventually fill a
  long-running host's disk.


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
KTPs (see "False-positive tested" in [CHANGELOG](CHANGELOG.md)).

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
actually caught the three bugs listed in [CHANGELOG](CHANGELOG.md) — worth re-running after any
change to detection or OCR, not just once.


## Why OCR is a separate service

This isn't just clean separation of concerns — it's a real Windows
constraint. `paddlepaddle-gpu` and PyTorch (used by the detector) each
bundle their own private CUDA/cuDNN DLLs under identical filenames, and
having both loaded in one process is unreliable (inconsistent `WinError
127` failures, not a clean crash every time). Running OCR as its own
process sidesteps that entirely, and as a side effect lets the GPU-bound
OCR stage scale independently of the cheap detection stage — which was
always the right shape for production regardless of the Windows DLL issue.
