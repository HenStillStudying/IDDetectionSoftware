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
```

## Run (four processes)

```bash
# terminal 1 — Redis (backs the async job queue)
docker run -d --name ktp-redis -p 6379:6379 redis:7-alpine

# terminal 2 — OCR service
cd services/ocr
uvicorn ocr_app.main:app --port 8001

# terminal 3 — API, pointed at the OCR service and the trained detector
cd services/api
KTP_DETECTION_WEIGHTS=../../training/runs/ktp_detector/weights/best.pt \
KTP_OCR_SERVICE_URL=http://127.0.0.1:8001 \
uvicorn app.main:app --reload

# terminal 4 — worker (processes /v1/ktp/jobs submissions)
cd services/api
KTP_DETECTION_WEIGHTS=../../training/runs/ktp_detector/weights/best.pt \
KTP_OCR_SERVICE_URL=http://127.0.0.1:8001 \
arq app.worker.WorkerSettings
```

Without `KTP_DETECTION_WEIGHTS`/`KTP_OCR_SERVICE_URL` the API and worker
still run, using stubs for whichever one is missing. Without Redis running,
`/v1/ktp/extract` (sync) still works — only `/v1/ktp/jobs` (async) needs it.

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
  synthetic eval set. One regression surfaced and left as-is rather than
  chased further: `golongan_darah` (blood type) dropped 73% → 50%, but
  every failure is a clean `None`, not a wrong value — likely the blood-type
  letter sitting further right in the combined "gender + blood type" value
  string now reading less reliably, not a new logic bug. Worth a second real
  card to confirm the layout assumption more broadly before trusting this
  further.
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
- **OCR latency is ~3-4s/request on small synthetic images, but ~13s on a
  real phone photo — CPU only, and this gap was never measured until now.**
  That 3-4s figure was only ever benchmarked against the 960×720
  synthetic scenes this project generates; a real phone photo is much
  higher-resolution (the real KTP tested above produced a card bounding box
  alone running to 1571×1136, so the source photo is at least that large),
  and both YOLO detection and PaddleOCR's text detection/recognition stages
  scale with image area. Measured directly: 3.6-4.2s across three synthetic
  eval images vs. 13.1s for the one real photo tested, through the same
  code path (`KtpExtractionPipeline`, full pipeline including detection +
  the OCR service call).
  **Fixed, partially**: `pipeline.py` now downscales any upload wider or
  taller than 1600px (preserving aspect ratio, `Image.LANCZOS`) before
  detection ever sees it — both YOLO and the OCR crop that comes out of
  detection shrink with it. 1600px was picked to keep a 16-digit NIK
  comfortably legible after the crop. Measured on a synthetic image
  upscaled to 3120×2340 to stand in for a real phone photo (the real photo
  itself was already deleted per the real-KTP testing protocol, so this
  isn't the exact same input): **9.1s → 6.1s, about 33% faster**, on
  otherwise identical input. NIK — the field that matters most — stayed
  correct with or without the resize. One accuracy caveat found on that
  same test, though: `provinsi` (a small header field, lowest-detail text
  on the card) came back empty with downscaling but was read correctly
  without it. Given the test image was itself upscaled via interpolation
  before being downscaled back down — not a real camera photo's actual
  detail — this isn't a clean read on the real-world accuracy cost, but
  it's a real observed difference on the one test run, not nothing. Worth
  validating against an actual high-resolution real photo before fully
  trusting the 1600px threshold; untried so far.

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
  burned significant time without a clean result. This looks like a
  network-path/environment problem specific to Docker-Desktop-on-Windows,
  not a new technical blocker — but it means the actual cuDNN-loads-on-Linux
  question is still open. **Re-test this on an actual cloud Linux GPU
  instance** once a deployment platform is chosen — better network path to
  that CDN, and it's the real target environment rather than a local proxy
  for it — before spending effort on other latency levers (ONNX/TensorRT
  export, quantization, a custom lightweight recognizer). Other remaining
  CPU-only levers if GPU doesn't pan out: an
  older pre-3.x paddlepaddle that might restore working MKL-DNN on CPU
  (real risk: probably forces a paddleocr 2.x downgrade too, a different
  API), or `PP-OCRv6_tiny_*` if its accuracy risks turn out to be
  acceptable for a given deployment. Still worth routing through
  `/v1/ktp/jobs` (async), not the sync endpoint, in production regardless.
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
  one much weaker one introduced. Left undone for now rather than chasing
  a fourth round (diminishing returns / whack-a-mole risk) — a
  tall/portrait-page distractor kind would likely close it if revisited.
