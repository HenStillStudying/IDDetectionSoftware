# Data handling & compliance


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
