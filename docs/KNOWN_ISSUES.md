# Known issues & limitations

Everything below is **open**: known, measured where possible, and not fixed yet.
The evidence behind each entry (how it was found and what was tried) is in
[`CHANGELOG.md`](CHANGELOG.md). When something here gets fixed, move it to the
changelog.

## Trust: what the output does *not* prove

- **`status: ok` does not mean the card is genuine.** It means "something that
  looks like a KTP was found and its fields were read." A fabricated card passes
  every check here. The project's own synthetic cards are detected 30/30 and
  carry structurally valid NIKs. See the full caveat in [`API.md`](API.md).
- **NIK consistency catches only careless fakes.** It checks that the birth date
  and gender encoded in the NIK match the printed ones. Anyone who knows the NIK
  format can make a fake agree with itself. A pass on gender alone is a coin
  flip. On 28 consented real KTP inputs it passed on 8 genuine cards and
  **falsely flagged 1**. Its owner confirmed the printed birth date does match
  the NIK, so OCR misread one of the two, getting one digit wrong in each of
  day, month and year, while reporting ~1.0 confidence on both. So a
  mismatch is not proof of forgery, and OCR confidence near 1.0 doesn't
  guarantee a correct read. A mismatch now makes the result
  `low_confidence`, but a misread that happens to stay self-consistent, or
  hits a field the check doesn't compare, can still come back `ok`. The check was also
  skipped on 7 of the 16 cards the old detector read, because the NIK was
  invalid or the birth date wasn't read confidently. These consistency
  figures predate the retrained detector and haven't been re-measured.
- **Forensics is EXIF-only.** It only reports that a photo has no EXIF data,
  which is common and innocent. Error-level analysis (ELA) was built,
  evaluated, and shelved because it flagged the genuine real card as
  suspicious.
- **No stronger anti-forgery layer exists yet.** These are not built: a
  photo-vs-screen/print classifier, physical security-feature checks, face match
  plus liveness, a Dukcapil registry lookup, and human review.

## Accuracy & coverage

- **Real-card accuracy rests on 27 cards, and there is no clean test set
  left.** The detector is now trained on all 27 consented real KTP fronts
  (plus one card back). Its expected detection rate, **26/27**, comes from
  4-fold cross-validation, not from cards it has never seen. Small sample:
  one card more or less moves it by ~4 points. Any new consented photos
  should be kept out of training as a fresh test set.
  - Cross-validation could be optimistic if two inputs show the same
    physical card and landed in different folds. Not checked.
  - Every fold trained once. Retraining on synthetic data alone swung
    between 1 and 17 of the real cards across seeds; with real cards in
    training, all four folds did well, but that is not a seed study.
- **The detector's weights are personal data.** They are derived from real
  cards, so they live outside the repo and OneDrive and can't be shared or
  committed. Anyone else running this project has to train their own, and
  the synthetic-only recipe they can reproduce finds far fewer real cards
  (16/27 for the old production model).
- **3 of 27 real cards are found but their NIK still fails validation.**
  Through the full pipeline, 24 of the 27 NIKs are valid. The remaining
  failures are OCR reads, not detection. The earlier read rates on the
  weaker fields (`tanggal_lahir`, `provinsi`, `kota_kabupaten`,
  `golongan_darah`, measured on the old detector's 16 cards) haven't been
  re-measured. Field values aren't compared against ground truth, so these
  are read rates, not accuracy.
- **Free-text fields are the weak spot.** On synthetic data: `alamat` 16/30,
  `kota_kabupaten` 24/30, `nama`, `kewarganegaraan` and `berlaku_hingga` 25/30.
  Most failures are single-character OCR noise from the fast `PP-OCRv6_small`
  tier, not logic bugs.
- **Steep camera angles are not detected.** A strongly tilted real photo
  returned `no_card_detected`. Accepted for now: users are expected to take a
  reasonably straight photo.
- **A card-like object next to a KTP can cause a refusal.** Photos with more
  than one card are now refused (`multiple_cards_detected`). In one probe of 20
  synthetic KTP + look-alike scenes, the detector gave one look-alike a 0.78
  box, which is enough to refuse that photo. That is the safe direction to
  fail, and the message tells the user to photograph one card at a time.
- **A blurry second card can go unnoticed.** With a real KTP and SIM farther
  apart and both out of focus, the detector never boxed the SIM and read the
  KTP normally. Harmless when the other card isn't a KTP, but if it were a
  second KTP this would be the old silent pick again.
- **A misread NIK colon glued to the number is not repaired.** The fix for a
  colon read as a leading `1` only applies when OCR leaves a gap between it and
  the digits (`l 3205…`). Glued on (`l3205…`), it can't be told apart from a
  real extra digit, so the NIK stays 17 digits and the result is
  `low_confidence`.
- **The 10 MB upload limit rejects some very high-resolution photos** such as
  full-size shots from high-megapixel phone modes. Those get a 413.

## Security & operations

- **One shared API key** (`KTP_API_KEY`). There are no per-client keys, no
  rotation, and no audit log.
- **`/demo` sends the API key over whatever connection it's opened on.** With no
  HTTPS yet, only use it locally or through an SSH tunnel.
- **The compliance work is scoped but not built.** This covers consent,
  per-user auth and audit trail, retention, data-subject rights, and a DPIA
  under UU PDP. See
  [`COMPLIANCE.md`](COMPLIANCE.md).
- **Not production-hardened:**
  - There is no HTTPS or reverse proxy yet.
  - Rate limiting behind a proxy would see every client as one IP.
  - There is no monitoring or alerting.
  - `KTP_OCR_INTERNAL_KEY` (api→ocr auth) is off by default.
- **The VPS runbook is untested.** [`infra/DEPLOY.md`](../infra/DEPLOY.md) has not
  been run on a real server yet. The compose stack itself is verified locally.
- **Platform coverage is limited:**
  - The GPU path is unvalidated end to end.
  - ARM (for example Graviton or Raspberry Pi) is untested.
  - Images are x86_64 CPU builds.

## Tooling blind spots

- **Dependabot deliberately skips some updates:**
  - `redis>=6`, because `arq` needs `redis<6`.
  - `numpy>=2.4`, because `paddlex` needs `numpy<2.4`.
  - Python base-image minor and major bumps.

  Those need manual review. Dependabot also can't see the inline `pip install`
  pins in the Dockerfiles.
- **CI runs the light test suite only.** Tests that need `ultralytics`, torch,
  or paddle (for example the ultralytics lockdown test) are skipped in CI and
  must be run locally. The same goes for a Dependabot PR that bumps one of
  those heavy dependencies: a green CI run proves nothing about it, so rebuild
  the images and smoke-test before merging.
