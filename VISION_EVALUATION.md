# VISION_EVALUATION.md — Could a vision LLM replace the OCR pipeline?

**Decision: no.** We evaluated reading the weekly schedule image with a
vision-capable LLM (DeepSeek V4.1 Flash) instead of — or alongside — the
PP-OCRv6 pipeline. The model reads the image well, and the approach is much
simpler, but it cannot provide the validated, deterministic output this
project is built on. **We keep the validated implementation.** This document
records the experiment so the question does not have to be re-litigated.

- **Date:** 2026-09-15
- **Model:** `deepseek-v4.1-flash` (vision), provider `opencode-go`
- **Image:** `testdata/schedules/10.08.2026-17.08.2026_page-0001-2.jpg` (1755×1240)
- **Ground truth:** `testdata/reference/rethymno_duty.html` (structured municipality duty page, 12–14 Aug 2026)
- **Pipeline build:** current source, Docker (Debian bookworm), 2 CPUs, CPU-only

---

## 1. What was compared

| | Vision LLM | Current pipeline |
|---|---|---|
| Reading | one image + prompt, no preprocessing | OpenCV preprocessing → PP-OCRv6 → layout |
| Post-processing | none | layout → parse → validate against the 73-entry catalog |
| Output | free text / model-decided structure | deterministic JSON, per-entry confidence, image SHA-256 |

The comparison focused on the question that matters: *does it understand the
picture better or faster than what we do here?*

## 2. Accuracy

| Check | DeepSeek V4.1 Flash | Pipeline |
|---|---|---|
| 4×2 grid, day names, dates | correct | correct (day names derived from dates) |
| Entries read | 21/21 | 21/21 |
| Phones read | 21/21 matched the pipeline | 21/21 valid (`phone_valid: true`) |
| Catalog match | n/a | 21/21 `catalog_match`, validation score 1.0 |
| vs. official duty page (12–14 Aug) | matches | 7/7 entries match |
| Raw text quality | clean Greek | **mixed-script output: 61/146 lines (42%) mix Greek and Latin, and 14 lines contain Cyrillic readings that never appear in the source** |

Raw PP-OCRv6 output examples (from the same run):

```
y=231 x=370 c=0.700 AATZAN            # ΠΑΠΑΤΖΑΝΗ
y=231 x=1280 c=0.954 APANAAKH         # ΔΡΑΝΔΑΚΗ
y=231 x=938 c=0.900 IPINIΩTAKHΣHAIAΣ  # ΠΡΙΝΙΩΤΑΚΗΣ ΗΛΙΑΣ
y=842 x=685 c=0.899 AIAΣKO∑           # ΛΙΑΣΚΟΣ
y=307 x=328 c=0.772 IΣΩAОTОΔMAPXЕIO   # ΠΙΣΩ ΑΠΟ ΤΟ ΔΗΜΑΡΧΕΙΟ (Cyrillic О/Е)
```

None of this reaches the final JSON: normalization plus the reference
catalog resolve it. That is the point of the current design. (Part of the
script mixing mirrors the image's own styling — `THΛ.` for `ΤΗΛ.`,
`ΔHMOKPATIAΣ` for `ΔΗΜΟΚΡΑΤΙΑΣ` — but the garbled names and the Cyrillic
readings are genuine recognition errors.)

### Both sides stumble on the same cell

Saturday 15/08, second 08:00–21:00 entry:

- PP-OCRv6 reads `AIAΣKO∑` (confidence 0.899) — fixed by the catalog to `ΔΡΑΝΔΑΚΗ - ΛΙΑΣΚΟΣ`.
- At page scale the vision model misread it too (as "ΔΙΔΑΣΚΟΣ"); a 2× zoom of the cell resolved it to `ΛΙΑΣΚΟΣ`.

Vision is not error-free, and the pipeline's catalog rescue is what turns a
misread into a correct, canonical answer.

### Where vision is genuinely better

- **Reading.** Greek text comes out clean in one pass; the look-alike chaos
  above never happens.
- **Names not in the catalog.** The catalog can only fix entries it knows.
  A new pharmacy appears in the image as garbage to PP-OCRv6 with no rescue
  path; the vision model reads it correctly. This is the strongest argument
  for a vision *fallback*, not a replacement.

### Where the pipeline is genuinely better

- **Canonicalization.** The image shows `ΚΕΡΑΜΙΑΝΑΚΗ`, `ΜΑΓΙΑΝΙΩΤΗΣ`,
  `ΛΙΑΣΚΟΣ`; the validated output is `ΜΑΣΤΟΡΑΚΗ - ΚΕΡΑΜΙΑΝΑΚΗ`,
  `ΜΑΓΓΑΝΙΩΤΗΣ ΣΤΑΜΑΤΗΣ`, `ΔΡΑΝΔΑΚΗ - ΛΙΑΣΚΟΣ` — the names the municipality
  and Google Places use.
- **Verification.** Every phone number is checked, every entry is matched
  against the official catalog, unknown entries produce warnings rather than
  confident-sounding guesses.
- **Determinism.** Same image in, same JSON out — including the embedded
  SHA-256 and per-stage timings.

## 3. Speed and cost

| | Vision LLM | Pipeline |
|---|---|---|
| Latency | ~11 s wall for the image-reading turn (single API call, 1.2k input / 0.3k output / 2.7k reasoning tokens) | 6.2 s parse (OCR 5.3 s, OpenCV 0.9 s); 7.1 s average over the 3 test images |
| Hardware | none locally (network + provider required) | 2 CPU cores, no network, no cloud |
| Cost | cents per run (the whole evaluation session, every turn, cost $0.019) | free per run after build |

For a once-a-day ingestion both are fast enough; latency was not the deciding
factor. The pipeline's no-network, no-provider property was.

## 4. Simplicity

| | Vision LLM | Pipeline |
|---|---|---|
| Code | one call | 5,785 LOC (non-test) across 10 `internal/` packages |
| Dependencies | an HTTP client | OpenCV 4.6, ONNX Runtime 1.23.2, ~157 MB of embedded models, an OCR worker subprocess |
| Failure modes | provider outage, model change, non-determinism, no confidence signal | glibc/GoCV/ORT quirks (solved; see [DEVELOPMENT_BLOG.md](DEVELOPMENT_BLOG.md)) |

The vision approach would delete `internal/ocr` (1,364 LOC),
`internal/vision` (671 LOC), the ONNX/OpenCV dependencies and the worker
split. That simplicity is real — but it would also delete the guarantees
listed above, and it would send the image to a third party, which contradicts
the project's "local, no cloud" promise.

## 5. Why we stick with the current implementation

1. **Trust.** A wrong phone number sends someone to a closed pharmacy. The
   pipeline validates every field against authoritative sources and emits
   confidence; a generative model cannot be held to that standard.
2. **Determinism.** Downstream consumers (HTTP cache, golden tests) depend on
   stable, reproducible output.
3. **Offline, CPU-only operation.** The whole pipeline runs in the container
   with no network and no provider account.
4. **Cost of switching is higher than the gain.** The reading step is not the
   bottleneck; validation is the product.

Vision stays a candidate for **auxiliary** use only — e.g. a weekly
cross-check or a fallback reader for entries the catalog cannot match — and
would still have to pass through the existing `validate` layer. No such path
is implemented; the pipeline remains the single source of truth.

## 6. Caveats

- One image, one model, one run: the test image is a clean, machine-generated
  table. Skewed photos or low-resolution scans were not evaluated.
- The vision model received the image without preprocessing; harness
  resizing/tiling may differ from a direct API call, so exact per-glyph
  accuracy is not directly comparable.
- Model behavior changes over time; the pipeline's pinned models and
  reference catalog do not.

## 7. Reproduce

```sh
# pipeline: parse one image (Docker, 2 CPUs)
docker run --rm --cpus 2 --entrypoint go -v "$PWD:/data" -w /data \
  rethymno-emergency-pharmacy:dev \
  run ./cmd/rethymno-emergency-pharmacy parse --models models \
  testdata/schedules/10.08.2026-17.08.2026_page-0001-2.jpg

# pipeline: stage timings across all test images
docker run --rm --cpus 2 --entrypoint go -v "$PWD:/data" -w /data \
  rethymno-emergency-pharmacy:dev \
  run ./cmd/rethymno-emergency-pharmacy benchmark --models models testdata/schedules
```

For the vision side, attach the same JPEG to a vision-capable model and ask
for the day/shift/pharmacy table; then compare against the pipeline JSON and
`testdata/reference/rethymno_duty.html`. That is exactly the comparison this
document summarizes.
