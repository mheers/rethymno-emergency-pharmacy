# TYPESAFE_EVALUATION.md — Can System One judgments replace the fragile parsing?

**Decision: yes — as an auxiliary adjudication layer, not as a reader.**
TypeSafe's System One models do not see the image, so the OCR pipeline stays and
the [VISION_EVALUATION.md](VISION_EVALUATION.md) decision is untouched. But the
semantic glue *around* OCR — deciding which catalog entry a garbled reading
refers to, what a line is, whether a phone-keyed match is trustworthy — is
currently hand-tuned string heuristics and edit-distance thresholds. Those are
exactly the decisions System One questions are built for, and they can be added
behind code that still validates, canonicalises and owns the output. Two
prototypes are implemented: `internal/adjudicate` behind `cmd/merge-golden`
(adjudication on by default, `-judge=false` forces the deterministic path; §4
measures it) and the catalog-identity adjudicator, measured in §4.1 and wired
into the runtime behind the opt-in `--judge` flag. The runtime pipeline is
otherwise unchanged, and this document records the evaluation so the work does
not have to be re-derived.

- **Date:** 2026-09-19
- **Model:** `jev-latest` (a concrete version must be pinned for reproducibility)
- **Endpoint:** `POST https://api.typesafe.ai/v1/systemone`
- **Scope:** static review of the current revision, plus the golden-merge
  experiment in §4. Every threshold and question wording must be re-evaluated
  on the full corpus before it is trusted.
- **Prototype:** `internal/adjudicate`; `cmd/merge-golden -judge`
- **Prior decision:** [VISION_EVALUATION.md](VISION_EVALUATION.md) — a vision LLM
  does not read the schedule; the validated pipeline stays the source of truth.

---

## 1. Why this is not a repeat of the vision evaluation

VISION_EVALUATION.md rejected a vision LLM *reading* the image because it could
not canonicalise names against the catalog, verify phone numbers, or guarantee
deterministic output. That argument stands and is not revisited here.

TypeSafe is a different kind of model. It takes text **state** and returns typed
judgments — a [Noul](https://docs.typesafe.ai/primitives/noul) (probability of
yes/no), a [Choice](https://docs.typesafe.ai/primitives/choice) (one option from
a set, plus the full probability distribution), or a
[Score](https://docs.typesafe.ai/primitives/score). It cannot read the image and
does not try to. It answers the questions the rejected approach got wrong — but
*after* OCR has produced text, with candidates supplied by the catalog, and with
code in charge of the result:

| | Vision LLM (rejected) | System One judgments |
|---|---|---|
| Input | the image | OCR text + code-supplied candidates |
| Output | freely generated names/structure | typed selection/verification |
| Confidence | none | calibrated probabilities + confidence |
| Failure handling | confident-sounding guesses | code consumes, gates, or ignores |
| Runtime | cloud, non-local | cloud, optional; build-time use is local |

The two strongest arguments *for* the pipeline in the vision evaluation —
canonicalisation and verification — are preserved here by construction: the
judgment never writes a pharmacy name into the output; it selects an existing
catalog entry or reports a probability that code turns into a warning.

## 2. The fragile surface

| Location | What it does today | Failure mode |
|---|---|---|
| `internal/parse/parse.go:105-207` | Entry segmentation by phone lines (`parseShift`), per-line classification (`looksLikeName`, `looksLikeAddress`, `isJunkLine`, `isPhoneLine`), Greek/Latin address merge | ~200 lines of ordered rules; a missed phone line merges two pharmacies; a misclassified line silently moves into name or address |
| `internal/validate/validate.go:178-283` | Catalog lookup by phone digits; fuzzy choice among same-phone entries by Levenshtein similarity (`<0.45` warning; `>=0.55` / `>=0.3` match bands) | thresholds are tuned guesses; the wrong pick among candidates is silent |
| `internal/pipeline/pipeline.go:425-477` | `FillFromCatalog` overwrites name/address and attaches Google details on a phone match; `chooseCatalogReference` does the selection | an OCR phone that mangles into a different *valid* catalog phone rewrites the entry with no similarity gate; after the fill, `ValidateResult` compares the catalog against itself, so the mismatch warning cannot fire |
| `internal/validate/validate.go:122-133, 227-243` | `ValidateAddress` passes any string containing a 1–4 digit run; confidence is a fixed weight formula (`0.35/0.3/0.2/0.05/0.1`) | `"TEPAKAPH96 96GERAKARISTR"` (README example) reaches JSON at `confidence: 0.9` |
| `cmd/merge-golden/main.go:214, 434-457` | Golden↔reference catalog merge by phone, then name similarity thresholds `0.55` / `0.4` | merge errors are baked into the embedded catalog permanently |
| `internal/extract/extract.go:41-93` | Schedule-image regex over the FSKriti page; `SelectCurrent` silently falls back to the newest image | page drift or an out-of-range week can serve a plausible-looking wrong week |

## 3. Opportunities

### A. Catalog identity adjudication (runtime, opt-in) — highest value

**Today.** A phone lookup can return several catalog entries. Four of the 73
catalog numbers resolve to two entries each; every pair shares the phone and an
identical or near-identical address (renamed or successor businesses at the
same premises):

| Number | Entries |
|---|---|
| 2831025123 | Μπροκαλάκη Ζαχαρένια / Περιστεράκη Τερέζα (Κουμουνδούρου 2) |
| 2831027264 | Βανδώρου Άννα / Πρινιωτάκης Ηλίας - Παναγιώτης (Λ. Κουντουριώτου 81) |
| 2831054706 | Δαμβακεράκης / Τσιομπίκας Γρηγόριος (Εμμ. Παχλά 33) |
| 2831055212 | Βαρούχα - Αναγνωστάκης / Καλαφατάκη Ελένη - Γεωργία (Δημητρακάκη 21) |

The OCR'd name is then the intended disambiguating signal — and it is the
noisiest one; the similarity score only consults it when it beats the address
score, which an exact address match to the wrong twin never loses (§4.1).
`ChoosePhoneReference` (`validate.go:264`) and `chooseCatalogReference`
(`pipeline.go:458`) decide it with raw edit-distance similarity, and
`FillFromCatalog` overwrites name/address/coordinates/Google details on the
result without a similarity gate. Validation runs *after* the overwrite, so a
wrong pick is compared against itself and produces no warning. Thirteen catalog
entries carry Google enrichment; a wrong pick can attach another pharmacy's
coordinates, hours and photos.

**Design.** Code retrieves candidates by phone; the judgment selects among them.
One question per ambiguous entry, all batched into a single request (questions
run in parallel). Sketch:

```json
{
  "state": {
    "ocr_entry": { "lines": ["KAAYΦATAKH EΛENH", "ΔHMHTPAKAKH 21", "THΛ 2831055212"] },
    "candidates": [
      { "id": "varoucha",   "name": "Βαρούχα - Αναγνωστάκης",    "address": "Δημητρακάκη 21" },
      { "id": "kalafataki", "name": "Καλαφατάκη Ελένη - Γεωργία", "address": "Δημητρακάκη 21" }
    ]
  },
  "questions": {
    "entry_0_catalog_identity": {
      "type": "choice",
      "instructions": "Which catalog entry do these OCR readings describe?",
      "criteria": {
        "varoucha": "The readings refer to Βαρούχα - Αναγνωστάκης",
        "kalafataki": "The readings refer to Καλαφατάκη Ελένη - Γεωργία",
        "none": "Neither candidate fits the readings"
      }
    },
    "entry_0_same_pharmacy": {
      "type": "noul",
      "instructions": "Do the OCR name and address readings describe the selected candidate pharmacy?",
      "criteria": {
        "true": "The readings are a plausible noisy rendering of the candidate",
        "false": "The readings point to a different business or are too garbled to tell"
      }
    }
  }
}
```

**Consumption rules (code).** Accept the selection only when `confidence` clears
a corpus-tuned threshold and the Noul is high; `none` or low confidence keeps
today's deterministic pick and raises the existing `Discrepancy` warning. The
model never generates a pharmacy — candidates are catalog entries chosen by code,
so every possible answer is a verified record or an explicit "none". Implemented
and measured: `internal/adjudicate/identity.go` (`AdjudicateIdentity` builds the
questions, `AcceptIdentity` applies the gates); §4.1 records the experiment and
the runtime wiring behind `--judge`.

**Why this is the best fit.** It is the reranking pattern
([rerank cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe)): retrieve
with code, select with judgment. The model's answer is one of at most 73 known
entries.

### B. Line classification and entry segmentation (fallback or build-time)

**Today.** `reconstructEntry` decides each line's role with Greek/Latin letter
presence, digit counts, a stop-word list (`parse.go:65`), and eleven
address-description prefixes (`parse.go:71`); the residual "ambiguous" rule
(`parse.go:175`) routes a line into the address when it contains digits, into the
name otherwise. `parseShift` closes an entry on each detected phone line; a
missed phone line merges two pharmacies into one entry. `mergeAddressLines`
(`parse.go:191`) keeps only the Greek address when both scripts are present, and
`NameLat`/`AddressLat` are never populated from OCR — only from the catalog
(`pipeline.go:426`).

**Design.** Send the shift's raw lines as state (`layout.Line.RawText` preserves
the unmodified OCR output for exactly this) and ask, per line,
name / Greek address / Latin transliteration / phone / location description /
junk, or per boundary whether a group of lines describes one pharmacy or two.
Code still assembles, cleans and validates the fields afterwards. Two deployment
modes avoid the runtime/offline tension:

- **fallback-only**: judge only entries whose deterministic parse produced an
  empty name/address or a validation warning;
- **build-time**: judge once over the test corpus, review the result, and check
  in an adjudication table — the runtime pipeline stays offline and
  byte-deterministic.

### C. Golden-catalog merge (build-time) — recommended first step

**Today.** `cmd/merge-golden` matches the Google-enriched golden catalog to the
reference catalog by phone, then resolves same-phone entries with name
similarity thresholds `0.55` (`main.go:214`) and `0.4` (`main.go:457`). The
input workflow already talks to the network; the output is embedded in the
binary permanently, so a bad threshold call is a lasting data defect.

**Design.** This is the
[entity-alignment cookbook](https://docs.typesafe.ai/cookbooks/entity_alignment):
one Score question per candidate pair whose three levels are the three things
that can happen — merge, leave unlinked, or send to a curator. Emit a merge
report with the answers and probabilities; apply confident merges, and route the
rest to review. No runtime contract changes, no determinism concern, and the
fragile thresholds disappear. This design is prototyped and measured in §4.

### D. Plausibility verification (runtime, opt-in, warnings only)

**Today.** `ValidateAddress` accepts any house-number-like digit run;
`ValidateNameGreek` accepts any string with two Greek letters. Unmatched entries
(the ones the catalog cannot rescue — exactly where the vision evaluation said
the pipeline is weakest) can reach consumers with plausible-looking garbage at
high confidence.

**Design.** A Noul per unmatched entry — "Is this a plausible Greek pharmacy
name?", "Is this a plausible street address for Rethymno?" — consumed as
warnings and, optionally in a later phase, as an extra dimension in the
confidence formula. The fixed weights remain code's business in the first phase.
The judgment reports; code decides; nothing is rewritten.

### E. Week and page gates (runtime, opt-in)

**Today.** `FindScheduleImages` relies on the FSKriti URL pattern; `SelectCurrent`
falls back to the newest image when today is outside every range, and
`ParseDutyPage` returns zero days on markup drift — both silently.

**Design.** Cheap Noul gates: "does this page contain the weekly Rethymno
pharmacy duty schedule?" and "is this the week containing `<date>`?". A fallback
selection or a zero-day parse should also become an explicit warning regardless;
the real fix for HTML drift is a proper parser, so this is a safety net, not the
primary tool.

## 4. Experiment: golden-merge adjudication (2026-09-19)

The C design is implemented: `internal/adjudicate` (the System One client, the
Score+Noul question builder, score routing) and `cmd/merge-golden`, where
adjudication is on by default (`-judge=false` forces the deterministic path)
and `-judge-strict` refuses to write the catalog while curator decisions or
judge fallbacks are unresolved. The similarity thresholds remain the fallback
on any API error or missing key. The experiment measured the adjudicator
against those thresholds.

```sh
# reproduce (uses TYPESAFE_API_KEY; writes the raw data to TYPESAFE_EXPERIMENT_OUT)
TYPESAFE_EXPERIMENT=1 TYPESAFE_EXPERIMENT_OUT=/tmp/ts-merge \
  go test -run TestMergeAdjudicationExperiment -v ./internal/adjudicate

# the merge tool itself: adjudication is the default; strict blocks the write
# while curator decisions or fallbacks are unresolved
go run ./cmd/merge-golden -judge-strict -judge-report /tmp/judge.json \
  -golden <golden.json> -out <reference.json>
```

**Dataset.** 14 phone groups / 140 candidate pairs built from the 73-entry
reference catalog: the four shared-phone pairs in full, "variant-only" versions
of them (the golden source never states the exact name, only short forms), and
single-entry cases with unrelated distractors. Golden records are synthetic name
variants of real entries — the real golden catalog is not in this repository —
and the ground truth is which entry each variant derives from. Model:
`jev-1.13.0`.

**Pair-level outcomes (arm A, one request per group):**

| Ground truth | Similarity thresholds | System One |
|---|---|---|
| same (84) | 71 same / 5 curator / 8 different | **84 same** / 0 / 0 |
| curator (2) | 0 / 0 / 2 different | 0 / **2 curator** / 0 |
| different (54) | 0 / 0 / 54 different | 0–3 same / 22–27 curator / rest different |

**Reference-level merges.** 22 expected merges; the thresholds found 19 — all
three misses were variant-only groups, where short forms like `ΒΑΡΟΥΧΑ` against
`Βαρούχα - Αναγνωστάκης` score below the 0.55 gate and are dropped. System One
found 22/22, and no wrong entry was ever enriched. The address-only name
`ΦΑΡΜΑΚΕΙΟ ΔΗΜΗΤΡΑΚΑΚΗ` — genuinely ambiguous — went to the curator queue
instead of being silently unlinked. The live `-judge` run on the same
shared-phone group merged both entries (scores 1.94/0.91 and 1.90/0.85) where
the threshold path merged only one.

**Strict mode.** `-judge-strict` was exercised on fixtures: it wrote output when
every reference decision was resolved (including a group whose cross pairs were
curator but each entry matched its own record), and refused to write when the
address-only `ΦΑΡΜΑΚΕΙΟ ΔΗΜΗΤΡΑΚΑΚΗ` record resolved to two curator decisions
(score 1.41–1.43, confidence 0.34–0.37, same-name Noul 0.18–0.20). The report
still captures those decisions, so the curator queue is reviewable after a
refused run.

**What the curator band is.** Of the 54 pairs that must not merge, 22–27 landed
on curator, at confidence ≤ 0.41 and same-name Noul ≤ 0.29. Curator is safe but
noisy: read it as "not merged, logged for review", never as a merge.

**Repeatability.** Four runs of the same dataset produced 3, 0, 1 and 1
pair-level false merges; the inspected ones sat at the 1.5 cut point with
confidence 0.26–0.37 and same-name Noul ≤ 0.27, while the ref-level merges taken
had confidence ≥ 0.69 and same-name ≥ 0.88 (last run). A 0.5 confidence gate
therefore falls in an empty band: it removed every observed false merge and kept
all 22 real ones. Two identical runs differed on 6 of 140 pair verdicts (4.3%),
all at confidence ≤ 0.40 and within ±0.1 of a cut point; no verdict at
confidence ≥ 0.5 changed between runs.

**Batching.** Putting all 14 groups in one request (Score-only) disagreed with
per-group requests on 30 of 140 pairs: 19 different→curator, 4 same→curator,
3 curator→same, 2 curator→different, 2 different→same. One request per phone
group is the right granularity; do not pack the whole catalog into one call.

**Cost.** 14 per-group requests: 47.9k input tokens — about $0.002 at the listed
$42/Btok — and 5.5 s wall. A single phone group: 1.6k tokens in well under a
second. This is a build-time tool, so cost is irrelevant next to the data-quality
gain.

**Language caveat.** TypeSafe's docs say English is the primary training
language and to test other languages. On this Greek dataset the judgments were
accurate (same-name Noul ≥ 0.79 on true pairs), but that must hold on the full
corpus before the runtime use in A is trusted.

### 4.1 Experiment: catalog-identity adjudication (2026-09-19)

Opportunity A is prototyped as a rerank: `internal/adjudicate/identity.go` builds
one Choice per entry (the code-supplied candidates plus an explicit `none`) and
one same-pharmacy Noul per candidate, and `AcceptIdentity` applies the
confidence and Noul gates. It is not wired into the runtime pipeline; the
experiment measures it first.

```sh
TYPESAFE_EXPERIMENT=1 TYPESAFE_EXPERIMENT_OUT=/tmp/ts-identity \
  go test -run TestIdentityAdjudicationExperiment -v ./internal/adjudicate
```

**Dataset.** 57 readings over the four shared-phone pairs (model `jev-1.13.0`):
41 decisive readings of the eight catalog entries — exact names, short forms,
first words, Greeklish transliterations, recognizer look-alike script (the
`KAAYΦATAKH` / `ΔHMHTPAKAKH` failure modes in the corpus), and swapped-character
typos — plus 16 readings that must not resolve to a candidate: address-only,
street-derived names (`ΦΑΡΜΑΚΕΙΟ ΔΗΜΗΤΡΑΚΑΚΗ`), another pharmacy's name, and an
unrelated business. Ground truth is the entry each decisive reading derives
from. The readings are simulated: the schedule corpus does not contain these
four numbers (the same synthesis caveat as §4).

**Results (three runs; picks were identical on all 57 readings in every run).**

| Reading class | n | Similarity baseline | System One |
|---|---|---|---|
| decisive: correct pick | 41 | 30 | **41** |
| decisive: declined | — | 0 | 0 |
| no-pick: declined | 16 | 0 | 9 |
| no-pick: forced pick | 16 | 16 | 7 |

- The baseline loses exactly where the merge experiment lost: short forms
  (`ΒΑΡΟΥΧΑ` against `Βαρούχα - Αναγνωστάκης` score below the gate).
- The acceptance signal `min(choice confidence, same-pharmacy Noul)` separates:
  correct picks 0.77–0.96, wrong picks 0.17–0.75. A gate at **0.8** on both
  values accepts 40/41 correct picks, with a 0.06 margin between the lowest
  accepted correct (0.81) and the highest wrong signal (0.75); the one rejected
  correct pick is the `ΒΑΡΟΥΧΑ` short form, which falls back to the
  deterministic pick.
- The Noul is what catches the forced picks: on the address-only reading of the
  `Δαμβακεράκης / Τσιομπίκας` pair the Choice said 0.93–0.94 while the Noul
  stayed at 0.75; for another pharmacy's name, Choice 0.85–0.88 with Noul
  0.35–0.36.
- Verdicts are stable: three runs produced identical picks on all 57 readings,
  with signals moving by at most 0.06. Only the `ΒΑΡΟΥΧΑ` short form sits
  near the gate.
- Batching fails, as in §4: all 57 readings in one request agreed with the
  per-reading requests on only 20–22/57 picks. One request per ambiguous entry.
- Cost: 57 requests, 50.7k in / 5.1k out tokens, ~20 s wall.

**Reading of the result.** The judgment repairs the runtime's silent wrong-pick
class — a phone match that rewrites the entry with the wrong twin — and its Noul
flags the readings that cannot be resolved. It does not decline every
unresolvable reading (7/16), but the Noul keeps those below the gate, so with
`MinConfidence` and `MinSamePharmacy` at 0.8 they become the deterministic pick
plus a warning instead of a silent overwrite. This supports the §3A consumption
rule; the runtime wiring landed the same day.

**Wiring (2026-09-19).** `pipeline.FillFromCatalogJudged` consults the judge only
when a phone resolves to several catalog entries, one request each; an accepted
verdict replaces the similarity pick, every other outcome keeps it and adds a
warning (additive; the JSON schema does not change). The fill records its picks,
and `ValidateResultWithPicks` validates each pharmacy against that entry, so the
validation block agrees with the filled name instead of re-picking behind it.
The public client enables all of this with `ClientConfig.IdentityJudge`
(`--judge` in the CLI, `--judge-model` / `--judge-cache` for the pinned model and
the decision file): pinned model, the measured 0.8 gates, and a required
decision cache — every decision is recorded and reused, so the output stays
deterministic.

Wiring also sharpened the diagnosis of the old pick. The similarity score is
`max(name similarity, address similarity)`, so a perfect address match to one
twin beats any name evidence for the other; where the twins share an identical
catalog address (`2831055212`, `2831025123`) both score 1.0, the name tie-break
decides, and the OCR name is never consulted:
`ChoosePhoneReference("Καλαφατάκη Ελένη - Γεωργία", <shared address>, refs)`
returns Βαρούχα. `internal/pipeline/identity_test.go` pins both the override and
the fallback behavior, including the case where the un-picked validator flags a
name mismatch against the judged entry.

## 5. Guardrails: what stays in code

- **Candidate retrieval, normalisation, canonicalisation, validation, JSON
  assembly, caching** — always code.
- Judgments only **classify lines, select among code-supplied candidates, verify
  matches, and score plausibility**.
- The model never generates a name, phone number or address, never invents a
  candidate, and never bypasses `validate`.
- Low confidence produces a **warning or fallback**, never a silent overwrite.
  Implemented: `internal/pipeline/identity.go` falls back to the similarity
  pick and records a warning for every non-accepted verdict.
- Answers to unused speculative questions are ignored; thresholds are evaluated
  on the corpus, not copied from cookbook examples.
- A pinned model version and a decision cache keyed by input hash are
  prerequisites for any runtime use. Implemented: `adjudicate.DefaultJudgeModel`
  and `DecisionCache` (`internal/adjudicate/cache.go`), required by
  `ClientConfig.IdentityJudge`.

## 6. Constraints

- **Determinism.** The project promises identical JSON for identical input.
  Options, in increasing risk: (a) build-time adjudication with checked-in
  results (C, B build-time mode); (b) runtime calls behind an opt-in flag with
  the current deterministic path as fallback; (c) cached runtime decisions with
  golden tests pinning the outputs. Never blend an uncached model decision
  silently into the output. The identity adjudicator takes (b) with a required
  cache (c): `--judge` is opt-in and every decision is persisted before it can
  enter the output.
- **Offline / no cloud.** Runtime judgment sends OCR text (public pharmacy data)
  to a third party. That is a data-policy decision, not just a technical one —
  make it opt-in (environment/flag), default off, and document it next to the
  "local, CPU-only" promise. Implemented as `--judge` / `ClientConfig.IdentityJudge`
  and documented in README/INTEGRATION; the default path is unchanged.
- **Trust.** Judgments are probabilities, not truth. Every use must define what
  happens below threshold; the existing warnings/discrepancy fields are the
  natural channel.
- **Cost and latency.** System One decisions are small and fast, questions in one
  request run in parallel, and the server OCRs at most once a day — so call
  volume is negligible. Adding questions still costs tokens; measure rather than
  assume.
- **Integration.** No official Go SDK: `POST /v1/systemone` with a Bearer key
  kept server-side. A small internal package (`internal/adjudicate`) with an
  interface makes the judgments fakeable in tests and keeps the parser pure.

## 7. Recommended path

1. **C — merge-golden entity alignment.** Implemented; adjudication is the
   default and `-judge-strict` refuses to write while curator decisions or
   fallbacks are unresolved. Run it with a report, resolve the curator queue,
   re-run until strict passes, and check in the resulting catalog — the runtime
   artifact stays deterministic. Measured defaults: `-judge-min-confidence 0.5`,
   one request per phone group.
2. **A — catalog identity adjudication**, opt-in, deterministic fallback intact,
   with a test covering the four shared-phone pairs above. **Done:**
   `internal/adjudicate/identity.go` and §4.1; wired through
   `pipeline.FillFromCatalogJudged`, `ValidateResultWithPicks` and
   `ClientConfig.IdentityJudge` (`--judge`), with the measured 0.8 gates and a
   required decision cache; the similarity pick stays the fallback.
3. **B / D / E** only after measuring on the corpus, keeping the default path
   untouched.

Every phase must leave the existing golden tests green, add no required network
call, and change no JSON schema (warnings are additive).

## 8. Caveats

- The fragile-surface review in §2 is static, and the §4 experiment ran one
  model version over a 14-case synthesized dataset, not the full corpus. The
  TypeSafe documentation's own thresholds and cookbook results are examples, not
  universal rules — validate them on this corpus.
- The §4.1 readings are simulated from catalog names, not live OCR, and its
  acceptance band is thin (0.05–0.06 between the highest wrong and lowest
  correct signal). Re-measure on live schedule readings before trusting the 0.8
  gate, and re-measure whenever the model version changes.
- Model behaviour changes across versions; the pinned OCR models and reference
  catalog do not. That asymmetry is the reason build-time use (C) is the safest
  entry point.
- Answers vary run to run on borderline pairs: the merge experiment flipped 6 of
  140 pair verdicts between two identical runs, all at confidence ≤ 0.40. Any
  runtime use must cache or check in decisions; never act on an unrecorded live
  answer.
- The provider is a third party. Even where the data is public, sending it there
  is a decision for the project owner, not a side effect of this document.

## 9. Sources

- TypeSafe documentation index — <https://docs.typesafe.ai/llms.txt>
- Primitives: [Choice](https://docs.typesafe.ai/primitives/choice),
  [Noul](https://docs.typesafe.ai/primitives/noul),
  [Score](https://docs.typesafe.ai/primitives/score)
- [HTTP API](https://docs.typesafe.ai/api) and
  [confidence](https://docs.typesafe.ai/confidence)
- Cookbooks: [entity alignment](https://docs.typesafe.ai/cookbooks/entity_alignment),
  [re-ranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe),
  [structure recovery](https://docs.typesafe.ai/cookbooks/autoformat),
  [pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook)
- [VISION_EVALUATION.md](VISION_EVALUATION.md) — the prior decision this builds on
