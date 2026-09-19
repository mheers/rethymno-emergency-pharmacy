# rethymno-emergency-pharmacy

A local, CPU-only OCR pipeline that turns the weekly **Rethymno pharmacy duty
schedule** — published as a JPEG image on [fskriti.gr](https://fskriti.gr/εφημερίες-φαρμακείων-ρεθύμνου/) —
into validated, deterministic JSON. Written in Go, no database, no cloud by
default: the CLI, the library behind it, and an optional internal HTTP API are
the whole surface. An opt-in TypeSafe System One adjudicator can resolve
ambiguous catalog matches — off unless asked for, and never able to write a
pharmacy the catalog does not contain (see `--judge` below).

```
image → OpenCV preprocessing → PP-OCRv6 (ONNX Runtime) → layout → parse → validate → JSON
```

## What it does

- Downloads the current weekly schedule from the FSKriti WordPress gallery, or
  parses any local schedule image (`rethymno-emergency-pharmacy parse ./schedule.jpg`).
- Preprocesses the image with OpenCV (via GoCV): deskew, grid-line removal,
  column detection.
- OCRs the table with **PP-OCRv6** on **ONNX Runtime** (CPU only).
- Reconstructs the 4-column × 2-day table layout, derives day names from dates
  (Zeller's congruence — day names are unreliable for OCR, dates are not).
- Parses each day's two duty shifts (`08:00–21:00` and `21:00–08:00`) into
  pharmacy entries: name, address, phone, per-entry OCR confidence.
- Validates the result against the municipality's official pharmacy catalog
  (73 entries, embedded in the binary) and emits warnings for unknown matches.
- **Enriches matched pharmacies** with the Google Places catalog block
  (corrected coordinates, contact details, opening hours, base64 photo
  thumbnails) when the reference catalog carries it.
- Outputs deterministic JSON: stable key order, embedded image SHA-256 and
  per-stage timing.

The models (PP-OCRv6 detection + recognition) and the ONNX Runtime C library
are **embedded into the binary** — a `go build` produces a self-contained
artifact that needs only the OpenCV shared libraries at runtime.

### Why not a vision LLM?

In September 2026 we evaluated reading the schedule image with a
vision-capable LLM (DeepSeek V4.1 Flash) instead of the OCR pipeline. The
model read the image well and the approach is much simpler, but it cannot
canonicalize names against the reference catalog, verify phone numbers, or
guarantee deterministic output — so the validated pipeline stays. The full
comparison is in [VISION_EVALUATION.md](VISION_EVALUATION.md).

## Requirements

- Go 1.25+ with CGO enabled
- OpenCV 4.6 (`libopencv-dev`; gocv v0.31.0)
- ONNX Runtime 1.23.2 (CPU build)

> **Known platform quirk:** on hosts with glibc ≥ 2.39 the combination of the
> Go runtime, GoCV and OpenCV allocations is known to crash (SIGSEGV). The
> same binary runs fine on Debian bookworm (glibc 2.36) — use the Docker image
> for development unless you are on a compatible host. See
> [DEVELOPMENT_BLOG.md](DEVELOPMENT_BLOG.md) for the full story.

## Quick start (Docker, recommended)

```sh
docker build -t rethymno-emergency-pharmacy:dev .
docker run --rm -v "$PWD:/data" -w /data rethymno-emergency-pharmacy:dev \
  go run ./cmd/rethymno-emergency-pharmacy parse testdata/schedules/10.08.2026-17.08.2026_page-0001-2.jpg
```

or use the Makefile helper:

```sh
make demo-json
```

## Build from source

Download the ONNX Runtime library and the PP-OCRv6 models (pinned and
SHA-256 verified), then build with the model directory on disk:

```sh
./scripts/bootstrap.sh
export PHARMA_OCR_ORT_LIB=$PWD/third_party/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so
go build -o rethymno-emergency-pharmacy ./cmd/rethymno-emergency-pharmacy
./rethymno-emergency-pharmacy parse --models models testdata/schedules/10.08.2026-17.08.2026_page-0001-2.jpg
```

Embedding happens automatically: `//go:embed` (see `assets_embed.go`) packs
the files from `models/` and `third_party/` into the binary, so a plain
`go build` after `bootstrap.sh` yields a binary that needs no model files at
runtime. Module consumers without the checked-in assets must build with
`-tags noembed_assets` and point `ClientConfig.ModelPath` at a models
directory.

## CLI

```
rethymno-emergency-pharmacy ingest                     download current FSKriti schedule, parse, print JSON
rethymno-emergency-pharmacy parse [flags] <image>      parse a local schedule image to JSON
rethymno-emergency-pharmacy inspect <image> [flags]    print preprocessing + OCR diagnostics
rethymno-emergency-pharmacy benchmark <dir> [flags]    run stage timings over images in <dir>
rethymno-emergency-pharmacy serve [flags]              internal HTTP API with a daily-refreshed cache
rethymno-emergency-pharmacy version                    print version

Flags:
  --models <dir>    ONNX model directory (default: embedded in the binary)
  --debug <dir>     write debugging artifacts under <dir>
  --source <url>    source URL recorded in the JSON
  --city <name>     city name recorded in the JSON (default: Ρέθυμνο)
  --iterations <n>  benchmark repetitions (default 1)
  --listen <addr>   serve: HTTP listen address (default 127.0.0.1:8080)
  --cache-ttl <d>   serve: cache TTL for parsed schedules (default 24h)
  --judge           ingest/parse/serve: adjudicate ambiguous catalog matches with
                    TypeSafe System One (sends OCR text to api.typesafe.ai;
                    requires TYPESAFE_API_KEY; decisions are cached)
  --judge-model     pinned System One model (default jev-1.13.0)
  --judge-cache     identity decision cache file (default: user cache dir)
```

### Optional: catalog-identity adjudication

When a phone number matches several catalog entries (four numbers in the
Rethymno catalog are two pharmacies on one line), the similarity matcher
scores `max(name, address)`: an exact address match to one twin overrides the
other twin's name evidence, and two of the four pairs share an identical
address, where the tie-break decides and the OCR name is never consulted — the
wrong pharmacy's name, coordinates and Google data can be filled silently.
With `--judge` (or
`ClientConfig.IdentityJudge`), TypeSafe System One reads the OCR name and
selects the one catalog entry it describes — or answers `none`. The measured
gate (TYPESAFE_EVALUATION.md §4.1) accepts a selection only at 0.8 confidence
and 0.8 same-pharmacy Noul; everything else keeps the deterministic pick and
adds a warning. Every decision is recorded in `--judge-cache` and reused, so
output stays deterministic; deleting the cache file (or one entry) forces
re-evaluation. In containers without `$HOME`, pass `--judge-cache` explicitly
(ideally into a mounted volume) so the decisions persist and stay reviewable.
This is the one code path that leaves the machine: it is
opt-in, default off, and sends only OCR text of public pharmacy data.

### HTTP API

`rethymno-emergency-pharmacy serve` exposes the schedule as an internal JSON API for
service-to-service use. It fetches the upstream schedule image **once a
day, at local midnight** (plus a startup warm-up) — the cache is never
trusted to be current across days, because the schedule can change
mid-week — and requests are served exclusively from the cache, so the OCR
pipeline runs at most once per day:

```sh
rethymno-emergency-pharmacy serve --listen 127.0.0.1:8080
curl -fsS localhost:8080/schedule/current | jq '.days'
curl -fsS "localhost:8080/schedule/week?date=15/08/2026" | jq '.days[0]'
```

Routes, cache semantics and the response contract are documented in
[INTEGRATION.md](INTEGRATION.md#4-http-api-serve).

### Example output

```sh
$ rethymno-emergency-pharmacy parse testdata/schedules/10.08.2026-17.08.2026_page-0001-2.jpg
```

```json
{
  "source_url": "",
  "city": "Ρέθυμνο",
  "days": [
    {
      "date": "10/08/2026",
      "day": "ΔΕΥΤΕΡΑ",
      "shifts": [
        {
          "from": "08:00",
          "to": "21:00",
          "pharmacies": [
            {
              "name": "ΠΑΠΑΤΖΑΝΗ ΜΑΡΙΑ",
              "address": "TEPAKAPH96 96GERAKARISTR",
              "phone": "2831023347",
              "confidence": 0.9
            }
          ]
        },
        {
          "from": "21:00",
          "to": "08:00",
          "pharmacies": [
            {
              "name": "ΔΑΦΝΟΜΗΛΗ ΓΕΩΡΓΙΑ",
              "address": "ΔHMOKPATIAΣ6",
              "phone": "2831056850",
              "confidence": 0.9
            }
          ]
        }
      ]
    }
  ]
}
```

## Use as a Go library

```go
package main

import (
	"context"
	"log"
	"os"

	rethymnoemergency "github.com/mheers/rethymno-emergency-pharmacy"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "ocr-worker" {
		os.Exit(rethymnoemergency.WorkerMain(os.Args[2:]))
	}
	client, err := rethymnoemergency.New(rethymnoemergency.ClientConfig{})
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()
	out, err := client.ParseJSON(context.Background(), "schedule.jpg", rethymnoemergency.Options{})
	if err != nil {
		log.Fatal(err)
	}
	os.Stdout.Write(out)
}
```

OCR inference runs in a **worker subprocess** by default: the ONNX Runtime
library corrupts OpenCV allocations when loaded into the same process. The
host binary must expose an `ocr-worker` subcommand (see above), or set
`ClientConfig.WorkerCommand` / `ClientConfig.Backend = BackendInProcess`.

## Project layout

```
cmd/rethymno-emergency-pharmacy/     CLI (ingest, parse, inspect, benchmark, serve)
cmd/merge-golden/                    one-off merge of the Google-enriched golden catalog into the reference
internal/adjudicate/  TypeSafe System One client, catalog-identity adjudicator, decision cache
internal/server/      internal HTTP API: TTL cache, midnight refresh
internal/fetch/       HTTP download of the FSKriti schedule page
internal/extract/     schedule-image detection and week selection
internal/vision/      OpenCV preprocessing (deskew, grid removal, columns)
internal/ocr/         PP-OCRv6 on ONNX Runtime, worker subprocess
internal/layout/      table reconstruction and day derivation
internal/parse/       domain entry parsing
internal/validate/    reference catalog lookup and validation
testdata/schedules/   weekly schedule images
testdata/reference/   pharmacy catalog and duty-page fixtures
pharmacyocr.go        public library API
scripts/bootstrap.sh  pinned, SHA-verified model/ORT downloads
```

## Documentation

- [INTEGRATION.md](INTEGRATION.md) — how an external application can use and
  import this tooling: Go library, CLI, Docker, and a proposed internal HTTP
  API with a 24 h cache.
- [DEVELOPMENT_BLOG.md](DEVELOPMENT_BLOG.md) — a full developer story: Greek
  OCR chaos, the GoCV/glibc crash, the ORT heap corruption, the NCHW/HWC
  tensor-order bug, and how each was solved.
- [VISION_EVALUATION.md](VISION_EVALUATION.md) — why the validated OCR
  pipeline stays: the September 2026 comparison against a vision LLM and the
  decision to keep the deterministic implementation.
- [TYPESAFE_EVALUATION.md](TYPESAFE_EVALUATION.md) — where System One judgments
  can replace fragile parsing heuristics (catalog identity adjudication, line
  classification, golden-catalog merge), the measured golden-merge adjudicator
  now used by `merge-golden`, the measured catalog-identity adjudicator wired
  into the runtime behind `--judge`, and the guardrails they run behind.
- `rethymno-emergency-pharmacy inspect <image>` — geometry, columns and raw OCR diagnostics
  for a single image.

## Data sources

- Weekly schedule image: [fskriti.gr](https://fskriti.gr/εφημερίες-φαρμακείων-ρεθύμνου/)
- Structured duty page (cross-check): [rethymno.gr](https://www.rethymno.gr/information-services/pharmacies/pharmacies.html)
- Pharmacy catalog (validation dictionary): [rethymno.gr](https://www.rethymno.gr/guide/pharmacies)
- Google Places enrichment: merged once into the reference catalog with
  `go run ./cmd/merge-golden -judge-strict -golden <path-to-catalog/pharmacies.json>`
  (the golden catalog is produced by the expat-map-guide enrichment workflow
  and is not part of this repository). Adjudication is on by default; strict
  mode refuses to write while curator decisions or fallbacks are unresolved.

## License

MIT — see [LICENSE](LICENSE).
