# INTEGRATION.md — Using rethymno-emergency-pharmacy from an external application

This document is for developers who want to feed the Rethymno pharmacy duty
schedule into their own application. It covers the four integration surfaces
in order of increasing decoupling:

| Surface | When to use it | Effort |
|---|---|---|
| Go library (`rethymnoemergency`) | your app is Go, wants the result synchronously | low |
| CLI (`rethymno-emergency-pharmacy`) | scripts, cron jobs, CI, one-off JSON | lowest |
| Docker image | anything that runs in containers | low |
| HTTP API (`serve`) | non-Go apps, service-to-service, caching | low (implemented) |

---

## 0. Constraints you must know first

These apply to every surface; they are the result of hard-won debugging
(details in [DEVELOPMENT_BLOG.md](DEVELOPMENT_BLOG.md)).

1. **OCR runs in a worker subprocess.** Loading the ONNX Runtime library into
   a process that also uses OpenCV (gocv) corrupts OpenCV allocations. The
   library therefore spawns `self ocr-worker` and talks to it over
   stdin/stdout. Your binary **must route the `ocr-worker` subcommand** to
   `rethymnoemergency.WorkerMain` (see §1.2), or set `ClientConfig.Backend =
   BackendInProcess` (unsafe on some glibc versions).
2. **Module consumers build with `-tags noembed_assets`.** The ONNX models
   and the ONNX Runtime library are *not* part of the Go module
   (`models/` and `third_party/` are gitignored; they exist only in full
   checkouts). A plain `go build` of the fetched module fails with
   `//go:embed: pattern ... no matching files found`. Consumers must:
   - build with `-tags noembed_assets`,
   - point `ClientConfig.ModelPath` at a models directory (downloaded with
     [`scripts/bootstrap.sh`](scripts/bootstrap.sh), pinned + SHA-256
     verified),
   - make the ONNX Runtime library loadable via `PHARMA_OCR_ORT_LIB` or the
     loader search path.
3. **You need OpenCV 4.6 system libraries** (gocv v0.31.0): `libopencv-core`,
   `imgproc`, `imgcodecs`, `highgui`, `videoio`, `calib3d`, `features2d`,
   `flann`, `objdetect`, `photo`, `video`, `dnn`. On Debian/Ubuntu:
   `apt-get install libopencv-dev` (the [Dockerfile](Dockerfile) lists the
   exact runtime packages).
4. **glibc ≥ 2.39 crashes.** The Go runtime + gocv + OpenCV allocation mix
   segfaults on modern host glibc. The same binary is fine on Debian bookworm
   (glibc 2.36). **Build and run in the Docker image unless you verified your
   host.** This is the single biggest reason to prefer the CLI/Docker surfaces
   over embedding the library on arbitrary hosts.
5. **JSON output is deterministic** for equal inputs (struct field order,
   sorted map keys) and carries `image_sha256` — you can cache, diff, and
   ETag it.

---

## 1. Go library

### 1.1 Install

```sh
go get github.com/mheers/rethymno-emergency-pharmacy
```

Build with the no-embed tag and provide the models (one-time):

```sh
git clone https://github.com/mheers/rethymno-emergency-pharmacy && cd rethymno-emergency-pharmacy
./scripts/bootstrap.sh                     # downloads models + ONNX Runtime, pinned
export PHARMA_OCR_ORT_LIB=$PWD/third_party/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so
go build -tags noembed_assets ./cmd/rethymno-emergency-pharmacy
```

### 1.2 Minimal consumer (subprocess backend — the supported default)

```go
package main

import (
	"context"
	"log"
	"os"

	rethymnoemergency "github.com/mheers/rethymno-emergency-pharmacy"
)

func main() {
	// Required: the OCR backend re-executes this binary as an "ocr-worker".
	// Route the subcommand before any application code runs.
	if len(os.Args) > 1 && os.Args[1] == "ocr-worker" {
		os.Exit(rethymnoemergency.WorkerMain(os.Args[2:]))
	}

	client, err := rethymnoemergency.New(rethymnoemergency.ClientConfig{})
	if err != nil {
		log.Fatal(err)
	}
	defer client.Close()

	ctx := context.Background()
	res, err := client.ParseSource(ctx, "schedule.jpg", rethymnoemergency.Options{City: "Ρέθυμνο"})
	if err != nil {
		log.Fatal(err)
	}
	out, _ := res.JSON()
	os.Stdout.Write(out)
}
```

### 1.3 ClientConfig

| Field | Default | Purpose |
|---|---|---|
| `ModelPath` | embedded models | directory containing `PP-OCRv6_*_onnx/inference.onnx`; **required when built with `-tags noembed_assets`** |
| `RecModel` | `"small"` | embedded recognition model: `small`, `medium`, `tiny` (ignored when `ModelPath` set) |
| `Backend` | `BackendSubprocess` | `BackendSubprocess` (worker subprocess) or `BackendInProcess` |
| `WorkerCommand` | `[os.Executable(), "ocr-worker"]` | override the worker binary |
| `NumThreads` | engine default | ONNX inference threads |
| `HTTPTimeout` | 30s | bounds `ParseURL` downloads |
| `Log` | stderr logger | pipeline diagnostics |

### 1.4 Client methods

| Method | Input | Output |
|---|---|---|
| `Parse(ctx, []byte, Options)` | already-fetched image bytes | `*Result` |
| `ParseFile(ctx, path, Options)` | local file | `*Result` |
| `ParseURL(ctx, url, Options)` | `http(s)` URL | `*Result` |
| `ParseSource(ctx, source, Options)` | URL **or** path (auto-detected) | `*Result` |
| `ParseJSON(ctx, source, Options)` | URL or path | marshaled `[]byte` |
| `Close()` | — | shuts down the worker |

`Options` fields: `SourceURL` (recorded in JSON), `City` (default
`Ρέθυμνο`), `DebugDir` (writes preprocessing + annotated OCR images — useful
for your own regression tests).

**Concurrency:** a `Client` is not meant for concurrent use — vision runs
in-process and the worker serializes requests. For a service that parses
many images concurrently, either shard across `Client` instances or accept
serialized throughput (a parse takes on the order of seconds).

### 1.5 Result JSON schema

```jsonc
{
  "schedule": {
    "source_url": "https://fskriti.gr/...",      // file path or URL, whatever you passed
    "city": "Ρέθυμνο",
    "days": [
      {
        "date": "10/08/2026",                     // DD/MM/YYYY — day name derived from it
        "day": "ΔΕΥΤΕΡΑ",                         // Zeller's congruence, not OCR
        "shifts": [
          {
            "from": "08:00", "to": "21:00",
            "pharmacies": [
              {
                "name": "ΠΑΠΑΤΖΑΝΗ ΜΑΡΙΑ",        // corrected against catalog when matched
                "address": "ΓΕΡΑΚΑΡΗ 96",
                "phone": "2831023347",
                "name_latin": "Papatzani Maria",  // filled from catalog, if matched
                "address_latin": "GERAKARI 96",
                "lat": 35.364, "lon": 24.475,     // catalog coordinates, if matched
                "confidence": 0.9,                // validation score, 0..1
                "warnings": [],
                "google": {                       // optional Places API enrichment
                  "place_id": "ChIJ...",
                  "formatted_address": "Gerakari 96, Rethymno 741 31, Greece",
                  "phone_international": "+30 2831 023347",
                  "website": "https://...",
                  "google_maps_url": "https://maps.google.com/?cid=...",
                  "rating": 4.5,
                  "user_rating_count": 12,
                  "business_status": "OPERATIONAL",
                  "types": ["pharmacy", "store"],
                  "opening_hours": {
                    "weekday_descriptions": ["Monday: 8:30 AM – 3:00 PM"],
                    "periods": [
                      { "open": {"day": 1, "hour": 8, "minute": 30},
                        "close": {"day": 1, "hour": 15, "minute": 0} }
                    ],
                    "open_now": false
                  },
                  "current_opening_hours": { ... },
                  "photos": [
                    { "content_type": "image/jpeg", "base64": "..." }
                  ]
                }
              }
            ]
          },
          { "from": "21:00", "to": "08:00", "pharmacies": [ /* ... */ ] }
        ]
      }
    ]
  },
  "raw_ocr": "y=106 x=695 c=0.878 ...",           // diagnostic dump, omitempty
  "image_sha256": "0eb319b1...",                   // deterministic key for caching
  "timings": { "decode": 123, "ocr": 4560, "total": 5120, /* ms */ },
  "validations": [
    { "phone": "2831023347", "name": "ΠΑΠΑΤΖΑΝΗ ΜΑΡΙΑ",
      "validation": { "phone_valid": true, "time_valid": true, "name_greek": true,
                      "address_ok": true, "catalog_match": {...}, "score": 0.9 } }
  ],
  "warnings": ["unknown phone 2831099999 (no catalog match)"]
}
```

`google` is a single optional block per pharmacy, copied verbatim from the
reference catalog entry when the OCR phone matched a catalog pharmacy that
carries Places API enrichment. Fields inside it use the raw provider values:
`day` in `periods` uses Google numbering (0 = Sunday), `weekday_descriptions`
are the provider's localized strings, and `photos[].base64` is a small
thumbnail. The block is absent for unmatched pharmacies and for catalog
entries that have no Google listing; `lat`/`lon` on the pharmacy itself are
the catalog coordinates (Google-corrected when enrichment exists).

Consumers should treat `warnings` as actionable: a schedule with unresolved
entries still parses, but the listed pharmacies are not catalog-verified.

---

## 2. CLI

The CLI is the library with a thin `flag` front end. It is the right surface
for cron jobs, CI, and anything that can live with a process per run.

```sh
# Parse a local image (the image files are the only input the parser needs)
rethymno-emergency-pharmacy parse ./schedule.jpg > schedule.json

# Parse the currently valid week straight from fskriti.gr
rethymno-emergency-pharmacy ingest --out /var/lib/schedules/current.json

# Diagnostics for your own images (geometry, columns, raw OCR lines)
rethymno-emergency-pharmacy inspect ./schedule.jpg

# Capacity planning on your host
rethymno-emergency-pharmacy benchmark ./testdata/schedules --iterations 3
```

**Scripting contract** (stable by design):

- exit code `0` = parsed, `1` = runtime error, `2` = usage error;
- **stdout carries only the JSON result**; logs and diagnostics go to stderr —
  safe to pipe into `jq`, files, or a queue;
- flags: `--models <dir>` (skip if embedded), `--debug <dir>`,
  `--source <url>`, `--city <name>`, `--iterations <n>`.

Weekly job example:

```sh
#!/bin/sh
# crontab:  15 8 * * 1  /opt/rethymno-emergency-pharmacy/run-weekly.sh
exec /usr/local/bin/rethymno-emergency-pharmacy ingest \
  --out /var/lib/schedules/$(date +%Y-%m-%d).json
```

Note: `ingest` is the only subcommand that touches the network (fskriti.gr);
`parse` is fully offline.

---

## 3. Docker

The [Dockerfile](Dockerfile) produces a runtime image with OpenCV libraries,
the ONNX Runtime library, and **embedded models** — the container needs no
mounted model files.

```sh
docker build -t rethymno-emergency-pharmacy:dev .

# Parse a schedule placed in your directory
docker run --rm -v "$PWD/schedules:/data" \
  rethymno-emergency-pharmacy:dev parse /data/10.08.2026-17.08.2026.jpg

# Weekly ingest, writing JSON next to your app
docker run --rm \
  -v /srv/app/schedules:/data \
  rethymno-emergency-pharmacy:dev ingest --out /data/current.json
```

Volume mounts: bind-mount **input** (schedule images) and **output**
(JSON/debug dirs) only. You do **not** need the Docker socket, and mounting
`/var/run/docker.sock` into the app container would be a security anti-pattern
— the image runs a plain CLI with no container-management functionality.

> The image runs as a **non-root user (UID 10001)**. A bind-mounted host
> directory must be readable by it — and writable for outputs (`ingest
> --out`, `serve`, debug dirs). Either `chown 10001:10001 <dir>` on the host
> or run with `--user "$(id -u):$(id -g)"`.

`docker-compose.yml` for a nightly job (a ready-made `compose.yaml` for the
`serve` + one-shot `ingest` pair is in the repo root):

```yaml
services:
  rethymno-emergency-pharmacy:
    image: ghcr.io/mheers/rethymno-emergency-pharmacy:latest
    volumes:
      - ./schedules:/data
    entrypoint: ["rethymno-emergency-pharmacy", "ingest", "--out", "/data/current.json"]
    restart: "no"
```

> Building a variant **without** embedded models (smaller image, models
> mounted): build the binary with `-tags noembed_assets` and mount
> `-v "$PWD/models:/models"`, then pass `--models /models`.

### 3.1 Use it as a base image

The image is a deliberately good base for anything that consumes the library:
it already ships `libopencv-*4.6`, `libonnxruntime.so`, CA certificates, and
the CLI — all that a Go app linked against the library needs at runtime. It
runs as UID 10001 and carries OCI labels (source, license, revision).

The build still needs the toolchain plus headers (`libopencv-dev`,
`libonnxruntime.so` for cgo), so build your app in a `golang:1.25-bookworm`
stage and only *land* it on this image:

```dockerfile
FROM golang:1.25-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev
ADD https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz /tmp/ort.tgz
RUN mkdir -p /opt/onnxruntime && tar xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && rm /tmp/ort.tgz
ENV CGO_ENABLED=1 CGO_CXXFLAGS="-I/opt/onnxruntime/include" CGO_LDFLAGS="-L/opt/onnxruntime/lib -lonnxruntime"
COPY . /src
WORKDIR /src
RUN go build -tags noembed_assets -o /out/my-app .

FROM ghcr.io/mheers/rethymno-emergency-pharmacy:latest
COPY --from=build /out/my-app /usr/local/bin/my-app
# non-root user, OpenCV/ORT libs, certs and PHARMA_OCR_ORT_LIB are inherited
ENTRYPOINT ["my-app"]
```

The app then needs models at runtime — mount them and point
`ClientConfig{ModelPath: "/models"}` at them (or embed them yourself in your
own checkout of the module). Simpler still: don't link the library at all and
shell out to the inherited CLI binary as a subprocess.

---

## 4. HTTP API (`serve`)

*Implemented.* An internal JSON endpoint with an in-memory TTL cache, a
daily midnight refresh, stale-while-revalidate, and single-flight fetches —
so fskriti.gr is hit at most once per day and consumers never block on a
slow or failing upstream.

### 4.1 Is it worth it?

Three concrete wins over calling the CLI/library per request:

1. The subprocess backend means each parse costs a process spawn + ONNX
   Runtime load (hundreds of ms overhead). A long-lived server amortizes
   that, making per-request latency and CPU predictable.
2. fskriti.gr is a small municipal WordPress site — the server fetches the
   schedule image **once a day, at local midnight** (plus at startup), and
   is a good citizen.
3. One process becomes the single source of truth, so N consumers stop
   maintaining their own model checkouts.

**It is NOT a public endpoint:** no auth, no TLS, no rate limiting. Bind
loopback / an internal network only.

### 4.2 Behavior

```sh
rethymno-emergency-pharmacy serve --listen 127.0.0.1:8080 --cache-ttl 24h
```

| Route | Behavior |
|---|---|
| `GET /healthz` | `200 {"ok":true}` — no pipeline work; for orchestrator probes |
| `GET /schedule/current` | the week covering today: fetch fskriti.gr → parse → validate |
| `GET /schedule/week?date=DD/MM/YYYY` | the schedule week containing `date` (ISO `YYYY-MM-DD` also accepted) |
| `GET /schedule/image?sha256=...` | replay a previously served result by `image_sha256` (last 20 cached) |
| `POST /parse` | multipart field `image` → on-demand parse, uncached |

**Cache & freshness — how updates are never lost:**

- **Daily fetch, unconditional.** On startup and then once per day at local
  midnight the server fetches the current week's image and replaces the
  cached result — **even when the cache looks fresh**. The schedule can
  change mid-week (corrections, substitutions), so the cache is never
  trusted to be current without a fresh fetch; the served data is therefore
  never older than one daily fetch.
- **Failure backoff.** A failed refresh retries hourly until it succeeds;
  the last known-good result keeps being served in the meantime.
- **TTL & stale-while-revalidate.** Entries are cached for `--cache-ttl`
  (default `24h`), keyed by ISO week (Monday-based, matching the schedule's
  week boundaries). A stale entry is still served immediately while a
  background goroutine revalidates it — a slow or down upstream never
  blocks a client that already has data.
- **Coverage check.** An entry is served for a requested date only when the
  parsed schedule actually covers that date. If the new week's image is not
  published yet, the newest result (previous week) is served as the best
  available answer and is replaced by the next daily fetch; a *stale* entry
  that provably does not cover the requested date is fetched synchronously,
  so a caller never receives a known-wrong week.
- **Single-flight.** Concurrent requests for the same week share one fetch,
  and fresh entries are never refetched per request — the upstream is hit
  at most once per day.

**Response contract** (identical JSON to §1.5):

- `ETag: "<image_sha256>"` — send it back in `If-None-Match` for a `304`
  when nothing changed;
- `Cache-Control: max-age=<ttl>` so a reverse proxy can layer on more cache;
- `X-Served-From: cache` marks responses served from cache (live fetches
  omit it) — handy when verifying the cache behavior;
- error responses are `{"error":"..."}` with `400` (bad request),
  `404` (unknown sha256), `502` (upstream fetch failed), `422` (image did
  not parse).

### 4.3 Client examples

```sh
curl -fsS localhost:8080/schedule/current | jq '.days[0]'
curl -fsS -H 'If-None-Match: "0eb319b1..."' localhost:8080/schedule/current
curl -fsS -F image=@schedule.jpg localhost:8080/parse | jq '.schedule.city'
```

Keep the cache warm and the fetch cadence independent of traffic (optional —
the midnight refresh already guarantees freshness):

```sh
# 15 8 * * 1  curl -fsS localhost:8080/schedule/current -o /dev/null
```

### 4.4 What it deliberately does not do

- no persistence (restart = cold cache, one upstream fetch at startup),
- no auth / multi-tenancy,
- no other municipalities or image formats,
- no push notifications — consumers poll.

---

## 5. Choosing a surface

| Your situation | Use |
|---|---|
| Go service, few parse calls, wants full control (`Options`, debug dirs) | library |
| Bash / Python / cron / CI; one schedule per run | CLI |
| Kubernetes, Nomad, or "I don't want OpenCV on my machine" | Docker image |
| Several non-Go services sharing one upstream | HTTP API (`serve`) |
| You only need *today's* pharmacies right now | any surface + a 5-line query on `days[].shifts[].pharmacies[]` with `date`/`day` filter — no reason to build anything |

## 6. Versioning & stability

The module has no tagged releases yet; the public API (`rethymnoemergency.New`,
`Client`, `WorkerMain`) is stable in shape but treat it as pre-1.0.
Pin what you build against:

```sh
go get github.com/mheers/rethymno-emergency-pharmacy@<commit-or-tag>
```

Stability guarantees you can rely on already: `Result.JSON()` byte-for-byte
determinism, the stdout/stderr/exit-code CLI contract, and the JSON field
names in §1.5.
