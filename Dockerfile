# Multi-stage build for rethymno-emergency-pharmacy.
#
# Version relationship (documented in README):
#   - gocv v0.31.0       <-> OpenCV 4.6.0  (Debian bookworm's libopencv-dev)
#   - onnxruntime 1.23.2 <-> yalue/onnxruntime_go v1.24.0 (ORT API version 22)
#
# The runtime image is a slim Debian with exactly the OpenCV shared
# libraries and the ONNX Runtime library. No Python, no CUDA, CPU only.

FROM golang:1.25-bookworm AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime C library (pinned, CPU-only build)
ARG ORT_VERSION=1.23.2
ADD https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz /tmp/ort.tgz
RUN mkdir -p /opt/onnxruntime && tar xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && rm /tmp/ort.tgz

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
ENV CGO_ENABLED=1 \
    CGO_CXXFLAGS="-I/opt/onnxruntime/include" \
    CGO_LDFLAGS="-L/opt/onnxruntime/lib -lonnxruntime" \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib
# Copy the ORT lib into the embed location (assets_embed.go) — third_party/
# is excluded from the build context, so the download above doubles as the
# embed source.
RUN mkdir -p third_party/onnxruntime-linux-x64-${ORT_VERSION}/lib \
 && cp /opt/onnxruntime/lib/libonnxruntime.so.1.23.2 \
        third_party/onnxruntime-linux-x64-${ORT_VERSION}/lib/ \
 && go build -trimpath -ldflags="-s -w" -o /out/rethymno-emergency-pharmacy ./cmd/rethymno-emergency-pharmacy

FROM debian:bookworm-slim AS runtime

ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="rethymno-emergency-pharmacy" \
      org.opencontainers.image.description="OCR pipeline for the Rethymno pharmacy duty schedules (OpenCV + PP-OCRv6 on ONNX Runtime, CPU only)" \
      org.opencontainers.image.source="https://github.com/mheers/rethymno-emergency-pharmacy" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.6 \
    libopencv-imgproc4.6 \
    libopencv-imgcodecs4.6 \
    libopencv-highgui4.6 \
    libopencv-videoio4.6 \
    libopencv-calib3d4.6 \
    libopencv-features2d4.6 \
    libopencv-flann4.6 \
    libopencv-objdetect4.6 \
    libopencv-photo4.6 \
    libopencv-video4.6 \
    libopencv-dnn4.6 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/onnxruntime/lib/libonnxruntime.so.1.23.2 /usr/local/lib/
RUN ln -s /usr/local/lib/libonnxruntime.so.1.23.2 /usr/local/lib/libonnxruntime.so.1 \
 && ln -s /usr/local/lib/libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so

COPY --from=build /out/rethymno-emergency-pharmacy /usr/local/bin/rethymno-emergency-pharmacy

# Non-root default user so the image can be reused as a base image.
RUN useradd --system --uid 10001 --create-home rethymno \
 && mkdir -p /data && chown rethymno: /data
USER rethymno

ENV PHARMA_OCR_ORT_LIB=/usr/local/lib/libonnxruntime.so \
    LD_LIBRARY_PATH=/usr/local/lib
WORKDIR /data

# Meaningful for `serve`; one-shot commands exit before the first probe.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

# Pass a subcommand explicitly, e.g.
# `docker run --rm -v "$PWD":/data rethymno-emergency-pharmacy:dev parse schedule.jpg`,
# or `... rethymno-emergency-pharmacy:dev serve --listen 0.0.0.0:8080` for the HTTP API.
ENTRYPOINT ["rethymno-emergency-pharmacy"]
