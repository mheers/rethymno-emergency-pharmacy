DOCKER_IMAGE ?= rethymno-emergency-pharmacy:dev
DOCKER_REPO ?= mheers/rethymno-emergency-pharmacy
DOCKER_TAG ?= latest
VERSION ?= $(shell git describe --tags --always 2>/dev/null || echo dev)
DEMO_IMAGE ?= testdata/schedules/10.08.2026-17.08.2026_page-0001-2.jpg
MODELS_DIR ?= models

.PHONY: demo-json docker-build docker-push

# Parse one demo image and print the complete JSON result to stdout.
# Dockerfile.dev provides the Go/OpenCV/ONNX Runtime environment used by the
# CLI, so invoke Go explicitly instead of depending on the image entrypoint.
demo-json:
	@docker run --rm --cpus 2 --entrypoint go \
		-v "$(CURDIR):/data" \
		-w /data \
		"$(DOCKER_IMAGE)" \
		run ./cmd/rethymno-emergency-pharmacy parse --models "$(MODELS_DIR)" "$(DEMO_IMAGE)"

# Build the production image (multi-stage Dockerfile; models + ORT lib
# embedded in the binary, only OpenCV shared libs at runtime).
# Tags: $(DOCKER_REPO):$(DOCKER_TAG) plus $(DOCKER_REPO):$(VERSION).
docker-build:
	docker build --build-arg VCS_REF="$(VERSION)" \
		-t "$(DOCKER_REPO):$(DOCKER_TAG)" -t "$(DOCKER_REPO):$(VERSION)" .

# Push the production image to the registry (Docker Hub by default,
# override DOCKER_REPO for ghcr.io, e.g. ghcr.io/mheers/rethymno-emergency-pharmacy).
docker-push: docker-build
	docker push "$(DOCKER_REPO):$(DOCKER_TAG)"
	docker push "$(DOCKER_REPO):$(VERSION)"
