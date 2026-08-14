package ocr

import (
	"os"
	"path/filepath"
)

// Embedded-model plumbing for self-contained builds: the public library
// package embeds the ONNX Runtime shared library and the PP-OCRv6 models
// into the binary and registers them here, so both the main process and
// the worker subprocess (which is the same executable) can run without
// any files on disk.

var (
	embeddedLibBytes []byte
	embeddedDetBytes []byte
	embeddedRecBytes []byte
)

// SetEmbeddedLib registers the ONNX Runtime shared library bytes. The
// library is extracted to a temporary file on first use when
// PHARMA_OCR_ORT_LIB is not set. Call before creating any Engine.
func SetEmbeddedLib(data []byte) {
	embeddedLibBytes = data
}

// SetEmbeddedModels registers the PP-OCRv6 detection and recognition
// model bytes so the engine can run without a models directory on disk.
func SetEmbeddedModels(det, rec []byte) {
	embeddedDetBytes = det
	embeddedRecBytes = rec
}

func embeddedLibPath() (string, error) {
	if len(embeddedLibBytes) == 0 {
		return "", nil
	}
	dir, err := os.MkdirTemp("", "rethymno-emergency-pharmacy-ort-*")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, "libonnxruntime.so")
	if err := os.WriteFile(path, embeddedLibBytes, 0o755); err != nil {
		return "", err
	}
	return path, nil
}
