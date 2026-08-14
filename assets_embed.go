//go:build !noembed_assets

package rethymnoemergency

import _ "embed"

// Embedded runtime assets: the PP-OCRv6 ONNX models and the ONNX Runtime
// shared library, bootstrap'd by scripts/bootstrap.sh. Compiling without
// these files (e.g. a module consumer without a full checkout) requires
// -tags noembed_assets, in which case ClientConfig.ModelPath must point at
// a models directory and PHARMA_OCR_ORT_LIB at the ONNX Runtime library.

//go:embed models/PP-OCRv6_medium_det_onnx/inference.onnx
var embeddedDetModel []byte

//go:embed models/PP-OCRv6_small_rec_onnx/inference.onnx
var embeddedRecModelSmall []byte

//go:embed models/PP-OCRv6_medium_rec_onnx/inference.onnx
var embeddedRecModelMedium []byte

//go:embed models/PP-OCRv6_tiny_rec_0515_onnx/inference.onnx
var embeddedRecModelTiny []byte

//go:embed third_party/onnxruntime-linux-x64-*/lib/libonnxruntime.so.1.23.2
var embeddedORTLib []byte
