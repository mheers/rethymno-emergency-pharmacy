//go:build noembed_assets

package rethymnoemergency

// Fallback for builds without the embedded assets (see assets_embed.go).
// The zero-length models force ClientConfig.ModelPath to be set; the
// runtime picks up the ONNX Runtime library via PHARMA_OCR_ORT_LIB or the
// loader search path.

var (
	embeddedDetModel      []byte
	embeddedRecModelSmall []byte
	embeddedRecModelMedium []byte
	embeddedRecModelTiny  []byte
	embeddedORTLib        []byte
)
