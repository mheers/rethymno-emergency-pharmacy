#!/usr/bin/env bash
# Bootstrap the third-party tooling required by rethymno-emergency-pharmacy:
#   - ONNX Runtime C library        -> third_party/onnxruntime-linux-x64-<ver>/
#   - PP-OCRv6 ONNX models          -> models/PP-OCRv6_*_onnx/
#   - char dict copy                -> models/char_dict.json (from the embedded source)
#
# All downloads are pinned by SHA-256 and verified before extraction.
# Idempotent: existing artifacts are reused; pass --force to re-download.
set -euo pipefail

ORT_VERSION=1.23.2
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_SHA256=1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b
ORT_DIR="onnxruntime-linux-x64-${ORT_VERSION}"

MODEL_BASE="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/tmp"
declare -A MODEL_SHA256=(
  [PP-OCRv6_medium_det_onnx]=46849788192d7882e52440d81cdfa437156f433368411c04b832ffa5433e5418
  [PP-OCRv6_medium_rec_onnx]=c3de894d4212b5d837375faeb05854ed4cbb9f29d579402795095dac843043be
  [PP-OCRv6_small_rec_onnx]=fd823f6dd80b1d2dbb32f9bf4e03582e9d32ed5639eb2a4dfda4496e265225dd
  [PP-OCRv6_tiny_rec_0515_onnx]=71a9a34ee45cb376cfdf8a849eb0cc66755e59baaccb6b94045d9ba6e5b6d012
)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
MODELS="$ROOT/models"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

fetch() { # url dest expected-sha256
  local url="$1" dest="$2" want="$3" got
  if [[ -f "$dest" ]] && [[ "$FORCE" -eq 0 ]]; then
    got="$(sha256sum "$dest" | awk '{print $1}')"
    if [[ "$got" == "$want" ]]; then
      echo "  reusing $dest"
      return 0
    fi
    echo "  checksum mismatch on $dest, re-downloading"
  fi
  echo "  downloading $url"
  curl -fsSL --retry 3 -o "$dest" "$url"
  echo "$want  $dest" | sha256sum -c - >/dev/null
}

extract() { # archive dest-dir
  local archive="$1" dir="$2"
  if [[ -d "$dir" ]] && [[ "$FORCE" -eq 0 ]]; then
    echo "  reusing $dir"
    return 0
  fi
  tar xf "$archive" -C "$(dirname "$dir")"
}

echo "== ONNX Runtime ${ORT_VERSION} (CPU) =="
mkdir -p "$THIRD_PARTY"
fetch "$ORT_URL" "$THIRD_PARTY/$ORT_DIR.tgz" "$ORT_SHA256"
extract "$THIRD_PARTY/$ORT_DIR.tgz" "$THIRD_PARTY/$ORT_DIR"

echo "== PP-OCRv6 ONNX models =="
mkdir -p "$MODELS"
for name in "${!MODEL_SHA256[@]}"; do
  echo "  $name"
  fetch "$MODEL_BASE/$name.tar" "$MODELS/$name.tar" "${MODEL_SHA256[$name]}"
  extract "$MODELS/$name.tar" "$MODELS/$name"
done
cp "$ROOT/internal/ocr/char_dict.json" "$MODELS/char_dict.json"

echo
echo "Done. Artifacts:"
du -sh "$THIRD_PARTY/$ORT_DIR" "$MODELS"
echo
echo "Outside Docker, point the runtime at the ORT library:"
echo "  export PHARMA_OCR_ORT_LIB=$THIRD_PARTY/$ORT_DIR/lib/libonnxruntime.so"
