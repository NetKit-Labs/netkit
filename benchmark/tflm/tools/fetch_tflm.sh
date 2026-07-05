#!/usr/bin/env bash
# Clone TensorFlow Lite Micro into third_party/ if missing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/tflite-micro"

if [ -d "$DEST/.git" ]; then
  echo "TFLM already present at $DEST"
  exit 0
fi

mkdir -p "$ROOT/third_party"
echo "Cloning tensorflow/tflite-micro into $DEST ..."
git clone --depth 1 https://github.com/tensorflow/tflite-micro.git "$DEST"

echo "Installing netkit benchmark Makefile.inc hook ..."
cp "$ROOT/Makefile.inc" "$DEST/tensorflow/lite/micro/tools/make/additional_tests.inc"

echo "Done."
