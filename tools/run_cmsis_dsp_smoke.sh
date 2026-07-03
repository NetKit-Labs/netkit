#!/usr/bin/env bash
# Build netkit with CMSIS-DSP and run forward smoke on small bundled models.
# Avoids the full 59-case regression (large backbones are slow on CI).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f third_party/CMSIS-DSP/Include/arm_math.h ]]; then
  echo "CMSIS-DSP not found — run make cmsis-init" >&2
  exit 1
fi

echo "Building libnetkit.a + netkit with NETKIT_CMSIS_DSP=1..."
make NETKIT_CMSIS_DSP=1 lib netkit

CNN_4X4_INPUT="$(python3 -c 'print(",".join(["0.1"] * 16))')"

echo "CMSIS-DSP forward smoke: test_mlp.nk"
./netkit run models/test_mlp.nk --input 1,2

echo "CMSIS-DSP forward smoke: cnn_4x4_single.nk"
./netkit run models/cnn_4x4_single.nk --input "${CNN_4X4_INPUT}"

echo "CMSIS-DSP smoke passed."
