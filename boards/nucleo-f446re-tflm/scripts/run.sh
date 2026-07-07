#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make "$@"
../nucleo-f446re/scripts/flash.sh "$ROOT/build/mnist_mlp_tflm_nucleo_f446re.elf"
../nucleo-f446re/scripts/monitor.sh
