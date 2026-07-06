#!/usr/bin/env bash
# Flash NUCLEO-F446RE firmware via onboard ST-Link (USB).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ELF="${1:-$ROOT/build/mnist_mlp_nucleo_f446re.elf}"

if [[ ! -f "$ELF" ]]; then
  echo "ELF not found: $ELF (run make first)" >&2
  exit 1
fi

if ! command -v openocd >/dev/null 2>&1; then
  echo "openocd not found — install with: brew install openocd" >&2
  exit 1
fi

echo "Flashing $ELF via ST-Link..."
openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
  -c "program $ELF verify reset exit"

echo "Done."
