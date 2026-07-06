#!/usr/bin/env bash
# Read USART2 benchmark output from the NUCLEO ST-Link virtual COM port.
set -euo pipefail

BAUD="${BAUD:-115200}"

pick_port() {
  local candidates=()
  if [[ "$(uname -s)" == "Darwin" ]]; then
    candidates=(/dev/cu.usbmodem* /dev/tty.usbmodem*)
  else
    candidates=(/dev/ttyACM* /dev/serial/by-id/*STLINK*)
  fi
  local path
  for path in "${candidates[@]}"; do
    if [[ -e "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

PORT="${PORT:-}"
if [[ -z "$PORT" ]]; then
  PORT="$(pick_port)" || true
fi

if [[ -z "$PORT" ]]; then
  echo "No ST-Link VCP port found. Set PORT=/dev/tty... manually." >&2
  exit 1
fi

echo "Monitoring $PORT at ${BAUD} baud (Ctrl+C to exit)"
echo "Tip: press the black RESET button on the NUCLEO to re-run the benchmark."
echo

if python3 -c "import serial" 2>/dev/null; then
  python3 - "$PORT" "$BAUD" <<'PY'
import sys
import serial

port, baud = sys.argv[1], int(sys.argv[2])
with serial.Serial(port, baud, timeout=0.2) as ser:
    while True:
        data = ser.read(4096)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.flush()
PY
elif command -v picocom >/dev/null 2>&1; then
  picocom -b "$BAUD" "$PORT"
else
  echo "Install pyserial (pip install pyserial) or picocom (brew install picocom)" >&2
  exit 1
fi
