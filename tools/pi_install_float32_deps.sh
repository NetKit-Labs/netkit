#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ -z "${NETKIT_PI_PASS:-}" && -f "$ROOT/.netkit_pi_pass" ]]; then
  NETKIT_PI_PASS="$(cat "$ROOT/.netkit_pi_pass")"
fi
PASS="${NETKIT_PI_PASS:?set NETKIT_PI_PASS or provide .netkit_pi_pass}"

echo "$PASS" | sudo -S apt-get update -qq
echo "$PASS" | sudo -S apt-get install -y -qq python3-venv python3-pip
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -U pillow numpy

ok=0
set +e
.venv/bin/pip install -U ai-edge-litert
pip1=$?
.venv/bin/python -c 'from ai_edge_litert.interpreter import Interpreter; print("litert ok")'
py1=$?
set -e
if [[ "$pip1" -eq 0 && "$py1" -eq 0 ]]; then
  ok=1
else
  echo "ai-edge-litert failed; trying tflite-runtime ..."
  set +e
  .venv/bin/pip install -U tflite-runtime
  pip2=$?
  .venv/bin/python -c 'from tflite_runtime.interpreter import Interpreter; print("tflite_runtime ok")'
  py2=$?
  set -e
  if [[ "$pip2" -eq 0 && "$py2" -eq 0 ]]; then
    ok=1
  else
    echo "falling back to tensorflow (heavy) ..."
    .venv/bin/pip install -U tensorflow
    .venv/bin/python -c 'from tensorflow.lite.python.interpreter import Interpreter; print("tensorflow lite ok")'
    ok=1
  fi
fi
echo "python deps ready (ok=$ok)"
