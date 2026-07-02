#!/usr/bin/env bash
# Compile-only smoke for Cortex-M4 + CMSIS-NN (no hardware required).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "arm-none-eabi-gcc not found; install gcc-arm-none-eabi" >&2
  exit 1
fi

make cmsis-init

export CC=arm-none-eabi-gcc
export CXX=arm-none-eabi-g++
make clean NETKIT_TARGET=mcu NETKIT_ARCH=CM4 NETKIT_CMSIS_NN=1 lib
echo "CM4 cross-compile OK: libnetkit.a"
