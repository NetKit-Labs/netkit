#!/usr/bin/env bash
# Compile-only smoke for hand FVP benchmark firmware (reference + CMSIS, mlp + cnn).
# Does not run Arm FVP — use make bench-hand-fvp locally when an FVP is available.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

setup_arm_toolchain_path() {
  if command -v arm-none-eabi-gcc >/dev/null 2>&1; then
    if arm-none-eabi-g++ -print-file-name=cstddef 2>/dev/null | rg -q '/'; then
      return 0
    fi
  fi
  local candidate
  for candidate in \
    "${HOME}/arm-gnu-toolchain/bin" \
    ; do
    if [[ -x "${candidate}/arm-none-eabi-gcc" ]]; then
      export PATH="${candidate}:${PATH}"
      if arm-none-eabi-g++ -print-file-name=cstddef 2>/dev/null | rg -q '/'; then
        return 0
      fi
    fi
  done
  if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
    for candidate in "${GITHUB_WORKSPACE}"/arm-gnu-toolchain-*/bin; do
      if [[ -d "${candidate}" && -x "${candidate}/arm-none-eabi-gcc" ]]; then
        export PATH="${candidate}:${PATH}"
        if arm-none-eabi-g++ -print-file-name=cstddef 2>/dev/null | rg -q '/'; then
          return 0
        fi
      fi
    done
  fi
  return 1
}

setup_arm_toolchain_path || true

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "arm-none-eabi-gcc not found; install Arm GNU Toolchain (arm-none-eabi with C++)" >&2
  exit 1
fi

for model in models/mlp_hand.nk models/cnn_hand.nk; do
  if [[ ! -f "${model}" ]]; then
    echo "Missing ${model}" >&2
    exit 1
  fi
done

make cmsis-init
PYTHONPATH=python python3 tools/embed_hand_fvp_data.py

export CC=arm-none-eabi-gcc
export CXX=arm-none-eabi-g++
make -C benchmarks/fvp all-models

echo "Hand FVP firmware compile OK:"
echo "  benchmarks/fvp/hand_fvp_bench_ref_{mlp,cnn}.elf"
echo "  benchmarks/fvp/hand_fvp_bench_cmsis_{mlp,cnn}.elf"
