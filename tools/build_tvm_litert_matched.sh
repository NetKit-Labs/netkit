#!/usr/bin/env bash
# Rebuild stock Apache TVM (Relay-era host peer) with the same compiler/linker
# flags used for TF Lite / LiteRT and netkit host benches.
#
# This builds *stock* TVM — LLVM codegen only. No XNNPACK BYOC.
#
# Mirrors benchmark/common/tflite_host_flags.mk:
#   - drivers: gcc / g++  (Darwin: Apple clang via /usr/bin/g++ — same as
#              LiteRT macOS wheels; Linux: system GNU g++)
#   - opt:     -O3 -DNDEBUG
#   - cxx:     -std=gnu++17 -fpermissive  (LiteRT wheel dialect)
#   - Darwin arm64 link: -ld_classic
#
# Do not point CC/CXX at Homebrew g++-N on Darwin — that is GNU GCC and
# would not match the LiteRT wheel.
#
# One libtvm serves both float32 and int8 host benches.
#
# Usage:
#   export TVM_HOME=/path/to/apache-tvm-v0.14   # optional
#   ./tools/build_tvm_litert_matched.sh
#   ./tools/build_tvm_litert_matched.sh /path/to/apache-tvm-v0.14
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TVM_HOME="${1:-${TVM_HOME:-$ROOT/../apache-tvm-v0.14}}"
TVM_HOME="$(cd "$TVM_HOME" && pwd)"
JOBS="${TVM_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
LLVM_CONFIG="${TVM_LLVM_CONFIG:-}"

if [[ ! -f "$TVM_HOME/CMakeLists.txt" ]]; then
  echo "TVM_HOME=$TVM_HOME does not look like an Apache TVM tree" >&2
  exit 1
fi
if [[ -f "$TVM_HOME/cmake/modules/contrib/XNNPACK.cmake" ]] || \
   [[ -f "$TVM_HOME/python/tvm/relay/op/contrib/xnnpack.py" ]]; then
  echo "ERROR: custom XNNPACK BYOC still present under $TVM_HOME" >&2
  echo "Remove cmake/modules/contrib/XNNPACK.cmake, python/.../xnnpack.py," >&2
  echo "src/relay/backend/contrib/xnnpack/, src/runtime/contrib/xnnpack/" >&2
  exit 1
fi

# Prefer Homebrew llvm@15 (TVM 0.14); fall back to llvm-config on PATH.
if [[ -z "$LLVM_CONFIG" ]]; then
  for cand in \
    /opt/homebrew/opt/llvm@15/bin/llvm-config \
    /usr/local/opt/llvm@15/bin/llvm-config \
    "$(command -v llvm-config-15 2>/dev/null || true)" \
    "$(command -v llvm-config 2>/dev/null || true)"
  do
    if [[ -n "$cand" && -x "$cand" ]]; then
      LLVM_CONFIG="$cand"
      break
    fi
  done
fi
if [[ -z "$LLVM_CONFIG" || ! -x "$LLVM_CONFIG" ]]; then
  echo "llvm-config not found — set TVM_LLVM_CONFIG=/path/to/llvm-config" >&2
  exit 1
fi

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

# Same drivers as tflite_host_flags.mk / TFLM host (gcc/g++).
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

CXX_EXTRA="-fpermissive -std=gnu++17"
CMAKE_C_FLAGS="-O3 -DNDEBUG"
CMAKE_CXX_FLAGS="-O3 -DNDEBUG ${CXX_EXTRA}"

LINK_FLAGS=""
if [[ "$UNAME_S" == "Darwin" && "$UNAME_M" == "arm64" ]]; then
  LINK_FLAGS="-ld_classic"
fi

BUILD_DIR="$TVM_HOME/build"
mkdir -p "$BUILD_DIR"

CONFIG_CMAKE="$BUILD_DIR/config.cmake"
if [[ ! -f "$CONFIG_CMAKE" ]]; then
  if [[ -f "$TVM_HOME/cmake/config.cmake" ]]; then
    cp "$TVM_HOME/cmake/config.cmake" "$CONFIG_CMAKE"
  fi
fi
# Ensure any leftover USE_XNNPACK is OFF (stock).
if [[ -f "$CONFIG_CMAKE" ]] && grep -q 'USE_XNNPACK' "$CONFIG_CMAKE"; then
  sed -i.bak 's/set(USE_XNNPACK[^)]*)/set(USE_XNNPACK OFF)/' "$CONFIG_CMAKE"
  rm -f "$CONFIG_CMAKE.bak"
fi

echo "Building stock TVM (LLVM only; no XNNPACK BYOC)"
echo "  TVM_HOME     = $TVM_HOME"
echo "  CC/CXX       = $CC / $CXX"
echo "  C/CXX flags  = $CMAKE_C_FLAGS / $CMAKE_CXX_FLAGS"
echo "  link flags   = ${LINK_FLAGS:-(none)}"
echo "  LLVM         = $LLVM_CONFIG"
echo "  JOBS         = $JOBS"

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo "Clearing CMake cache (keeping config.cmake) for a clean configure"
  find "$BUILD_DIR" -mindepth 1 -maxdepth 1 \
    ! -name 'config.cmake' \
    ! -name 'config.cmake.bak' \
    -exec rm -rf {} +
fi

CMAKE_ARGS=(
  -S "$TVM_HOME"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER="$CC"
  -DCMAKE_CXX_COMPILER="$CXX"
  -DCMAKE_C_FLAGS="$CMAKE_C_FLAGS"
  -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS"
  -DCMAKE_C_FLAGS_RELEASE="$CMAKE_C_FLAGS"
  -DCMAKE_CXX_FLAGS_RELEASE="$CMAKE_CXX_FLAGS"
  -DUSE_LLVM="$LLVM_CONFIG"
)

if [[ -n "$LINK_FLAGS" ]]; then
  CMAKE_ARGS+=(
    -DCMAKE_EXE_LINKER_FLAGS="$LINK_FLAGS"
    -DCMAKE_SHARED_LINKER_FLAGS="$LINK_FLAGS"
    -DCMAKE_MODULE_LINKER_FLAGS="$LINK_FLAGS"
  )
fi

cmake "${CMAKE_ARGS[@]}"

for key in CMAKE_BUILD_TYPE CMAKE_C_COMPILER CMAKE_CXX_COMPILER; do
  val="$(grep -E "^${key}:" "$BUILD_DIR/CMakeCache.txt" | head -1 | cut -d= -f2- || true)"
  echo "  cache $key = ${val:-<missing>}"
done
bt="$(grep -E '^CMAKE_BUILD_TYPE:' "$BUILD_DIR/CMakeCache.txt" | head -1 | cut -d= -f2- || true)"
cxxflags="$(grep -E '^CMAKE_CXX_FLAGS:' "$BUILD_DIR/CMakeCache.txt" | head -1 | cut -d= -f2- || true)"
cxxflags_rel="$(grep -E '^CMAKE_CXX_FLAGS_RELEASE:' "$BUILD_DIR/CMakeCache.txt" | head -1 | cut -d= -f2- || true)"
if [[ "$bt" != "Release" ]]; then
  echo "ERROR: CMAKE_BUILD_TYPE='$bt' (expected Release)" >&2
  exit 1
fi
if [[ "$cxxflags" != *"-fpermissive"* && "$cxxflags_rel" != *"-fpermissive"* ]]; then
  echo "ERROR: -fpermissive missing from CMAKE_CXX_FLAGS*=('$cxxflags' / '$cxxflags_rel')" >&2
  exit 1
fi
if [[ "$cxxflags" != *"-O3"* && "$cxxflags_rel" != *"-O3"* ]]; then
  echo "ERROR: -O3 missing from CMAKE_CXX_FLAGS*=('$cxxflags' / '$cxxflags_rel')" >&2
  exit 1
fi

cmake --build "$BUILD_DIR" --config Release -j"$JOBS"

echo
echo "Done (stock TVM / LLVM)."
echo "  export TVM_HOME=$TVM_HOME"
echo "  export PYTHONPATH=\$TVM_HOME/python"
echo "Host/CPU TVM peer benches are not maintained; use this build for MCU/microTVM only."
