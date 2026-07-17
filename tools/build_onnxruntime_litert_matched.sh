#!/usr/bin/env bash
# Build ONNX Runtime (Python package) with XNNPACK EP using the same host
# compiler/linker policy as LiteRT-matched netkit benches
# (benchmark/common/tflite_host_flags.mk).
#
# LiteRT host peer is a prebuilt wheel; ORT desktop wheels do not ship the
# XNNPACK EP. This script builds ORT from source so XNNPACK ON/OFF is a real
# peer knob:
#   ON  — Session providers=["XNNPACKExecutionProvider","CPUExecutionProvider"]
#   OFF — providers=["CPUExecutionProvider"] only
#
# Flags (mirror tflite_host_flags.mk / LiteRT macOS wheels):
#   - drivers: gcc / g++  (Darwin: Apple clang via /usr/bin/g++ — not Homebrew g++-N)
#   - opt:     -O3 -DNDEBUG
#   - cxx:     -fpermissive  (LiteRT .bazelrc)
#   - Darwin arm64 link: -ld_classic
#
# Usage:
#   ./tools/build_onnxruntime_litert_matched.sh
#   ./tools/build_onnxruntime_litert_matched.sh /path/to/onnxruntime
#   ORT_HOME=... ORT_GIT_TAG=v1.20.1 ./tools/build_onnxruntime_litert_matched.sh
#
# Installs the wheel into benchmark/onnxruntime/.venv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_HOME="${1:-${ORT_HOME:-$ROOT/../onnxruntime}}"
ORT_GIT_URL="${ORT_GIT_URL:-https://github.com/microsoft/onnxruntime.git}"
ORT_GIT_TAG="${ORT_GIT_TAG:-v1.20.1}"
JOBS="${ORT_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)}"
VENV="${ORT_VENV:-$ROOT/benchmark/onnxruntime/.venv}"
PY_VER="${ORT_PYTHON:-python3}"

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

# Same drivers as tflite_host_flags.mk
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

# Prefer Xcode/system python over Anaconda (Anaconda re2/absl break ORT FetchContent).
if [[ -x /usr/bin/python3 ]]; then
  PY_VER="${ORT_PYTHON:-/usr/bin/python3}"
fi

CMAKE_C_FLAGS="-O3 -DNDEBUG"
CMAKE_CXX_FLAGS="-O3 -DNDEBUG -fpermissive"
EXTRA_LINK=""
if [[ "$UNAME_S" == "Darwin" && "$UNAME_M" == "arm64" ]]; then
  EXTRA_LINK="-ld_classic"
fi

# ORT vendors absl/re2/protobuf. Ignore Homebrew + Anaconda CMake packages.
export CMAKE_IGNORE_PREFIX_PATH="/opt/homebrew;/opt/anaconda3;/usr/local${CMAKE_IGNORE_PREFIX_PATH:+;$CMAKE_IGNORE_PREFIX_PATH}"
export CMAKE_IGNORE_PATH="/opt/homebrew;/opt/anaconda3;/usr/local${CMAKE_IGNORE_PATH:+;$CMAKE_IGNORE_PATH}"
# Do not inherit a polluted prefix path from the shell.
export CMAKE_PREFIX_PATH="${ORT_CMAKE_PREFIX_PATH:-}"

if [[ ! -d "$ORT_HOME/.git" ]]; then
  echo "Cloning ONNX Runtime $ORT_GIT_TAG -> $ORT_HOME"
  git clone --recursive --depth 1 --branch "$ORT_GIT_TAG" "$ORT_GIT_URL" "$ORT_HOME"
else
  echo "Using existing ORT tree at $ORT_HOME"
fi
ORT_HOME="$(cd "$ORT_HOME" && pwd)"

if [[ ! -f "$ORT_HOME/tools/ci_build/build.py" ]]; then
  echo "ERROR: $ORT_HOME does not look like an onnxruntime checkout" >&2
  exit 1
fi

# Third-party archive SHA1s in cmake/deps.txt often drift (GitLab zip
# re-packs). Patch known-bad entries to the hashes CMake measured locally.
DEPS_TXT="$ORT_HOME/cmake/deps.txt"
if [[ -f "$DEPS_TXT" ]]; then
  python3 - "$DEPS_TXT" <<'PY'
import sys
from pathlib import Path

# name -> measured SHA1 of the archive currently served at that URL
PATCHES = {
    # eigen e7248b26… zip (GitLab)
    "eigen": "32b145f525a8308d7ab1c09388b2e288312d8eba",
    # kleidiai v0.2.0 zip (Arm GitLab)
    "kleidiai": "61886eac853e2235359cbcc73c5830b8abf71467",
}
p = Path(sys.argv[1])
out = []
changed = []
for ln in p.read_text().splitlines(True):
    raw = ln.rstrip("\n")
    if raw and not raw.startswith("#") and ";" in raw:
        parts = raw.split(";")
        name = parts[0]
        if name in PATCHES and len(parts) >= 2:
            new = f"{name};{parts[1]};{PATCHES[name]}"
            if new != raw:
                changed.append(name)
            out.append(new + "\n")
            continue
    out.append(ln if ln.endswith("\n") else ln + "\n")
p.write_text("".join(out))
print(f"Patched deps.txt hashes for: {', '.join(changed) or '(none)'}")
PY
fi

echo "Building ONNX Runtime with XNNPACK EP (LiteRT-matched flags)"
echo "  ORT_HOME     = $ORT_HOME"
echo "  tag/pin      = $ORT_GIT_TAG"
echo "  CC/CXX       = $CC / $CXX"
echo "  C/CXX flags  = $CMAKE_C_FLAGS / $CMAKE_CXX_FLAGS"
echo "  link flags   = ${EXTRA_LINK:-(none)}"
echo "  JOBS         = $JOBS"
echo "  venv         = $VENV"

BUILD_DIR="$ORT_HOME/build/litert_matched"
mkdir -p "$BUILD_DIR"

CMAKE_DEFINES=(
  --cmake_extra_defines "CMAKE_C_COMPILER=$CC"
  --cmake_extra_defines "CMAKE_CXX_COMPILER=$CXX"
  --cmake_extra_defines "CMAKE_C_FLAGS=$CMAKE_C_FLAGS"
  --cmake_extra_defines "CMAKE_CXX_FLAGS=$CMAKE_CXX_FLAGS"
  --cmake_extra_defines "CMAKE_IGNORE_PREFIX_PATH=/opt/homebrew;/opt/anaconda3;/usr/local"
  --cmake_extra_defines "CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
  --cmake_extra_defines "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"
  # Homebrew CMake 4.x rejects ancient cmake_minimum_required in ORT deps (nsync).
  --cmake_extra_defines "CMAKE_POLICY_VERSION_MINIMUM=3.5"
)
if [[ -n "$EXTRA_LINK" ]]; then
  CMAKE_DEFINES+=(
    --cmake_extra_defines "CMAKE_EXE_LINKER_FLAGS=$EXTRA_LINK"
    --cmake_extra_defines "CMAKE_SHARED_LINKER_FLAGS=$EXTRA_LINK"
  )
fi

# Wipe prior failed configure that may have cached bad package paths.
rm -rf "$BUILD_DIR/Release/CMakeCache.txt" "$BUILD_DIR/Release/CMakeFiles"

# Release + XNNPACK EP + Python wheel. Skip tests for speed.
# --compile_no_warning_as_error: Apple clang / newer ORT often warn-as-error.
"$PY_VER" "$ORT_HOME/tools/ci_build/build.py" \
  --build_dir "$BUILD_DIR" \
  --config Release \
  --parallel "$JOBS" \
  --build_shared_lib \
  --build_wheel \
  --use_xnnpack \
  --skip_tests \
  --compile_no_warning_as_error \
  --skip_submodule_sync \
  --update \
  --build \
  "${CMAKE_DEFINES[@]}"

WHEEL="$(ls -1t "$BUILD_DIR"/Release/dist/onnxruntime-*.whl 2>/dev/null | head -1 || true)"
if [[ -z "$WHEEL" ]]; then
  # Some layouts put dist under build_dir/Release/Release/dist or similar.
  WHEEL="$(find "$BUILD_DIR" -type f -name 'onnxruntime-*.whl' | head -1 || true)"
fi
if [[ -z "$WHEEL" || ! -f "$WHEEL" ]]; then
  echo "ERROR: onnxruntime wheel not found under $BUILD_DIR" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY_VER" -m venv "$VENV"
fi
"$VENV/bin/pip" install -U pip wheel
"$VENV/bin/pip" install --force-reinstall "$WHEEL" numpy pillow onnx

echo
echo "Verifying providers..."
"$VENV/bin/python" - <<'PY'
import onnxruntime as ort
print("onnxruntime", ort.__version__)
providers = ort.get_available_providers()
print("providers:", providers)
# ORT registers as XnnpackExecutionProvider (mixed case).
if not any(p.lower() == "xnnpackexecutionprovider" for p in providers):
    raise SystemExit("XnnpackExecutionProvider missing from build")
print("OK")
PY

echo
echo "Done. Use:"
echo "  export ORT_HOME=$ORT_HOME"
echo "  benchmark/onnxruntime/.venv/bin/python benchmark/onnxruntime/mnist_cnn_bench.py"
echo "  python3 benchmark/tools/run_host_ab_suite_float32.py"
