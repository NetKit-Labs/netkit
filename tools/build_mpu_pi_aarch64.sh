#!/usr/bin/env bash
# Cross-build host A/B netkit benches for Raspberry Pi (linux/aarch64) in Docker.
#
# Outputs under:
#   third_party/XNNPACK/netkit_lib_linux_aarch64/   (XNNPACK static libs)
#   benchmark/mpu_pi/bin/                           (bench ELFs)
#
# Usage:
#   ./tools/build_mpu_pi_aarch64.sh                  # float32 + int8
#   ./tools/build_mpu_pi_aarch64.sh --dtype int8
#   ./tools/build_mpu_pi_aarch64.sh --dtype float32 --skip-xnnpack
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BIN="$ROOT/benchmark/mpu_pi/bin"
OUT_XNN="$ROOT/third_party/XNNPACK/netkit_lib_linux_aarch64"
SKIP_XNN=0
DTYPE="both"
for arg in "$@"; do
  case "$arg" in
    --skip-xnnpack) SKIP_XNN=1 ;;
    --dtype=*) DTYPE="${arg#*=}" ;;
    --dtype)
      shift_next=1
      ;;
    float32|int8|both)
      if [[ "${shift_next:-0}" == "1" ]]; then DTYPE="$arg"; shift_next=0; fi
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
  esac
done
# Also accept: --dtype int8 as two tokens
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--dtype" && $((i+1)) -lt ${#args[@]} ]]; then
    DTYPE="${args[$((i+1))]}"
  fi
done
case "$DTYPE" in
  float32|int8|both) ;;
  *) echo "bad --dtype $DTYPE (float32|int8|both)" >&2; exit 2 ;;
esac

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to cross-build linux/aarch64 on this host" >&2
  exit 1
fi

mkdir -p "$OUT_BIN" "$OUT_XNN"

IMAGE="${NETKIT_PI_DOCKER_IMAGE:-debian:trixie}"

docker run --rm --platform linux/arm64 \
  -v "$ROOT:/src:rw" \
  -e SKIP_XNN="$SKIP_XNN" \
  -e DTYPE="$DTYPE" \
  -w /src \
  "$IMAGE" \
  bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential cmake git python3 ca-certificates pkg-config

if [[ "${SKIP_XNN}" != "1" ]]; then
  if [[ ! -d third_party/XNNPACK/.git ]]; then
    echo "XNNPACK sources missing — run ./tools/fetch_xnnpack.sh on the host first" >&2
    exit 1
  fi
  PIN="$(cat third_party/XNNPACK/netkit_lib/.xnnpack_pin 2>/dev/null || true)"
  BUILD_DIR=third_party/XNNPACK/build_linux_aarch64
  if [[ -f third_party/XNNPACK/netkit_lib_linux_aarch64/libXNNPACK.a ]]; then
    echo "Reusing existing third_party/XNNPACK/netkit_lib_linux_aarch64/"
  else
    rm -rf "$BUILD_DIR"
    cmake -S third_party/XNNPACK -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DXNNPACK_LIBRARY_TYPE=static \
      -DXNNPACK_BUILD_TESTS=OFF \
      -DXNNPACK_BUILD_BENCHMARKS=OFF \
      -DXNNPACK_BUILD_ALL_MICROKERNELS=ON \
      -DXNNPACK_ENABLE_ASSEMBLY=ON
    cmake --build "$BUILD_DIR" -j"$(nproc)"
    mkdir -p third_party/XNNPACK/netkit_lib_linux_aarch64
    find "$BUILD_DIR" \( -name "libXNNPACK.a" -o -name "libxnnpack.a" \
      -o -name "libpthreadpool.a" -o -name "libcpuinfo.a" \
      -o -name "libfxdiv.a" -o -name "libkleidiai.a" \
      -o -name "libxnnpack-microkernels-*.a" \) \
      | while read -r lib; do
          cp -f "$lib" third_party/XNNPACK/netkit_lib_linux_aarch64/
        done
    if [[ -f third_party/XNNPACK/netkit_lib_linux_aarch64/libxnnpack.a && \
          ! -f third_party/XNNPACK/netkit_lib_linux_aarch64/libXNNPACK.a ]]; then
      cp -f third_party/XNNPACK/netkit_lib_linux_aarch64/libxnnpack.a \
            third_party/XNNPACK/netkit_lib_linux_aarch64/libXNNPACK.a
    fi
    if [[ -n "$PIN" ]]; then
      echo "$PIN" > third_party/XNNPACK/netkit_lib_linux_aarch64/.xnnpack_pin
    fi
  fi
  ls -la third_party/XNNPACK/netkit_lib_linux_aarch64/
fi

# Float + int8 DS-CNN image fixtures for linking.
if [[ ! -f benchmark/tflm/generated/cnn_dw/mnist_cnn_test_images.cc || \
      ! -f benchmark/tflm/generated/cnn_dw/mnist_cnn_int8_test_images.cc ]]; then
  python3 tools/export_mnist_cnn_dw_assets.py
fi

build_one() {
  local dtype="$1" model="$2" xnn="$3"
  local xt tag name common
  if [[ "$xnn" == "1" ]]; then xt=xnn; else xt=ref; fi
  if [[ "$dtype" == "float32" ]]; then tag="f32_${xt}"; else tag="i8_${xt}"; fi
  common=(
    make -C benchmark/netkit -f bench.mk
    "BACKEND=$([[ $xnn == 1 ]] && echo xnnpack || echo reference)"
    CMSIS_NN=0
    "XNNPACK=$xnn"
    NETKIT_IM2COL=0
    NETKIT_LOOP_UNROLL=0
    BENCH_FLAG_PROFILE=tflite
    "BENCH_OBJDIR=bench_obj_pi_${tag}"
    "BENCH_LIB=libnetkit_bench_pi_${tag}.a"
    "EXTRA_BENCH_CPPFLAGS=-DNETKIT_BENCH_ARENA_MB=96"
  )
  if [[ "$xnn" == "1" ]]; then
    common+=("XNNPACK_LIB_DIR=../../third_party/XNNPACK/netkit_lib_linux_aarch64")
  fi

  if [[ "$dtype" == "float32" ]]; then
    case "$model" in
      cnn)
        name="mnist_cnn_bench_pi_${tag}"
        "${common[@]}" \
          "CNN_BENCH=$name" \
          "CNN_MAIN_OBJ=src/mnist_cnn_main_pi_${tag}.o" \
          CNN_IMAGES_OBJ=../tflm/generated/mnist_cnn_test_images_pi.o \
          build-cnn
        ;;
      cnn_dw)
        name="mnist_cnn_dw_bench_pi_${tag}"
        "${common[@]}" \
          EXTRA_BENCH_INCLUDES=-I../tflm/generated/cnn_dw \
          "CNN_BENCH=$name" \
          "CNN_MAIN_OBJ=src/mnist_cnn_dw_main_pi_${tag}.o" \
          CNN_IMAGES_CC=../tflm/generated/cnn_dw/mnist_cnn_test_images.cc \
          CNN_IMAGES_OBJ=../tflm/generated/cnn_dw/mnist_cnn_test_images_pi.o \
          build-cnn
        ;;
      imagenet)
        name="mobilenetv4_imagenet_bench_pi_${tag}"
        "${common[@]}" \
          "MNV4_IMAGENET_BENCH=$name" \
          "MNV4_IMAGENET_MAIN_OBJ=src/mobilenetv4_imagenet_main_pi_${tag}.o" \
          IMAGENET_IMAGES_OBJ=../tflm/generated/imagenet_mnv4_test_images_pi.o \
          build-mobilenetv4-imagenet
        ;;
      *) echo "unknown model $model" >&2; exit 1 ;;
    esac
  else
    case "$model" in
      cnn)
        name="mnist_cnn_int8_bench_pi_${tag}"
        "${common[@]}" \
          "CNN_INT8_BENCH=$name" \
          "CNN_INT8_MAIN_OBJ=src/mnist_cnn_int8_main_pi_${tag}.o" \
          CNN_INT8_IMAGES_OBJ=../tflm/generated/mnist_cnn_int8_test_images_pi.o \
          build-cnn-int8
        ;;
      cnn_dw)
        name="mnist_cnn_dw_int8_bench_pi_${tag}"
        "${common[@]}" \
          EXTRA_BENCH_INCLUDES=-I../tflm/generated/cnn_dw \
          "CNN_INT8_BENCH=$name" \
          "CNN_INT8_MAIN_OBJ=src/mnist_cnn_dw_int8_main_pi_${tag}.o" \
          CNN_INT8_IMAGES_CC=../tflm/generated/cnn_dw/mnist_cnn_int8_test_images.cc \
          CNN_INT8_IMAGES_OBJ=../tflm/generated/cnn_dw/mnist_cnn_int8_test_images_pi.o \
          build-cnn-int8
        ;;
      imagenet)
        name="mobilenetv4_imagenet_int8_bench_pi_${tag}"
        "${common[@]}" \
          "MNV4_IMAGENET_INT8_BENCH=$name" \
          "MNV4_IMAGENET_INT8_MAIN_OBJ=src/mobilenetv4_imagenet_int8_main_pi_${tag}.o" \
          IMAGENET_INT8_IMAGES_OBJ=../tflm/generated/imagenet_mnv4_netkit_int8_test_images_pi.o \
          build-mobilenetv4-imagenet-int8
        ;;
      *) echo "unknown model $model" >&2; exit 1 ;;
    esac
  fi
  install -m 755 "benchmark/netkit/$name" "benchmark/mpu_pi/bin/$name"
  echo "built benchmark/mpu_pi/bin/$name"
}

mkdir -p benchmark/mpu_pi/bin
dtypes=()
case "$DTYPE" in
  float32) dtypes=(float32) ;;
  int8) dtypes=(int8) ;;
  both) dtypes=(float32 int8) ;;
esac
for dtype in "${dtypes[@]}"; do
  for model in cnn cnn_dw imagenet; do
    build_one "$dtype" "$model" 0
    if [[ "${SKIP_XNN}" != "1" ]]; then
      build_one "$dtype" "$model" 1
    fi
  done
done

echo "=== linux/aarch64 benches ==="
file benchmark/mpu_pi/bin/* || true
ls -la benchmark/mpu_pi/bin/
'

echo "Done. Binaries in $OUT_BIN"
