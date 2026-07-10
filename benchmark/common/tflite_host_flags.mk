# Host compile/link flags for fair comparison with TensorFlow Lite / LiteRT (MPU).
#
# Distinct from tflm_host_flags.mk, which mirrors TFLM Micro (-O2 +
# TF_LITE_DISABLE_X86_NEON) for MNIST/TFLM apples-to-apples runs.
#
# LiteRT pip wheels (ai_edge_litert) are built with Bazel roughly as:
#   bazel build -c opt --copt=-O3 --cxxopt=-std=gnu++17 ...
#   (see LiteRT ci/build_pip_package_with_bazel.sh; Darwin adds --config=macos_wheel)
# That means: host Apple/Linux clang or gcc, -O3 -DNDEBUG, SIMD on, exceptions/RTTI
# left at toolchain defaults (LiteRT's libLiteRt.dylib references __cxa_throw).
#
# Policy: mimic those flags; add netkit-only options only when required.
# Variable names reuse the TFLM_* namespace so bench.mk can switch profiles
# without duplicating recipes.
#
# Include from benchmark/netkit/bench.mk when BENCH_FLAG_PROFILE=tflite.

# Same driver names LiteRT/Bazel use on Darwin (cc/c++ → Apple clang).
TFLM_HOST_CC := cc
TFLM_HOST_CXX := c++
TFLM_HOST_AR := ar

# --- LiteRT-matched (no TFLM Micro extras) ---
# Do NOT add: -fno-rtti, -fno-exceptions, -fno-threadsafe-statics,
# -ffunction-sections, -fdata-sections, TF_LITE_STATIC_MEMORY,
# TF_LITE_DISABLE_X86_NEON, or TFLM's warning laundry list.
TFLM_CXXFLAGS := -std=c++20

# Bazel -c opt is -O2 -DNDEBUG; LiteRT opt wheels also pass --copt=-O3.
TFLM_CORE_OPT := -O3 -DNDEBUG
TFLM_KERNEL_OPT := -O3 -DNDEBUG

TFLM_LDFLAGS := -lm

# --- netkit-only (required for this runtime / fair file-load path) ---
NETKIT_IM2COL ?= 0
NETKIT_LOOP_UNROLL ?= 0
# CPU/host benches: mmap .nk file loads (matches TF Lite model_path mmap default).
NETKIT_BENCH_CPPFLAGS := \
  -DNETKIT_TARGET_CPU=1 \
  -DNETKIT_USE_MMAP=1 \
  -DNETKIT_IM2COL=$(NETKIT_IM2COL) \
  -DNETKIT_LOOP_UNROLL=$(NETKIT_LOOP_UNROLL)

TFLM_BENCH_INCLUDES := -I$(ROOT)/include -I$(SHARED_GEN) -I../common

TFLM_BENCH_CORE_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_CORE_OPT) $(NETKIT_BENCH_CPPFLAGS) $(TFLM_BENCH_INCLUDES)
TFLM_BENCH_KERNEL_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_KERNEL_OPT) $(NETKIT_BENCH_CPPFLAGS) -I$(ROOT)/include
