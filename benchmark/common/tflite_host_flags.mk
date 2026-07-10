# Host compile/link flags for fair comparison with TensorFlow Lite / LiteRT (MPU).
#
# Distinct from tflm_host_flags.mk, which mirrors TFLM Micro (-O2 +
# TF_LITE_DISABLE_X86_NEON) for MNIST/TFLM apples-to-apples runs.
#
# Matched to LiteRT pip wheels (ai_edge_litert), built roughly as:
#   bazel build -c opt --copt=-O3 --cxxopt=-std=gnu++17 ...
#   + --config=macos_wheel on Darwin
#   + --linkopt=-ld_classic on Darwin arm64
#   + --cxxopt=-fpermissive on macos/linux
# (see LiteRT ci/build_pip_package_with_bazel.sh and .bazelrc)
#
# Policy: same compiler/linker drivers and shared flags as LiteRT; add
# netkit-only options only when required (C++20, NETKIT_* defines).
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
#
# Dialect: LiteRT uses -std=gnu++17. netkit needs C++20 (std::span, etc.), so
# use gnu++20 — same GNU dialect, newer language mode.
TFLM_CXXFLAGS := -std=gnu++20

# LiteRT .bazelrc: build:macos/linux --cxxopt=-fpermissive
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  TFLM_CXXFLAGS += -fpermissive
else ifeq ($(UNAME_S),Linux)
  TFLM_CXXFLAGS += -fpermissive
endif

# Bazel -c opt implies -O2 -DNDEBUG; LiteRT opt wheels also pass --copt=-O3.
TFLM_CORE_OPT := -O3 -DNDEBUG
TFLM_KERNEL_OPT := -O3 -DNDEBUG

TFLM_LDFLAGS := -lm

# LiteRT Darwin arm64 wheels pass --linkopt=-ld_classic (deprecated but still
# what ai_edge_litert 2.1.6 ships with). Keep for peer parity on Apple Silicon.
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    TFLM_LDFLAGS += -ld_classic
  endif
endif

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
