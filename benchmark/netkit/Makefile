# netkit MNIST invoke benchmarks (pairs with benchmark/tflm/)
#
# Builds reference and CMSIS-DSP variants with TFLM-matched host flags.

ROOT := $(abspath ../..)

.PHONY: all run run-mlp run-cnn run-mlp-cmsis run-cnn-cmsis run-cmsis run-cnn-profile run-mlp-profile \
  run-cnn-int8 run-cnn-int8-ref run-cnn-xnnpack run-cnn-int8-xnnpack \
  run-cnn-dw-xnnpack run-cnn-dw-int8-xnnpack \
  run-mlp-xnnpack run-mlp-int8 run-mlp-int8-ref run-mlp-int8-xnnpack \
  run-mobilenetv4 run-mobilenetv4-int8 run-mobilenetv4-cmsis \
  run-mobilenetv4-imagenet run-mobilenetv4-imagenet-cmsis run-mobilenetv4-imagenet-xnnpack \
  run-mobilenetv4-imagenet-int8 run-mobilenetv4-imagenet-int8-xnnpack \
  run-aot run-aot-mlp run-aot-cnn run-aot-cmsis run-aot-mlp-cmsis run-aot-cnn-cmsis run-all clean clean-all

all: run

run-all: run run-cmsis

run: run-mlp run-cnn

run-cmsis: run-mlp-cmsis run-cnn-cmsis

run-mlp:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MLP_BENCH=mnist_mlp_bench \
	  CNN_BENCH=mnist_cnn_bench \
	  MLP_MAIN_OBJ=src/main.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main.o \
	  run-mlp

run-cnn:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MLP_BENCH=mnist_mlp_bench \
	  CNN_BENCH=mnist_cnn_bench \
	  MLP_MAIN_OBJ=src/main.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main.o \
	  run-cnn

run-mlp-unroll:
	$(MAKE) -f bench.mk BACKEND=loop-unroll CMSIS_DSP=0 NETKIT_LOOP_UNROLL=1 \
	  BENCH_OBJDIR=bench_obj_unroll \
	  BENCH_LIB=libnetkit_bench_unroll.a \
	  MLP_BENCH=mnist_mlp_bench_unroll \
	  CNN_BENCH=mnist_cnn_bench_unroll \
	  MLP_MAIN_OBJ=src/main_unroll.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_unroll.o \
	  run-mlp

run-cnn-unroll:
	$(MAKE) -f bench.mk BACKEND=loop-unroll CMSIS_DSP=0 NETKIT_LOOP_UNROLL=1 \
	  BENCH_OBJDIR=bench_obj_unroll \
	  BENCH_LIB=libnetkit_bench_unroll.a \
	  MLP_BENCH=mnist_mlp_bench_unroll \
	  CNN_BENCH=mnist_cnn_bench_unroll \
	  MLP_MAIN_OBJ=src/main_unroll.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_unroll.o \
	  run-cnn

run-mlp-cmsis:
	$(MAKE) -f bench.mk BACKEND=cmsis-dsp CMSIS_DSP=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj_cmsis \
	  BENCH_LIB=libnetkit_bench_cmsis.a \
	  MLP_BENCH=mnist_mlp_bench_cmsis \
	  CNN_BENCH=mnist_cnn_bench_cmsis \
	  MLP_MAIN_OBJ=src/main_cmsis.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_cmsis.o \
	  run-mlp

run-cnn-cmsis:
	$(MAKE) -f bench.mk BACKEND=cmsis-dsp CMSIS_DSP=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj_cmsis \
	  BENCH_LIB=libnetkit_bench_cmsis.a \
	  MLP_BENCH=mnist_mlp_bench_cmsis \
	  CNN_BENCH=mnist_cnn_bench_cmsis \
	  MLP_MAIN_OBJ=src/main_cmsis.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_cmsis.o \
	  run-cnn

# Host MNIST CNN int8: QuantOps reference (CMSIS-NN is MCU-only — see nucleo boards).
run-cnn-int8-ref:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 CMSIS_NN=0 XNNPACK=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_ref \
	  BENCH_LIB=libnetkit_bench_tflite_ref.a \
	  CNN_INT8_BENCH=mnist_cnn_int8_bench_ref \
	  CNN_INT8_MAIN_OBJ=src/mnist_cnn_int8_main_ref.o \
	  run-cnn-int8

run-cnn-int8: run-cnn-int8-ref

# Host MNIST CNN float/int8 with XNNPACK full-network subgraph (LiteRT peer).
run-cnn-xnnpack:
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  CNN_BENCH=mnist_cnn_bench_xnnpack \
	  CNN_MAIN_OBJ=src/mnist_cnn_main_xnnpack.o \
	  run-cnn

run-cnn-int8-xnnpack:
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 CMSIS_NN=0 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  CNN_INT8_BENCH=mnist_cnn_int8_bench_xnnpack \
	  CNN_INT8_MAIN_OBJ=src/mnist_cnn_int8_main_xnnpack.o \
	  run-cnn-int8

# Depthwise-separable MNIST CNN (PW/DW/Pool x2) — same XNNPACK peer path as run-cnn-xnnpack.
run-cnn-dw-xnnpack:
	@test -f $(ROOT)/models/mnist_cnn_dw.nk || \
	  (cd $(ROOT) && python3 tools/export_mnist_cnn_dw.py)
	@test -f $(ROOT)/benchmark/tflm/generated/cnn_dw/mnist_cnn_test_images.cc || \
	  (cd $(ROOT) && python3 tools/export_mnist_cnn_dw_assets.py --float-only)
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  EXTRA_BENCH_INCLUDES=-I../tflm/generated/cnn_dw \
	  CNN_BENCH=mnist_cnn_dw_bench_xnnpack \
	  CNN_MAIN_OBJ=src/mnist_cnn_dw_main_xnnpack.o \
	  CNN_IMAGES_CC=../tflm/generated/cnn_dw/mnist_cnn_test_images.cc \
	  CNN_IMAGES_OBJ=../tflm/generated/cnn_dw/mnist_cnn_test_images.o \
	  CNN_MODEL=models/mnist_cnn_dw.nk \
	  run-cnn

run-cnn-dw-int8-xnnpack:
	@test -f $(ROOT)/models/mnist_cnn_dw_int8.nk || \
	  (cd $(ROOT) && python3 tools/export_mnist_cnn_dw_int8.py)
	@test -f $(ROOT)/benchmark/tflm/generated/cnn_dw/mnist_cnn_int8_test_images.cc || \
	  (cd $(ROOT) && python3 tools/export_mnist_cnn_dw_assets.py)
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 CMSIS_NN=0 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  EXTRA_BENCH_INCLUDES=-I../tflm/generated/cnn_dw \
	  CNN_INT8_BENCH=mnist_cnn_dw_int8_bench_xnnpack \
	  CNN_INT8_MAIN_OBJ=src/mnist_cnn_dw_int8_main_xnnpack.o \
	  CNN_INT8_IMAGES_CC=../tflm/generated/cnn_dw/mnist_cnn_int8_test_images.cc \
	  CNN_INT8_IMAGES_OBJ=../tflm/generated/cnn_dw/mnist_cnn_int8_test_images.o \
	  CNN_INT8_MODEL=models/mnist_cnn_dw_int8.nk \
	  run-cnn-int8

# Host MNIST MLP float/int8 with XNNPACK (LiteRT peer).
run-mlp-xnnpack:
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  MLP_BENCH=mnist_mlp_bench_xnnpack \
	  MLP_MAIN_OBJ=src/main_xnnpack.o \
	  run-mlp

run-mlp-int8-ref:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 CMSIS_NN=0 XNNPACK=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_ref \
	  BENCH_LIB=libnetkit_bench_tflite_ref.a \
	  MLP_INT8_BENCH=mnist_mlp_int8_bench_ref \
	  MLP_INT8_MAIN_OBJ=src/mnist_mlp_int8_main_ref.o \
	  run-mlp-int8

run-mlp-int8-xnnpack:
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 CMSIS_NN=0 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  MLP_INT8_BENCH=mnist_mlp_int8_bench_xnnpack \
	  MLP_INT8_MAIN_OBJ=src/mnist_mlp_int8_main_xnnpack.o \
	  run-mlp-int8

run-mlp-int8: run-mlp-int8-ref

# MobileNetV4-small float32 (host/CPU) — depthwise-heavy invoke benchmark.
run-mobilenetv4:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MNV4_BENCH=mobilenetv4_bench \
	  MNV4_MAIN_OBJ=src/mobilenetv4_main.o \
	  run-mobilenetv4

run-mobilenetv4-cmsis:
	$(MAKE) -f bench.mk BACKEND=cmsis-dsp CMSIS_DSP=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj_cmsis \
	  BENCH_LIB=libnetkit_bench_cmsis.a \
	  MNV4_BENCH=mobilenetv4_bench_cmsis \
	  MNV4_MAIN_OBJ=src/mobilenetv4_main_cmsis.o \
	  run-mobilenetv4

run-mobilenetv4-int8:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 XNNPACK=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MNV4_BENCH=mobilenetv4_bench \
	  MNV4_MAIN_OBJ=src/mobilenetv4_main.o \
	  run-mobilenetv4-int8

# Int8 MobileNetV4-small with XNNPACK qs8 (same NETKIT_XNNPACK / XNNPACK=1 flag as float).
run-mobilenetv4-int8-xnnpack:
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  MNV4_BENCH=mobilenetv4_bench_int8_xnnpack \
	  MNV4_MAIN_OBJ=src/mobilenetv4_main_int8_xnnpack.o \
	  run-mobilenetv4-int8

# Pretrained ImageNet MobileNetV4-Conv-Small (224x224x3, 1000 classes).
# Uses BENCH_FLAG_PROFILE=tflite (LiteRT-matched -O3 -DNDEBUG, gnu++20,
# -fpermissive, Darwin arm64 -ld_classic) — not the TFLM Micro host profile.
run-mobilenetv4-imagenet:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 XNNPACK=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_ref \
	  BENCH_LIB=libnetkit_bench_tflite_ref.a \
	  MNV4_IMAGENET_BENCH=mobilenetv4_imagenet_bench \
	  MNV4_IMAGENET_MAIN_OBJ=src/mobilenetv4_imagenet_main.o \
	  run-mobilenetv4-imagenet

run-mobilenetv4-imagenet-cmsis:
	$(MAKE) -f bench.mk BACKEND=cmsis-dsp CMSIS_DSP=1 XNNPACK=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_cmsis \
	  BENCH_LIB=libnetkit_bench_tflite_cmsis.a \
	  MNV4_IMAGENET_BENCH=mobilenetv4_imagenet_bench_cmsis \
	  MNV4_IMAGENET_MAIN_OBJ=src/mobilenetv4_imagenet_main_cmsis.o \
	  run-mobilenetv4-imagenet

# XNNPACK LayerFast + CMSIS-DSP VectorFast (cpu/mpu default profile).
run-mobilenetv4-imagenet-xnnpack:
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  MNV4_IMAGENET_BENCH=mobilenetv4_imagenet_bench_xnnpack \
	  MNV4_IMAGENET_MAIN_OBJ=src/mobilenetv4_imagenet_main_xnnpack.o \
	  run-mobilenetv4-imagenet

# Int8 ImageNet: prequantized int8 fixtures (Python export); no C++ float↔int8.
# Default uses XNNPACK qs8 (same as float ImageNet xnnpack target).
run-mobilenetv4-imagenet-int8:
	$(MAKE) -f bench.mk BACKEND=xnnpack CMSIS_DSP=1 XNNPACK=1 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_xnnpack \
	  BENCH_LIB=libnetkit_bench_tflite_xnnpack.a \
	  MNV4_IMAGENET_INT8_BENCH=mobilenetv4_imagenet_int8_bench_xnnpack \
	  MNV4_IMAGENET_INT8_MAIN_OBJ=src/mobilenetv4_imagenet_int8_main_xnnpack.o \
	  run-mobilenetv4-imagenet-int8

run-mobilenetv4-imagenet-int8-xnnpack: run-mobilenetv4-imagenet-int8

run-mobilenetv4-imagenet-int8-ref:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 XNNPACK=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_FLAG_PROFILE=tflite \
	  BENCH_OBJDIR=bench_obj_tflite_ref \
	  BENCH_LIB=libnetkit_bench_tflite_ref.a \
	  MNV4_IMAGENET_INT8_BENCH=mobilenetv4_imagenet_int8_bench_ref \
	  MNV4_IMAGENET_INT8_MAIN_OBJ=src/mobilenetv4_imagenet_int8_main_ref.o \
	  run-mobilenetv4-imagenet-int8

run-cnn-profile:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  CNN_PROFILE_BENCH=mnist_cnn_profile_bench \
	  CNN_PROFILE_MAIN_OBJ=src/mnist_cnn_profile_main.o \
	  run-cnn-profile

run-mlp-profile:
	$(MAKE) -f bench.mk BACKEND=reference CMSIS_DSP=0 NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=bench_obj \
	  BENCH_LIB=libnetkit_bench.a \
	  MLP_PROFILE_BENCH=mnist_mlp_profile_bench \
	  MLP_PROFILE_MAIN_OBJ=src/mnist_mlp_profile_main.o \
	  run-mlp-profile

run-aot: run-aot-mlp run-aot-cnn

run-aot-mlp:
	$(MAKE) -f aot_bench.mk run-mlp

run-aot-cnn:
	$(MAKE) -f aot_bench.mk run-cnn

run-aot-cmsis: run-aot-mlp-cmsis run-aot-cnn-cmsis

run-aot-mlp-cmsis:
	$(MAKE) -f aot_bench.mk run-mlp-cmsis

run-aot-cnn-cmsis:
	$(MAKE) -f aot_bench.mk run-cnn-cmsis

clean:
	rm -f mnist_mlp_bench mnist_cnn_bench mnist_cnn_profile_bench mnist_mlp_profile_bench \
	  mnist_mlp_bench_cmsis mnist_cnn_bench_cmsis \
	  mobilenetv4_bench mobilenetv4_bench_cmsis \
	  mobilenetv4_imagenet_bench mobilenetv4_imagenet_bench_cmsis \
	  mobilenetv4_imagenet_bench_xnnpack \
	  mnist_mlp_bench_aot mnist_cnn_bench_aot \
	  mnist_mlp_bench_aot_cmsis mnist_cnn_bench_aot_cmsis \
	  libnetkit_bench.a libnetkit_bench_cmsis.a \
	  libnetkit_bench_tflite_ref.a libnetkit_bench_tflite_cmsis.a \
	  libnetkit_bench_tflite_xnnpack.a \
	  src/main.o src/mnist_cnn_main.o src/mnist_cnn_profile_main.o src/mnist_mlp_profile_main.o \
	  src/main_cmsis.o src/mnist_cnn_main_cmsis.o \
	  src/mobilenetv4_main.o src/mobilenetv4_main_cmsis.o \
	  src/mobilenetv4_imagenet_main.o src/mobilenetv4_imagenet_main_cmsis.o \
	  src/mobilenetv4_imagenet_main_xnnpack.o \
	  src/mnist_mlp_aot_main.o src/mnist_cnn_aot_main.o \
	  src/mnist_mlp_aot_main_cmsis.o src/mnist_cnn_aot_main_cmsis.o \
	  generated/aot/mnist_mlp_aot.o generated/aot/mnist_cnn_aot.o \
	  ../tflm/generated/mnist_test_images.o ../tflm/generated/mnist_cnn_test_images.o \
	  ../tflm/generated/imagenet_mnv4_test_images.o

clean-all: clean
	rm -rf bench_obj bench_obj_cmsis bench_obj_xnnpack \
	  bench_obj_tflite_ref bench_obj_tflite_cmsis bench_obj_tflite_xnnpack \
	  generated/aot
