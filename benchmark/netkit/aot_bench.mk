# AOT netkit MNIST benchmarks (lowered static kernel call chain + TFLM-matched flags).
# CMSIS-DSP is not used — reference backend only.

ROOT := $(abspath ../..)
SHARED_GEN := ../tflm/generated
AOT_GEN := generated/aot
include ../common/tflm_host_flags.mk

BENCH_KERNEL_CPPFLAGS := $(NETKIT_BENCH_CPPFLAGS)
BENCH_KERNEL_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_KERNEL_OPT) $(BENCH_KERNEL_CPPFLAGS) -I$(ROOT)/include
BENCH_CORE_CXXFLAGS := $(TFLM_CXXFLAGS) $(TFLM_CORE_OPT) $(BENCH_KERNEL_CPPFLAGS) $(TFLM_BENCH_INCLUDES) -I.

REF_BENCH_OBJDIR := bench_obj
REF_BENCH_LIB := libnetkit_bench.a

MLP_AOT_BENCH := mnist_mlp_bench_aot
CNN_AOT_BENCH := mnist_cnn_bench_aot
MLP_AOT_GEN := $(AOT_GEN)/mnist_mlp_aot.cpp
CNN_AOT_GEN := $(AOT_GEN)/mnist_cnn_aot.cpp
MLP_AOT_MAIN := src/mnist_mlp_aot_main.o
CNN_AOT_MAIN := src/mnist_cnn_aot_main.o
MLP_IMAGES_CC := $(SHARED_GEN)/mnist_test_images.cc
CNN_IMAGES_CC := $(SHARED_GEN)/mnist_cnn_test_images.cc
MLP_IMAGES_OBJ := $(SHARED_GEN)/mnist_test_images.o
CNN_IMAGES_OBJ := $(SHARED_GEN)/mnist_cnn_test_images.o

.PHONY: run-mlp run-cnn

$(MLP_AOT_GEN) $(AOT_GEN)/mnist_mlp_aot.hpp $(CNN_AOT_GEN) $(AOT_GEN)/mnist_cnn_aot.hpp:
	@python3 ../tools/export_aot_assets.py

$(MLP_IMAGES_CC) $(CNN_IMAGES_CC):
	$(MAKE) -C ../tflm export-assets

$(REF_BENCH_LIB):
	$(MAKE) -f bench.mk BACKEND=reference NETKIT_LOOP_UNROLL=0 \
	  BENCH_OBJDIR=$(REF_BENCH_OBJDIR) \
	  BENCH_LIB=$(REF_BENCH_LIB) \
	  MLP_BENCH=mnist_mlp_bench \
	  CNN_BENCH=mnist_cnn_bench \
	  MLP_MAIN_OBJ=src/main.o \
	  CNN_MAIN_OBJ=src/mnist_cnn_main.o \
	  build-lib

$(MLP_AOT_MAIN): src/mnist_mlp_aot_main.cc $(AOT_GEN)/mnist_mlp_aot.hpp $(MLP_AOT_GEN) $(MLP_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -DNETKIT_BENCH_BACKEND=\"aot\" -c $< -o $@

$(CNN_AOT_MAIN): src/mnist_cnn_aot_main.cc $(AOT_GEN)/mnist_cnn_aot.hpp $(CNN_AOT_GEN) $(CNN_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -DNETKIT_BENCH_BACKEND=\"aot\" -c $< -o $@

$(MLP_IMAGES_OBJ): $(MLP_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -c $< -o $@

$(CNN_IMAGES_OBJ): $(CNN_IMAGES_CC)
	$(TFLM_HOST_CXX) $(BENCH_CORE_CXXFLAGS) -c $< -o $@

$(AOT_GEN)/%.o: $(AOT_GEN)/%.cpp $(AOT_GEN)/%.hpp
	$(TFLM_HOST_CXX) $(BENCH_KERNEL_CXXFLAGS) -I$(AOT_GEN) -I. -c $< -o $@

$(MLP_AOT_BENCH): $(REF_BENCH_LIB) $(MLP_AOT_MAIN) $(MLP_IMAGES_OBJ) $(AOT_GEN)/mnist_mlp_aot.o
	$(TFLM_HOST_CXX) $(TFLM_CXXFLAGS) $(NETKIT_BENCH_CPPFLAGS) $(BENCH_CORE_CXXFLAGS) \
	  -o $@ $(MLP_AOT_MAIN) $(MLP_IMAGES_OBJ) $(AOT_GEN)/mnist_mlp_aot.o $(REF_BENCH_LIB) $(TFLM_LDFLAGS)

$(CNN_AOT_BENCH): $(REF_BENCH_LIB) $(CNN_AOT_MAIN) $(CNN_IMAGES_OBJ) $(AOT_GEN)/mnist_cnn_aot.o
	$(TFLM_HOST_CXX) $(TFLM_CXXFLAGS) $(NETKIT_BENCH_CPPFLAGS) $(BENCH_CORE_CXXFLAGS) \
	  -o $@ $(CNN_AOT_MAIN) $(CNN_IMAGES_OBJ) $(AOT_GEN)/mnist_cnn_aot.o $(REF_BENCH_LIB) $(TFLM_LDFLAGS)

run-mlp: $(MLP_AOT_BENCH)
	@cd $(ROOT) && ./benchmark/netkit/$(MLP_AOT_BENCH)

run-cnn: $(CNN_AOT_BENCH)
	@cd $(ROOT) && ./benchmark/netkit/$(CNN_AOT_BENCH)
