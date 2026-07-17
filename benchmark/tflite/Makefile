# TensorFlow Lite (LiteRT) host benches — MPU-oriented interpreter path.
#
# Distinct from benchmark/tflm/ (TensorFlow Lite Micro for MCUs).
#
# Fair comparison with netkit ImageNet benches:
#   make -C benchmark/netkit run-mobilenetv4-imagenet-xnnpack
# uses BENCH_FLAG_PROFILE=tflite from benchmark/common/tflite_host_flags.mk
# (LiteRT opt: -O3 -DNDEBUG, host gcc/g++, gnu++20, -fpermissive; Darwin arm64 -ld_classic).

BENCH_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(BENCH_DIR)/../..)
VENV := $(BENCH_DIR)/.venv
PYTHON := $(VENV)/bin/python

.PHONY: venv run-mobilenetv4-imagenet run-mobilenetv4-imagenet-ref \
  run-mobilenetv4-imagenet-int8 run-mobilenetv4-imagenet-int8-ref \
  run-yolox run-yolox-ref \
  run-cnn run-cnn-ref run-cnn-int8 run-cnn-int8-ref \
  run-cnn-dw run-cnn-dw-int8 \
  run-mlp run-mlp-ref run-mlp-int8 run-mlp-int8-ref clean

venv:
	@test -x $(PYTHON) || python3 -m venv $(VENV)
	@$(PYTHON) -c "import ai_edge_litert" 2>/dev/null || \
	  $(VENV)/bin/pip install -U pip ai-edge-litert pillow

# Default: XNNPACK CPU delegate (typical TF Lite MPU/desktop path), 1 thread.
run-mobilenetv4-imagenet: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mobilenetv4_imagenet_f32.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mobilenetv4-imagenet
	@test -d $(ROOT)/benchmark/tflm/generated/imagenet_sample_cache || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-imagenet-mnv4-images
	@$(PYTHON) $(BENCH_DIR)/mobilenetv4_imagenet_bench.py --num-threads 1

# Builtin reference kernels only (no XNNPACK) — closer to TFLM-style portable float.
run-mobilenetv4-imagenet-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mobilenetv4_imagenet_f32.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mobilenetv4-imagenet
	@$(PYTHON) $(BENCH_DIR)/mobilenetv4_imagenet_bench.py --num-threads 1 --no-xnnpack

run-mobilenetv4-imagenet-int8: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mobilenetv4_imagenet_int8.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mobilenetv4-imagenet-int8
	@test -d $(ROOT)/benchmark/tflm/generated/imagenet_sample_cache || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-imagenet-mnv4-images
	@$(PYTHON) $(BENCH_DIR)/mobilenetv4_imagenet_int8_bench.py --num-threads 1

run-mobilenetv4-imagenet-int8-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mobilenetv4_imagenet_int8.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mobilenetv4-imagenet-int8
	@$(PYTHON) $(BENCH_DIR)/mobilenetv4_imagenet_int8_bench.py --num-threads 1 --no-xnnpack

run-yolox: venv
	@test -f $(ROOT)/benchmark/tflm/generated/yolox_mnv4_pafpn_320.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-yolox
	@test -d $(ROOT)/benchmark/tflm/generated/yolox_sample_cache || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-yolox-images
	@$(PYTHON) $(BENCH_DIR)/yolox_bench.py --num-threads 1

run-yolox-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/yolox_mnv4_pafpn_320.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-yolox
	@$(PYTHON) $(BENCH_DIR)/yolox_bench.py --num-threads 1 --no-xnnpack

# MNIST CNN float32 — XNNPACK vs builtin-ref.
run-cnn: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_cnn.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-assets
	@$(PYTHON) $(BENCH_DIR)/mnist_cnn_bench.py --num-threads 1

run-cnn-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_cnn.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-assets
	@$(PYTHON) $(BENCH_DIR)/mnist_cnn_bench.py --num-threads 1 --no-xnnpack

# MNIST CNN int8 — XNNPACK is the host optimized peer (LiteRT has no CMSIS-NN on CPU).
run-cnn-int8: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_cnn_int8.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-cnn-int8
	@$(PYTHON) $(BENCH_DIR)/mnist_cnn_int8_bench.py --num-threads 1

run-cnn-int8-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_cnn_int8.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-cnn-int8
	@$(PYTHON) $(BENCH_DIR)/mnist_cnn_int8_bench.py --num-threads 1 --no-xnnpack

# Depthwise-separable MNIST CNN float32 / int8 (XNNPACK).
run-cnn-dw: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_cnn_dw.tflite || \
	  (cd $(ROOT) && python3 tools/export_mnist_cnn_dw_assets.py --float-only)
	@$(PYTHON) $(BENCH_DIR)/mnist_cnn_bench.py --num-threads 1 \
	  --model $(ROOT)/benchmark/tflm/generated/mnist_cnn_dw.tflite \
	  --nk $(ROOT)/models/mnist_cnn_dw.nk

run-cnn-dw-int8: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_cnn_dw_int8.tflite || \
	  (cd $(ROOT) && python3 tools/export_mnist_cnn_dw_assets.py)
	@$(PYTHON) $(BENCH_DIR)/mnist_cnn_int8_bench.py --num-threads 1 \
	  --model $(ROOT)/benchmark/tflm/generated/mnist_cnn_dw_int8.tflite \
	  --nk $(ROOT)/models/mnist_cnn_dw.nk \
	  --quant-json $(ROOT)/benchmark/tflm/generated/mnist_cnn_dw_int8_tflite_quant.json

# MNIST MLP float32 — XNNPACK vs builtin-ref.
run-mlp: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_mlp.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-assets
	@$(PYTHON) $(BENCH_DIR)/mnist_mlp_bench.py --num-threads 1

run-mlp-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_mlp.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-assets
	@$(PYTHON) $(BENCH_DIR)/mnist_mlp_bench.py --num-threads 1 --no-xnnpack

# MNIST MLP int8 — XNNPACK is the host optimized peer.
run-mlp-int8: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_mlp_int8.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mlp-int8
	@$(PYTHON) $(BENCH_DIR)/mnist_mlp_int8_bench.py --num-threads 1

run-mlp-int8-ref: venv
	@test -f $(ROOT)/benchmark/tflm/generated/mnist_mlp_int8.tflite || \
	  $(MAKE) -C $(ROOT)/benchmark/tflm export-mlp-int8
	@$(PYTHON) $(BENCH_DIR)/mnist_mlp_int8_bench.py --num-threads 1 --no-xnnpack

clean:
	rm -rf $(VENV)
