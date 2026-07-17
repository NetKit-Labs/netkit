# ONNX Runtime host peer benches (LiteRT-matched build with XNNPACK EP).
#
#   ./tools/build_onnxruntime_litert_matched.sh   # once
#   python3 benchmark/tools/export_host_onnx_assets.py --all
#   make -C benchmark/onnxruntime run-cnn
#   make -C benchmark/onnxruntime run-cnn-ref

ORT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(abspath $(ORT_DIR)/../..)
PY := $(ORT_DIR)/.venv/bin/python

.PHONY: venv check-ort run-cnn run-cnn-ref run-cnn-dw run-cnn-dw-ref \
	run-imagenet run-imagenet-ref run-cnn-int8 run-cnn-int8-ref \
	run-imagenet-int8 run-imagenet-int8-ref

venv:
	@test -x $(PY) || (echo "missing $(PY) — run ./tools/build_onnxruntime_litert_matched.sh" && exit 1)

check-ort: venv
	$(PY) -c "import onnxruntime as o; ps=o.get_available_providers(); assert any(p.lower()=='xnnpackexecutionprovider' for p in ps), ps; print(o.__version__, ps)"

run-cnn: check-ort
	$(PY) $(ORT_DIR)/mnist_cnn_bench.py --num-threads 1

run-cnn-ref: check-ort
	$(PY) $(ORT_DIR)/mnist_cnn_bench.py --num-threads 1 --no-xnnpack

run-cnn-dw: check-ort
	$(PY) $(ORT_DIR)/mnist_cnn_bench.py --num-threads 1 \
	  --model $(ORT_DIR)/models/mnist_cnn_dw.onnx \
	  --nk $(ROOT)/models/mnist_cnn_dw.nk

run-cnn-dw-ref: check-ort
	$(PY) $(ORT_DIR)/mnist_cnn_bench.py --num-threads 1 --no-xnnpack \
	  --model $(ORT_DIR)/models/mnist_cnn_dw.onnx \
	  --nk $(ROOT)/models/mnist_cnn_dw.nk

run-imagenet: check-ort
	$(PY) $(ORT_DIR)/mobilenetv4_imagenet_bench.py --num-threads 1

run-imagenet-ref: check-ort
	$(PY) $(ORT_DIR)/mobilenetv4_imagenet_bench.py --num-threads 1 --no-xnnpack

run-cnn-int8: check-ort
	$(PY) $(ORT_DIR)/mnist_cnn_int8_bench.py --num-threads 1

run-cnn-int8-ref: check-ort
	$(PY) $(ORT_DIR)/mnist_cnn_int8_bench.py --num-threads 1 --no-xnnpack

run-imagenet-int8: check-ort
	$(PY) $(ORT_DIR)/mobilenetv4_imagenet_int8_bench.py --num-threads 1

run-imagenet-int8-ref: check-ort
	$(PY) $(ORT_DIR)/mobilenetv4_imagenet_int8_bench.py --num-threads 1 --no-xnnpack
