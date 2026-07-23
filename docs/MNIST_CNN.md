# MNIST CNN Regression Test

Trained **784→CNN→10** classifier on MNIST using a stack common in Keras/TensorFlow tutorials. Runs as part of `make test` alongside the [MNIST MLP suite](MNIST.md).

## Architecture

| Stage | Config | Activation |
|-------|--------|------------|
| Input | 28×28×1 (NHWC) | — |
| Conv2D | 3×3, 32 filters, stride 1 | ReLU |
| MaxPool2D | 2×2, stride 2 | — |
| Conv2D | 3×3, 64 filters, stride 1 | ReLU |
| MaxPool2D | 2×2, stride 2 | — |
| Flatten | 5×5×64 → 1600 | — |
| Dense | 128 units | ReLU |
| Dense | 10 units | Softmax |

## Accuracy

| Metric | netkit (committed export) | Typical baseline |
|--------|---------------------------|------------------|
| Test accuracy | **99.02%** | ~98.5–99.3% (Keras/TF CNN tutorials) |
| Training set | **60,000** images | 60,000 |
| Optimizer | Adam, lr=0.001, **20 epochs** | Similar |

The [MNIST MLP suite](MNIST.md) on the same data reaches **98.06%** test accuracy.

## Files

| Path | Purpose |
|------|---------|
| `models/mnist_cnn.onnx` | Source graph (ONNX parity) |
| `models/mnist_cnn.nk` | Float32 runtime model + 10 embedded TCAS cases |
| `models/mnist_cnn_int8.nk` | Int8 quantized model (MCU / CMSIS-NN) |
| `tools/export_mnist_cnn.py` | Train + export float script |
| `tools/export_mnist_cnn_int8.py` | Quantize float `.nk` to int8 (optional TFLite input-quant alignment) |
| `benchmark/tflm/tools/export_int8_test_images.py` | Preqantized int8 benchmark test vectors (separate from float images) |
| `tools/compare_nk_tflite_quant.py` | Compare activation quant params vs TFLite |

The MNIST CNN suite uses the default heap arena (`Arena::kDefaultCapacity`, **64 MiB** on CPU) in `src/nk_regression.cpp`. See [ARENA.md](ARENA.md).

## Running

Part of `make test` / `./netkit test` — see [TESTING.md](TESTING.md).

## Benchmarks

Same 10 TCAS vectors as the MLP suite; compared against TFLM in [benchmark/README.md](../benchmark/README.md). CNN invoke and per-op profile tables (Conv2D vs pool vs FC) are produced by `./benchmark/compare.sh`. On host builds both runtimes use reference conv kernels in that script — CMSIS-NN applies only on Arm MCU targets.

**Host XNNPACK peers** (LiteRT-matched flags): `make -C benchmark/netkit run-cnn-xnnpack` / `run-cnn-int8-xnnpack` vs `make -C benchmark/tflite run-cnn` / `run-cnn-int8`. Recent numbers: [STATUS.md](STATUS.md).

### Depthwise-separable peer (host)

Separable tutorial topology (PW→DW→Pool ×2 → Dense): `models/mnist_cnn_dw.nk` / `_int8.nk`.

```bash
python3 tools/export_mnist_cnn_dw.py
python3 tools/export_mnist_cnn_dw_int8.py
python3 tools/export_mnist_cnn_dw_assets.py
make -C benchmark/netkit run-cnn-dw-xnnpack
make -C benchmark/netkit run-cnn-dw-int8-xnnpack
make -C benchmark/tflite run-cnn-dw
make -C benchmark/tflite run-cnn-dw-int8
```

MCU peer A/B (`NETKIT_EMBED=1`, matched toolchain) — [STATUS.md](STATUS.md):

| Backend | netkit | TFLM | microTVM |
|---------|-------:|-----:|---------:|
| CMSIS-NN | **58.3 ms** | 61.4 ms | 86.4 ms |
| reference | **140.3 ms** | 826.8 ms | 236.0 ms |

### Int8 on-device (NUCLEO-F446RE)

```bash
make export-mnist-cnn-int8
python3 benchmark/tflm/tools/export_int8_test_images.py
make -C boards/nucleo-f446re-cnn-int8
cd boards/nucleo-f446re-cnn-int8 && ./scripts/flash.sh && ./scripts/monitor.sh
```

Verified peer A/B on NUCLEO-F446RE @ 180 MHz (matched −O2/−flto toolchain; `NETKIT_EMBED=1`; 10×10, discard first invoke) — full tables in [STATUS.md](STATUS.md) and [`benchmark/mcu_ab_logs/`](../benchmark/mcu_ab_logs/):

| Backend | netkit | TFLM | microTVM |
|---------|-------:|-----:|---------:|
| CMSIS-NN | **95.3 ms** | 95.5 ms | 112.3 ms |
| reference | **336.2 ms** | 2593.5 ms | 343.0 ms |

All **10/10**. Board default is **quant lowered** (`make` / `make deploy-lowered`); use `NETKIT_EMBED=1` for the TFLM-fair interpreter numbers above. Reference: `NETKIT_REFERENCE_QUANT_LOOPS=1`. Peers: [cnn-int8](../boards/nucleo-f446re-cnn-int8/README.md), [tflm-cnn-int8](../boards/nucleo-f446re-tflm-cnn-int8/README.md), [tvm-cnn-int8](../boards/nucleo-f446re-tvm-cnn-int8/README.md).

### Int8 on-device (Seeed XIAO ESP32C3)

Peer A/B vs TFLM @ 160 MHz (matched `-O3` C++ flags; **interpreter embed**; 10×10; order swaps) — [STATUS.md](STATUS.md#mcu-seeed-xiao-esp32c3):

**ESP-NN on** — [`esp32c3_int8_ab_results.txt`](../benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ab_results.txt)

| Model | netkit | TFLM |
|-------|-------:|-----:|
| MNIST CNN | 252.0 ms | **251.4 ms** |
| MNIST DS-CNN | 87.7 ms | **87.5 ms** |

**ESP-NN off (reference)** — [`esp32c3_int8_ref_ab_results.txt`](../benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ref_ab_results.txt)

| Model | netkit | TFLM |
|-------|-------:|-----:|
| MNIST CNN | **226.8 ms** | 1205.5 ms |
| MNIST DS-CNN | **85.8 ms** | 392.3 ms |

All **10/10**. Arena: CNN **64 KiB** (`NK_ARENA_DEFAULT_CAPACITY`); DS-CNN **96 KiB**. Flash/RAM (ESP-NN embeds): CNN ~420 KiB / 75 KiB; DS-CNN ~405 KiB / 107 KiB. Index: [boards/xiao-esp32c3/](../boards/xiao-esp32c3/README.md). Runners: `run_esp_int8_ab.sh` / `run_esp_int8_ref_ab.sh`.

Quant lowered AOT is expected to be faster than embed but measured a hair slower under ESP-NN on this board — investigate before promoting it ([STATUS.md](STATUS.md#mcu-seeed-xiao-esp32c3)).

UART captures `DIGIT_SUMMARY` lines with raw int8 softmax (`pred_i8`, `out_i8=...`). Dequantized per-digit confidence is computed offline — not on device:

```bash
python3 benchmark/tools/parse_mcu_cnn_int8_log.py uart.log
python3 benchmark/tools/parse_mcu_cnn_int8_log.py --compare netkit_uart.log tflm_uart.log
```

Int8 export aligns **layer-0 input quant** with TFLite when `benchmark/tflm/generated/mnist_cnn_int8.tflite` is present (`input_scale=1/255`, `zero_point=-128`). Weight and per-layer output scales are calibrated from netkit float weights. Benchmark and TCAS inputs are **prequantized int8 in Python** — no float→int8 conversion in C++ (MCU firmware or host `libnetkit`).

## Regenerating

```bash
make export-mnist-cnn
make export-mnist-cnn-int8   # optional: int8 quant from float .nk
make export-onnx-test
make test
```

Commit `models/mnist_cnn.nk`, `models/mnist_cnn.onnx`, and `models/mnist_cnn_int8.nk` after regenerating so tests stay offline (no training at test time).
