# NUCLEO-F446RE — TFLM MNIST DS-CNN int8 benchmark firmware

Bare-metal **TensorFlow Lite Micro** firmware with a **full int8** MNIST DS-CNN graph
(int8 input/output, CMSIS-NN optimized kernels; `TFLM_OPT_KERNEL=` for reference).

## Benchmark parity

| Item | Value |
|------|--------|
| Model | `benchmark/tflm/generated/mnist_cnn_dw_int8.tflite` |
| Embed | `mnist_cnn_dw_int8_model_data.{h,cc}` (`make export-cnn-dw-int8-model-data`) |
| Images | `benchmark/tflm/generated/cnn_dw/mnist_cnn_int8_test_images.*` |
| Runs | **10** outer × 10 images; discard first invoke each run |
| Arena | **116 KiB** |
| UART | USART2 @ 115200 (ST-Link VCP) |

## Build

```bash
cd boards/nucleo-f446re-tflm-cnn-dw-int8
make export-cnn-dw-int8-model-data
make setup-deps
make                          # CMSIS-NN
make TFLM_OPT_KERNEL= clean all   # reference
./scripts/flash.sh
./scripts/monitor.sh
```

```text
BENCHMARK_SUMMARY runtime=tflm model=cnn_dw_int8 backend=cmsis-nn mean_us=... runs=10
```

Compare with netkit: [nucleo-f446re-cnn-dw-int8](../nucleo-f446re-cnn-dw-int8/README.md).
