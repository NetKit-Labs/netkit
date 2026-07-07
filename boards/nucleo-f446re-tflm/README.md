# NUCLEO-F446RE — TFLM MNIST MLP benchmark firmware

Bare-metal **TensorFlow Lite Micro** firmware for the same board and benchmark as
`boards/nucleo-f446re/` (netkit).

## Benchmark parity

| Item | Value |
|------|--------|
| Model | `benchmark/tflm/generated/mnist_mlp.tflite` (784→128→10, same weights as `models/mnist_mlp.nk`) |
| Images | 10 embedded digits from `mnist_test_images.*` (shared with netkit/TFLM host) |
| Runs | 100 outer × 10 images; discard first invoke each run |
| Metric | Mean of per-run averages (images 1–9), DWT µs @ 180 MHz |
| UART | USART2 @ 115200 (ST-Link VCP) |

## Build

```bash
# Prerequisites (once)
make -C ../../benchmark/tflm fetch-tflm export-assets
../nucleo-f446re/scripts/setup-toolchain.sh

cd boards/nucleo-f446re-tflm
make
./scripts/flash.sh   # or: make flash
./scripts/monitor.sh # press RESET on board
```

## Resource notes (STM32F446RE: 512 KiB flash / 128 KiB SRAM)

- TFLite model: ~399 KiB flash (`.rodata`)
- Tensor arena: **64 KiB** SRAM (`kTensorArenaSize` in `src/main.cc`) — smaller than the host TFLM default (128 KiB) to fit alongside stack/BSS
- TFLM `libtensorflow-microlite.a`: reference FC + Softmax kernels only at link time (full archive linked; unused ops GC'd via `-Wl,--gc-sections` where possible)

If `AllocateTensors` fails at boot, increase `kTensorArenaSize` or reduce stack in the linker script.

## Compare with netkit

```text
BENCHMARK_SUMMARY runtime=tflm model=mlp backend=reference mean_us=... runs=100
```

vs netkit:

```text
BENCHMARK_SUMMARY runtime=netkit model=mlp backend=cmsis-dsp mean_us=... runs=100
```
