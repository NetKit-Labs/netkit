# Seeed XIAO ESP32C3 — shared board support

Common ESP-IDF / PlatformIO pieces for XIAO ESP32C3 firmwares under
`boards/xiao-esp32c3-*`.

| Item | Value |
|------|--------|
| Chip | ESP32-C3 · RISC-V · 160 MHz |
| netkit profile | `mcu_esp` + `NETKIT_ARCH=ESP32C3` + **ESP-NN** |
| Console | USB Serial/JTAG |
| Peer deploy | **Interpreter embed** (`NETKIT_LOWERED=0`) |
| Arena | **64 KiB** default (`NK_ARENA_DEFAULT_CAPACITY`); DS-CNN **96 KiB** |

**Target ≠ ISA:** this is Espressif RISC-V → still `mcu_esp`, not `mcu_risc`.
See [PLATFORMS.md — Target ≠ CPU ISA](../../docs/PLATFORMS.md#target--cpu-isa).

## Firmwares

| Directory | Runtime | Model |
|-----------|---------|-------|
| [`../xiao-esp32c3-mlp-int8/`](../xiao-esp32c3-mlp-int8/README.md) | netkit | MNIST MLP int8 |
| [`../xiao-esp32c3-cnn-int8/`](../xiao-esp32c3-cnn-int8/README.md) | netkit | MNIST CNN int8 |
| [`../xiao-esp32c3-cnn-dw-int8/`](../xiao-esp32c3-cnn-dw-int8/README.md) | netkit | MNIST DS-CNN int8 |
| [`../xiao-esp32c3-tflm-cnn-int8/`](../xiao-esp32c3-tflm-cnn-int8/README.md) | TFLM (ESP-NN) | MNIST CNN int8 |
| [`../xiao-esp32c3-tflm-cnn-dw-int8/`](../xiao-esp32c3-tflm-cnn-dw-int8/README.md) | TFLM (ESP-NN) | MNIST DS-CNN int8 |

## Int8 peer A/B (published)

Methodology: **10×10**, discard first invoke; order swaps `nk→tflm` / `tflm→nk`. All **10/10**.
Canonical: [STATUS.md](../../docs/STATUS.md#mcu-seeed-xiao-esp32c3).

**ESP-NN on** — [`scripts/run_esp_int8_ab.sh`](scripts/run_esp_int8_ab.sh) · [`esp32c3_int8_ab_results.txt`](../../benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ab_results.txt)

| Model | netkit | TFLM |
|-------|-------:|-----:|
| MNIST CNN | 252.0 ms | **251.4 ms** |
| MNIST DS-CNN | 87.7 ms | **87.5 ms** |

**ESP-NN off (reference)** — [`scripts/run_esp_int8_ref_ab.sh`](scripts/run_esp_int8_ref_ab.sh) · [`esp32c3_int8_ref_ab_results.txt`](../../benchmark/mcu_ab_logs/xiao_esp32c3/esp32c3_int8_ref_ab_results.txt) · `PIO_ENV=xiao_esp32c3_ref`

| Model | netkit | TFLM |
|-------|-------:|-----:|
| MNIST CNN | **226.8 ms** | 1205.5 ms |
| MNIST DS-CNN | **85.8 ms** | 392.3 ms |

Quant lowered AOT should beat embed but measured a hair slower under ESP-NN — under investigation; peer default stays embed.

**Compiler match:** netkit C++ uses the same speed flags as `esp-tflite-micro`
(`-O3`, `-fno-rtti`, `-fno-exceptions`, `-fno-threadsafe-statics`,
`-fno-unwind-tables`) via [`mcu_esp_tflm_match_compile.cmake`](mcu_esp_tflm_match_compile.cmake).
ESP-NN C stays at IDF `-O2` on both sides. IDF global: `CONFIG_COMPILER_OPTIMIZATION_PERF`.

**ImageNet / MobileNetV4:** not run on this part — int8 weights alone are ~2.5–3.8 MiB,
above the default 1 MiB factory app partition (and far above NUCLEO’s 512 KiB flash).
