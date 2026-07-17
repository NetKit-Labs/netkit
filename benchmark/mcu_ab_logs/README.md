# MCU A/B logs (NUCLEO-F446RE)

Canonical **netkit vs TFLM vs microTVM** int8 results for MNIST CNN and DS-CNN.

| Artifact | Contents |
|----------|----------|
| **[mcu_int8_ab_results.txt](mcu_int8_ab_results.txt)** | Latency + flash/RAM tables (source of truth) |
| `*_cmsis.log` / `*_ref.log` | UART captures per runtime / model / backend |
| [docs/STATUS.md](../../docs/STATUS.md#mcu-nucleo-f446re) | Published summary |

## Latency summary (10×10, discard first invoke; all 10/10)

**CMSIS-NN**

| Model | netkit embed | TFLM | microTVM AOT |
|-------|-------------:|-----:|-------------:|
| MNIST CNN | **95.3 ms** | 95.5 ms | 112.3 ms |
| MNIST DS-CNN | **58.3 ms** | 61.4 ms | 86.4 ms |

**Reference (CMSIS-NN off)**

| Model | netkit embed | TFLM | microTVM C AOT |
|-------|-------------:|-----:|---------------:|
| MNIST CNN | **336.2 ms** | 2593.5 ms | 343.0 ms |
| MNIST DS-CNN | **140.3 ms** | 826.8 ms | 236.0 ms |

Board: STM32F446RE @ 180 MHz. Matched toolchain (`mcu_tflm_toolchain.mk`). No XNNPACK on MCU.
