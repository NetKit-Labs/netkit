# TVM / microTVM

Host/CPU TVM peer benches are **not maintained**. Stock TVM’s LLVM CPU path
is not a useful comparison to netkit or TF Lite / LiteRT with XNNPACK.

**Use TVM here for MCU / microTVM peers only.**

## MCU results (NUCLEO-F446RE, int8)

Matched toolchain; 10×10, discard first invoke; all **10/10**. Full tables:
[mcu_int8_ab_results.txt](../mcu_ab_logs/mcu_int8_ab_results.txt),
[docs/STATUS.md](../../docs/STATUS.md#mcu-nucleo-f446re).

**CMSIS-NN**

| Model | netkit embed | TFLM | microTVM AOT |
|-------|-------------:|-----:|-------------:|
| MNIST CNN | **95.3 ms** | 95.5 ms | **112.3 ms** |
| MNIST DS-CNN | **58.3 ms** | 61.4 ms | **86.4 ms** |

**Reference (CMSIS-NN off)**

| Model | netkit embed | TFLM | microTVM C AOT |
|-------|-------------:|-----:|---------------:|
| MNIST CNN | **336.2 ms** | 2593.5 ms | **343.0 ms** |
| MNIST DS-CNN | **140.3 ms** | 826.8 ms | **236.0 ms** |

Boards: [`nucleo-f446re-tvm-cnn-int8`](../../boards/nucleo-f446re-tvm-cnn-int8/),
[`nucleo-f446re-tvm-cnn-dw-int8`](../../boards/nucleo-f446re-tvm-cnn-dw-int8/).

Host CPU peers: **netkit vs TF Lite** via
`python3 benchmark/tools/run_host_ab_suite_{float32,int8}.py`.
