# Platform and dtype status

Snapshot of what works today, what was measured, and what is still open. Companion to [PHILOSOPHY.md](PHILOSOPHY.md), [BUILD_TARGETS.md](BUILD_TARGETS.md), and [DATATYPES.md](DATATYPES.md).

## Dtypes

| Dtype | Status | Notes |
|-------|--------|-------|
| **float32** | **Complete** | Default path on cpu / MCU / MPU; ImageNet MobileNetV4, MNIST CNN/MLP, fused blocks |
| **int8** | **Complete** | End-to-end int8 I/O (no C++ float↔int8); MNIST CNN/MLP MCU (CMSIS-NN); host/MPU via XNNPACK qs8 or QuantOps reference; ImageNet MNv4 int8 |
| float16 / int16 / int4 | Planned (Phase 2) | See [DATATYPES.md](DATATYPES.md) |

## Target maturity

| Target | Role | Kernels | Maturity |
|--------|------|---------|----------|
| **cpu** | Desktop / CI / peer benches | XNNPACK (default); reference fallback | **Done** — float32 + int8 |
| **mcu_arm** | Arm Cortex-M firmware | CMSIS-NN (int8 production); float32 via reference; XNNPACK **forbidden** | **Done** — float32 + int8 (NUCLEO-F446RE) |
| **mpu_arm** | Arm Cortex-A / RTOS-class | XNNPACK (default); CMSIS-NN off | **Done** — float32 + int8 |
| **mpu_risc** | RISC-V MPU | XNNPACK (default); CMSIS-NN **forbidden** | **Mostly done** — same portable XNNPACK + generic path as other MPUs |
| **mcu_risc** | RISC-V MCU | Generic / reference kernels only; CMSIS + XNNPACK **forbidden** | **Works today** — no RISC-specific optimized kernels yet; **generic fallbacks are fast** and suitable until ISA-tuned kernels land |

**Policy reminder:** XNNPACK is default on cpu and all MPUs, never on MCU. CMSIS-NN is Arm MCU only (production int8). **CMSIS-DSP is not used.** Float32 on MCU uses reference kernels only (no optimized float32 MCU plan). `NETKIT_IM2COL` defaults to **0** on all targets (see [BUILD_TARGETS.md](BUILD_TARGETS.md#netkit_im2col-guidance)).

## Host file mmap

| Host OS | Status | Implementation |
|---------|--------|----------------|
| **macOS / Linux** | **Complete** | POSIX `mmap` (`MAP_PRIVATE`) |
| **Windows** | **Complete** | Win32 `CreateFileMapping` / `MapViewOfFile` (`FILE_MAP_COPY`) |
| **MCU** | **Forbidden** | Use `Load*FromBuffer` / flash; `NETKIT_MMAP=1` is forced off |

Default **on** for cpu + any MPU; opt out with `NETKIT_MMAP=0` on RTOS / bare-metal MPU. See [ARENA.md](ARENA.md) and [BUILD_TARGETS.md](BUILD_TARGETS.md).

## Host A/B suite (preliminary)

Fair CPU peer suite vs TF Lite / LiteRT (`benchmark/tools/run_host_ab_suite_{int8,float32}.py`):

- Prebuild netkit binaries (untimed); discard first process per timed slot; order swaps (nk→TF, TF→nk)
- LiteRT-matched `-O3` flags; `NETKIT_IM2COL=0`; MLP `--runs 100` both sides
- **Latency** metric: MNIST `mean_us`; ImageNet `warm_mean_us`
- **Flash**: on-disk deploy footprint (netkit ELF + `.nk`; TF Lite `.tflite` + core LiteRT CPU libs)
- **RAM**: peak RSS of the kept timed process
- Ratio column is always **TF ÷ netkit** (>1 ⇒ netkit faster / smaller)

```bash
python3 benchmark/tools/run_host_ab_suite_int8.py
python3 benchmark/tools/run_host_ab_suite_float32.py
```

Results files: `benchmark/host_ab_suite_results_{int8,float32}.txt`.

### Preliminary results (host Apple Silicon, Jul 2026)

#### INT8 — latency / flash / ram (TF÷netkit)

| model | XNNPACK | latency | flash | ram |
|-------|---------|---------|-------|-----|
| mlp | ON | 1.37× | 9.4× | 16.2× |
| cnn | ON | 0.72× | 8.9× | 13.0× |
| cnn_dw | ON | 0.97× | 9.0× | 13.2× |
| imagenet | ON | 1.00× | 3.1× | 4.6× |
| mlp | OFF | 1.38× | 45.7× | 21.7× |
| cnn | OFF | 4.30× | 35.1× | 20.9× |
| cnn_dw | OFF | 2.31× | 36.1× | 20.5× |
| imagenet | OFF | 3.70× | 4.0× | 7.0× |

#### FLOAT32 — latency / flash / ram (TF÷netkit)

| model | XNNPACK | latency | flash | ram |
|-------|---------|---------|-------|-----|
| mlp | ON | 1.18× | 8.1× | 13.2× |
| cnn | ON | 0.97× | 6.9× | 9.4× |
| cnn_dw | ON | 1.00× | 7.0× | 9.8× |
| imagenet | ON | 0.97× | 1.4× | 1.8× |
| mlp | OFF | 2.24× | 25.0× | 19.0× |
| cnn | OFF | 2.40× | 15.5× | 14.9× |
| cnn_dw | OFF | 1.47× | 16.2× | 14.9× |
| imagenet | OFF | 1.92× | 1.5× | 2.1× |

**Takeaways:** with XNNPACK ON, latency is roughly tied (ImageNet ~parity; MNIST within noise). With XNNPACK OFF, netkit reference is clearly ahead on latency. **Flash and RAM favor netkit substantially** on every model (self-contained ELF vs LiteRT shared libs + Python process RSS). Absolute ImageNet float32 warm means were ~5.1 ms (netkit XNN) vs ~4.9 ms (TF); int8 ~1.6 ms both sides.

### `NETKIT_IM2COL` note (from earlier host sweep)

With XNNPACK ON, im2col does not move the needle (accelerated path ignores it). With XNNPACK OFF, **`NETKIT_IM2COL=1` (partial)** can give a **small** float CNN reference bump on MPU/cpu; **`2` (full)** was not a clear win. **Default and recommendation: leave `NETKIT_IM2COL=0`.** At most try `1` on MCU or reference-only MPU builds.

### MCU (NUCLEO-F446RE)

| Board | Result |
|-------|--------|
| MNIST CNN int8 (CMSIS-NN) | 10/10 @ ~95 ms (10×10 methodology) |
| MNIST MLP int8 (CMSIS-NN) | 10/10 @ ~3.4 ms |
| XNNPACK in MCU ELF | **None** — `nm` shows no `xnn*` / XNNPACK symbols on nucleo CNN int8 firmware |

## What “done” means here

- **Arm MCU / MPU:** production-oriented paths exist (CMSIS-NN on MCU; XNNPACK on MPU/cpu) with float32 and int8 models, benches, and docs.
- **RISC MPU:** uses the same XNNPACK LayerFast stack as other MPUs; CMSIS is correctly unavailable.
- **RISC MCU:** builds and runs on **generic reference kernels** only. Those kernels are the portable fallback used everywhere else when accelerators are off — they are already competitive on CPU “OFF” peers. Dedicated RISC-V vector / DSP microkernels are **not** implemented yet.

## Open / next

- RISC-V MCU-optimized kernels (optional; generic path remains the default)
- Broader int8 model coverage beyond MNIST + ImageNet MNv4 fixtures
- float16 / int16 / int4 (Phase 2)
- Voice modality fixtures; Kalman estimation (Phase 3)
