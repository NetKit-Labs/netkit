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

**Models**

| key | What it is |
|-----|------------|
| `cnn` | **MNIST CNN** — digit classifier |
| `cnn_dw` | **MNIST DS-CNN** — depthwise-separable digit peer |
| `imagenet` | **MobileNetV4-Conv-Small** on ImageNet (10-class fixture) |

- Prebuild netkit binaries (untimed); discard first process per timed slot; order swaps (nk→TF, TF→nk)
- LiteRT-matched `-O3` flags; `NETKIT_IM2COL=0`
- **Latency** metric: MNIST CNN/DS-CNN `mean_us` (discard run 0 + image 0 each run); MobileNetV4-Small ImageNet `warm_mean_us` (discard full first image pass)
- **Flash / RAM**: MCU-style **runtime image only** — netkit bench ELF `__TEXT`/`__DATA` minus hard-coded test-image `.o` fixtures; TF Lite = core LiteRT CPU libs the same way. **Models and bench fixture images are excluded** (production would not embed those test vectors).
- Ratio column is always **TF ÷ netkit** (>1 ⇒ netkit faster / smaller)
- Modes: **XNNPACK ON** (both sides) and **XNNPACK OFF** (both reference). No MLP; no TF builtin-NEON-only peer.

```bash
python3 benchmark/tools/run_host_ab_suite_int8.py
python3 benchmark/tools/run_host_ab_suite_float32.py
```

Results: `benchmark/host_ab_suite_results_{int8,float32}.txt`, summary PDF `benchmark/host_ab_suite_results.pdf`.

### Preliminary results (host Apple Silicon, Jul 2026)

Flash/RAM = **runtime image only** (`size` TEXT≈flash, DATA≈static RAM). Models and hard-coded bench fixture images excluded.

**Absolute runtime sizes (same LiteRT libs for all models):**

| mode | netkit flash | netkit RAM | TF Lite flash | TF Lite RAM |
|------|--------------|------------|---------------|-------------|
| XNNPACK ON | 1.31–1.32 MiB | 191.8 KiB | 12.41 MiB | 752.0 KiB |
| XNNPACK OFF | 193–200 KiB | 15.8 KiB | 12.41 MiB | 752.0 KiB |

#### INT8 — latency / flash / ram (TF÷netkit)

| model | XNNPACK | latency | flash | ram |
|-------|---------|---------|-------|-----|
| MNIST CNN | ON | 1.02× | 9.4× | 3.9× |
| MNIST DS-CNN | ON | 1.03× | 9.4× | 3.9× |
| MNv4-Small ImageNet | ON | 1.05× | 9.4× | 3.9× |
| MNIST CNN | OFF | 3.78× | 63.5× | 47.7× |
| MNIST DS-CNN | OFF | 2.17× | 63.5× | 47.7× |
| MNv4-Small ImageNet | OFF | 3.78× | 65.6× | 47.7× |

#### FLOAT32 — latency / flash / ram (TF÷netkit)

| model | XNNPACK | latency | flash | ram |
|-------|---------|---------|-------|-----|
| MNIST CNN | ON | 1.03× | 9.4× | 3.9× |
| MNIST DS-CNN | ON | 1.09× | 9.4× | 3.9× |
| MNv4-Small ImageNet | ON | 1.08× | 9.4× | 3.9× |
| MNIST CNN | OFF | 1.99× | 65.8× | 47.7× |
| MNIST DS-CNN | OFF | 1.63× | 65.8× | 47.7× |
| MNv4-Small ImageNet | OFF | 1.88× | 63.6× | 47.7× |

**Takeaways:** With XNNPACK ON, netkit is slightly ahead on every model (float and int8; TF÷nk ≈ 1.02–1.09×). With XNNPACK OFF (TF `BUILTIN_REF` vs netkit reference), netkit is clearly ahead. **Runtime flash/RAM favor netkit** — ~1.3 MiB TEXT (XNN) or ~194–200 KiB (reference) vs ~12.4 MiB LiteRT CPU libs. Absolute MobileNetV4-Small ImageNet warm means: float32 ~1.09 ms (netkit XNN) vs ~1.17 ms (TF); int8 ~0.68 ms vs ~0.71 ms.

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
