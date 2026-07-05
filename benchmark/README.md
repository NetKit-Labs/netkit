# netkit vs TFLM MNIST benchmarks

Side-by-side invoke latency on the same 10 MNIST test vectors per model (from TCAS cases in `models/mnist_*.nk`): **one correctly-classified test image per digit 0–9**, sorted by label.

## Methodology

Each benchmark:

1. Runs **100 outer iterations**
2. Each iteration invokes all **10 test images**
3. **Discards the first invoke** (image 0) as warmup
4. Records the average of images 1–9 for that run
5. Reports **mean** invoke latency across the 100 run averages

Only `MLPNetwork::forward()` / `CNNNetwork::forward()` / `MicroInterpreter::Invoke()` is timed.

Profile builds use `forward_timed()` (NETKIT) or TFLM `MicroProfiler` (TFLM) to accumulate per-op means over the same 100×10 schedule.

## Compiler flags (fair comparison)

The netkit benchmark builds its own `libnetkit_bench.a` with the **same host toolchain and flags as TFLM** (`benchmark/common/tflm_host_flags.mk`):

| Setting | Value |
|---------|--------|
| Compiler | `g++` / `gcc` |
| Core / harness | `-Os` + TFLM `CXXFLAGS` |
| Runtime kernels | `-O2` + TFLM `CXXFLAGS` |
| Link | `-lm` (osx has no `--gc-sections`) |
| Exceptions / RTTI | `-fno-exceptions -fno-rtti` |

The only intentional deviation is `-std=c++20` instead of TFLM's `-std=c++17`, because netkit's runtime API uses `std::span`.

The main repo `libnetkit.a` (clang, debug) is **not** used by these benchmarks.

## Run comparison (recommended)

```bash
make export-mnist export-mnist-cnn   # if models not present
make -C benchmark/tflm export-assets # once, or after retraining
./tools/fetch_cmsis_dsp.sh           # once, for CMSIS-DSP netkit variant

./benchmark/compare.sh
```

`compare.sh` builds and runs **10 variants**, prints ASCII comparison tables, and writes PNG exports:

| Output | Contents |
|--------|----------|
| `benchmark/mnist_latency_comparison.png` | MLP + CNN mean invoke (NETKIT ref/CMSIS vs TFLM) |
| `benchmark/mnist_mlp_profile_comparison.png` | MLP per-op profile (NETKIT vs TFLM) |
| `benchmark/mnist_cnn_profile_comparison.png` | CNN per-op profile (NETKIT vs TFLM) |

Variants executed (in order):

1. NETKIT MLP reference
2. NETKIT CNN reference
3. NETKIT MLP CMSIS-DSP
4. NETKIT CNN CMSIS-DSP
5. TFLM MLP
6. TFLM CNN
7. NETKIT MLP profile
8. NETKIT CNN profile
9. TFLM MLP profile
10. TFLM CNN profile

Speedup in tables is **TFLM mean ÷ NETKIT mean** (e.g. `5× faster` = NETKIT is five times faster).

## Run individually

```bash
make -C benchmark/netkit run              # netkit reference MLP + CNN
make -C benchmark/netkit run-cmsis        # netkit CMSIS-DSP MLP + CNN
make -C benchmark/netkit run-aot          # optional: lowered AOT (not in compare.sh)
make -C benchmark/tflm run                # TFLM MLP + CNN (needs GNU make)
make -C benchmark/netkit run-mlp-profile  # NETKIT MLP per-layer profile
make -C benchmark/netkit run-cnn-profile  # NETKIT CNN per-layer profile
make -C benchmark/tflm run-mlp-profile    # TFLM MLP per-op MicroProfiler breakdown
make -C benchmark/tflm run-cnn-profile    # TFLM CNN per-op MicroProfiler breakdown
```

Or per model/backend:

```bash
make -C benchmark/netkit run-mlp
make -C benchmark/netkit run-mlp-cmsis
make -C benchmark/tflm run-mlp
make -C benchmark/netkit run-cnn
make -C benchmark/netkit run-cnn-cmsis
make -C benchmark/tflm run-cnn
```

## Models

| Model | netkit file | Architecture |
|-------|-------------|--------------|
| MLP | `models/mnist_mlp.nk` | 784 → 128 ReLU → 10 softmax |
| CNN | `models/mnist_cnn.nk` | Conv32/Pool/Conv64/Pool/Flatten/Dense128/Dense10 |

Shared test vectors: `benchmark/tflm/generated/mnist_*_test_images.{h,cc}`

Each binary prints machine-readable summary lines:

```
BENCHMARK_SUMMARY runtime=netkit model=mlp backend=reference mean_us=... runs=100
BENCHMARK_SUMMARY runtime=netkit model=mlp backend=cmsis-dsp mean_us=... runs=100
BENCHMARK_SUMMARY runtime=tflm model=mlp backend=reference mean_us=... runs=100
PROFILE_SUMMARY runtime=netkit model=cnn kind=op tag=Conv2D mean_us=... pct=...
PROFILE_SUMMARY runtime=tflm model=cnn kind=op tag=Conv2D mean_us=... pct=...
```

Re-render PNG tables from a saved log:

```bash
python3 benchmark/tools/render_benchmark_tables.py --log /path/to/compare.log --out-dir benchmark
```

## Interpreting results (host vs MCU)

These benchmarks run on the **host desktop** (macOS/Linux), not on Cortex-M firmware.

| Label in compare output | What it actually is on host |
|-------------------------|----------------------------|
| NETKIT (without CMSIS-DSP) | Reference kernels only |
| NETKIT (with CMSIS-DSP) | CMSIS-DSP vector ops + reference layer ops |
| TFLM reference | TFLM reference kernels (not CMSIS-NN on desktop) |

**CMSIS-NN is only enabled on MCU + Cortex-M builds** (`NETKIT_CMSIS_NN_ALLOWED`). On the host, TFLM does not use CMSIS-NN either — both stacks exercise portable reference conv paths. The large CNN gap on Apple Silicon is therefore **not** a direct preview of Cortex-M4F ratios: on MCU, TFLM typically links CMSIS-NN while netkit can use CMSIS-NN for conv/pool/FC when the case is supported.

Use these numbers for **relative regression tracking on the same machine** and for **per-op breakdown** (where Conv2D dominates the CNN gap on host). For firmware SLA, re-run on the target board or an cycle-accurate simulator with the intended `NETKIT_TARGET` and CMSIS flags.

See also [docs/KERNELS.md](../docs/KERNELS.md) for reference conv optimizations (HWIO repack, input-stationary, im2col) that apply when CMSIS-NN is unavailable or falls back.

## Layout

```
benchmark/
  compare.sh                 # full comparison + PNG export
  tools/
    render_benchmark_tables.py
    export_aot_assets.py
  common/                    # shared stats + profile headers + host flags
  netkit/                    # netkit bench Makefile + mains
  tflm/                      # TFLM wrapper (clone via tools/fetch_tflm.sh)
  mnist_*_comparison.png     # generated by compare.sh (committed as samples)
```

TFLM setup: [tflm/README.md](tflm/README.md).
