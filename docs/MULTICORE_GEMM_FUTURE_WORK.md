# Future Work: Multicore GEMM on ESP32-S3 and ESP32-P4

> **Status: PROPOSED / NOT IMPLEMENTED / NOT BENCHMARKED.**
> Design note recorded October 29, 2026. Nothing in this document is a measured NetKit multicore GEMM result or a claim that current firmware already enables multicore ESP-NN kernels.
> Current released NetKit inference paths remain as described in [STATUS.md](STATUS.md) and [DATATYPES.md](DATATYPES.md).

## Objective and scope

Evaluate **dual-core generalized matrix multiplication (GEMM)** for NetKit's Espressif MCU target on:

- **ESP32-S3**: two Xtensa LX7 CPU cores, 240 MHz in existing NetKit tests.
- **ESP32-P4**: two high-performance RISC-V CPU cores (360 MHz in existing NetKit tests); the board's companion ESP32-C6 is used for networking, **not** as an inference core.
- **Data types**: **INT8, INT16, Float32**. INT8 and Float32 are established NetKit inference data types; **end-to-end INT16 inference remains planned** ([DATATYPES.md](DATATYPES.md)). An INT16 GEMM microkernel can be prototyped independently without implying that INT16 model loading, quantization, or inference are complete.

**Goal:** measure whether partitioning *one* matrix multiplication over the two high-performance cores improves *single-inference latency*. This is distinct from running two independent model instances on different cores, which primarily improves aggregate throughput.

**Priority:** a practical, reusable multicore GEMM kernel is more relevant here than a dependency-aware, multithreaded graph scheduler. Leave NetKit's graph interpreter/AOT forward pass sequential; parallelize selected expensive kernels internally.

## Existing library capabilities: do not duplicate an established feature

The following table refers to **publicly inspected Espressif ESP-NN and ESP-DSP implementations**, not to every possible third-party implementation. Recheck dependency versions and implementations before beginning work.

| Operation | ESP32-S3 (Xtensa) | ESP32-P4 (RISC-V) | Notes |
|---|---|---|---|
| INT8 convolution | Optimized single-core **and optional dual-core** ESP-NN | Optimized single-core **and optional dual-core** ESP-NN | Multicore only for supported kernel shapes/paths, not all convolutions |
| INT8 depthwise convolution | Optimized single-core **and optional dual-core** ESP-NN | Optimized single-core **and optional dual-core** ESP-NN | Same shape/backend restrictions |
| INT8 fully connected / batched fully connected | ESP-NN optimized kernels | ESP-NN optimized kernels | Matrix-vector / specialized matrix-matrix work, **not verified as multicore** |
| Float32 general matrix multiplication | ESP-DSP single-core optimized function | ESP-DSP single-core optimized function | `dspm_mult_f32` / `dspm_mult_ex_f32` |
| INT16 general matrix multiplication | ESP-DSP single-core optimized function | ESP-DSP single-core optimized function | `dspm_mult_s16`; fixed-point output scaling must be respected |
| INT8 general-purpose GEMM | No general public ESP-NN/ESP-DSP GEMM API identified | No general public ESP-NN/ESP-DSP GEMM API identified | ESP-NN's specialized batched FC may be a useful single-core comparator |
| **Multicore GEMM (INT8 / INT16 / Float32)** | **No library-provided implementation identified** | **No library-provided implementation identified** | **NetKit future-work opportunity**, not a claimed first-ever solution |

**Important TFLM distinction:** TFLM's graph interpreter ordinarily invokes operators sequentially. The invoked operator *may itself* use both CPU cores: newer ESP-NN versions expose `esp_nn_dual_core_enable()`, with supported convolution kernels splitting independent output regions between cores. This does **not** imply a fully multithreaded TFLM graph engine, nor does it make general multicore GEMM available.

**Current NetKit dependency/benchmark caveats:**

- NetKit's Espressif INT8 implementation already invokes ESP-NN optimized kernels. Baseline NetKit/TFLM peers that both use ESP-NN are near parity ([STATUS.md](STATUS.md)); the existing numbers should **not** be represented as multicore results.
- The NetKit [ESP-NN fetch script](../tools/fetch_esp_nn.sh) pins a specific upstream commit. The existing S3 board compilation assembles a specific set of ESP-NN source files; the new ESP-NN multicore runtime is **not automatically part of that build**. Audit and deliberately upgrade/integrate the backend before claiming dual-core convolution support in current NetKit firmware.
- P4 peers presently use portable ESP-NN rather than optimized PIE assembly because of a PlatformIO assembler compatibility issue: [KI-003](KNOWN_ISSUES.md#ki-003--esp32-p4-esp-nn-pie-asm-disabled-under-platformio-gas). Resolve or account for this before drawing performance conclusions about RISC-V optimized or multicore kernels.
- The Pi Zero 2 W is a **four-core Cortex-A53 Linux MPU**, not the primary target here. Its existing TF Lite/XNNPACK comparison intentionally uses **one thread** ([board setup](../boards/pi-zero-2w/README.md)). Regular TF Lite/LiteRT with XNNPACK is the more relevant peer on that board than TFLM; a multicore MPU experiment needs a separate thread-count-controlled benchmark.

### Upstream implementation evidence

- [ESP-NN multicore interface](https://github.com/espressif/esp-nn/blob/master/include/esp_nn_multicore.h) and [multicore worker implementation](https://github.com/espressif/esp-nn/blob/master/src/common/esp_nn_multicore.c): optional worker pinned to the other core, opt-in API, completion wait.
- [ESP-NN S3 convolution](https://github.com/espressif/esp-nn/blob/master/src/convolution/esp_nn_conv_esp32s3.c), [P4 RISC-V convolution](https://github.com/espressif/esp-nn/blob/master/src/convolution/esp_nn_conv_riscv_pie.c), and [P4 depthwise convolution](https://github.com/espressif/esp-nn/blob/master/src/convolution/esp_nn_depthwise_conv_riscv_pie.c): conditional multicore splitting.
- [ESP-NN S3 fully connected](https://github.com/espressif/esp-nn/blob/master/src/fully_connected/esp_nn_fully_connected_esp32s3.c) and [portable fully connected](https://github.com/espressif/esp-nn/blob/master/src/fully_connected/esp_nn_fully_connected_ansi.c): optimized/specialized FC including batched FC, but no multicore dispatch identified.
- [ESP-DSP matrix API](https://github.com/espressif/esp-dsp/blob/master/modules/matrix/mul/include/dspm_mult.h): Float32/INT16 general matrix multiplication and INT8 matrix-vector. No library-managed dual-core GEMM API identified.
- [ESP-NN README and dual-core network benchmarks](https://github.com/espressif/esp-nn/blob/master/README.md): useful **external** reference only; numbers are **not NetKit measurements**.

## Mathematical decomposition

For matrix multiplication `C = A × B` where `A` and `B` are 64×64, partition each into **four 32×32 blocks**:

```text
A = [A11 A12; A21 A22],   B = [B11 B12; B21 B22]

C11 = A11 B11 + A12 B21
C12 = A11 B12 + A12 B22
C21 = A21 B11 + A22 B21
C22 = A21 B12 + A22 B22
```

This requires **eight 32×32 block multiplications**, not four:

| Core | Destination blocks | 32×32 block multiplies | Writes |
|---|---|---|---|
| Core 0 | `C11` and `C12` (top 32 rows of `C`) | **4** | Only top half of `C` |
| Core 1 | `C21` and `C22` (bottom 32 rows of `C`) | **4** | Only bottom half of `C` |

Each core must accumulate its *own two products per output block* along the reduction (K) dimension. There is **no cross-core reduction of the final matrix** if outputs are partitioned by row, so concurrent writes do not overlap. A final completion barrier is still necessary.

An even simpler initial implementation need not form explicit 32×32 copies: pass **row ranges or output tile coordinates and original matrix strides**, letting each core compute a disjoint half of `C`. Avoid temporary whole-matrix buffers when MCU RAM is constrained.

For general `M×K` multiplied by `K×N`, split output rows (M) or independent output tiles (M,N). **Do not independently split K across cores initially**: that would require partial sums and a cross-core reduction, increasing scratch memory, synchronization, and numerical complications.

## Partitioning versus cache/memory tiling

**Parallel work assignment** answers *which output rows/tiles each core owns*. **Tiling/blocking** answers *how one core reuses A/B data efficiently within its assigned work*. They need not use the same tile size.

Proposed first experiment:

1. Keep the interpreter/AOT graph execution sequential and run GEMM as one synchronous operator.
2. Spawn or initialize a **persistent** FreeRTOS worker pinned to the second high-performance core **once**, not per multiply.
3. If the matrix is large enough, split the output rows approximately evenly; the invoking core and worker each execute an optimized GEMM kernel on separate output slices.
4. Within each slice, benchmark internal tile sizes such as **8×8, 16×16, and 32×32** (plus unblocked) for memory locality, data layout, and register pressure; use original strides when possible.
5. Wait for the worker, then return to the next graph operator. Preserve a **single-core fallback** when the model is small, the worker is unavailable, or the schedule is unfavorable.

A 64×64 × 64×64 example is a **correctness and initial timing fixture**, not an assertion that 32×32 tiling is optimal. Small matrix workloads may run **slower** with parallel dispatch. Memory bandwidth, SRAM vs PSRAM, caches, alignment, core startup/IPC, worker scratch, and chip-specific SIMD/PIE instructions affect performance.

### Numeric design considerations

- **INT8:** accumulate products in a sufficiently wide intermediate (commonly INT32 for bounded K); handle asymmetric zero-points, per-channel scales/bias, activation clamps, and final requantization exactly as required by NetKit's model semantics. Check accumulator bounds for larger K.
- **INT16:** prototype fixed-point GEMM with defined accumulator width (INT64 or proved safe narrower bounds), rounding, shifting/saturation, and test vectors. **This does not establish INT16 inference support in NetKit.** Compare equivalent numeric semantics against ESP-DSP.
- **Float32:** match numerical tolerance expectations; tile order/accumulation order can change floating-point rounding. Benchmark against ESP-DSP's optimized Float32 multiply, not only its portable ANSI path.
- Keep per-core scratch independent; ensure lifetimes are valid until the completion barrier. Avoid allocations in the hot loop.

## Proposed implementation stages (all future)

- [ ] **Audit actual versions and baselines.** Record NetKit's ESP-NN pinned SHA, TFLM peer versions, ESP-IDF / PlatformIO versions, clocks, backend selection, build flags, memory placement, and whether ESP-NN's optional dual-core convolution is actually integrated/enabled.
- [ ] **Establish single-core GEMM baseline.** On both S3 and P4, compare NetKit's suitable existing kernels with **ESP-DSP Float32/INT16** and **ESP-NN INT8 FC/batched FC** where shapes/quantization match. Avoid comparing non-equivalent operations or dtypes.
- [ ] **Prototype a shared, row-partitioned dual-core GEMM dispatcher** (two cores, persistent worker, threshold/fallback) with **Float32 and INT8** microkernels. Keep build-time feature opt-in and no change to existing model execution by default.
- [ ] **Prototype INT16 GEMM independently**; implement full INT16 runtime/packager functionality only as a separate dtype roadmap item.
- [ ] **Benchmark tile sizes and core-split strategies** on both actual boards: 8×8, 16×16, 32×32 internal tiling; half-row slabs vs independent output tiles; single-core vs two-core; SRAM vs PSRAM where supported.
- [ ] **Investigate P4 toolchain/PIE issue** before optimizing or publishing RISC-V SIMD comparisons ([KI-003](KNOWN_ISSUES.md#ki-003--esp32-p4-esp-nn-pie-asm-disabled-under-platformio-gas)).
- [ ] **Integrate only on demonstrated benefit** for chosen workload shapes; retain one-core fallback, correctness regression tests, and fair reproducible NetKit/TFLM/ESP-NN/ESP-DSP comparisons.
- [ ] **Only then consider** expanding multicore kernels to additional operators, or a dependency-aware graph scheduler if representative branched models show material unused operator-level parallelism.

**Not initially planned:** replacing ESP-NN's already-available multicore convolutions with custom ones, distributing a model's dependency graph across tasks, or assuming two independent interpreters reduce the latency of one inference.

## Suggested evaluation and success criteria

For each **processor × dtype × matrix shape × memory location**, retain raw timing and correctness logs for:

- Shapes: 16×16, 32×32, **64×64**, 128×128, 256×256 (as memory allows), plus non-square `M×K` by `K×N` and small batched FC workloads.
- Baselines: reference scalar; vendor **optimized single-core** kernel where mathematically comparable; NetKit optimized **single-core** kernel; NetKit **dual-core** variant.
- Measurements: cold/warm latency, median and spread over repeated trials, full invocation latency, optional energy/power, allocated scratch/stack and firmware size; record compiler optimization, clocks, core affinity, and RAM/PSRAM.
- Correctness: exact outputs where appropriate (INT8 requantized and INT16 rules), bounded Float32 error, randomized operands, saturation/overflow cases, odd/tail dimensions, repeated invocations, and scheduler/worker failure fallback.
- Report both **speedup = T(single core) / T(dual core)** and **latency reduction = 1 − T(dual core)/T(single core)**. A theoretical upper bound of ~2× on two equal cores is not an expected result; synchronization and shared memory can substantially lower the practical gain.

**Decision gate:** integrate only if actual target-model inference benefits enough to justify extra code size, worker stack/scratch memory, power, and maintenance. Do not promise a specific speedup before measurements.

## Why not a fully multithreaded interpreter first?

Typical TinyML MLP/CNN/DS-CNN and many transformer subgraphs have strongly sequential **inter-operator dependencies**. Parallel GEMM/conv exploits plentiful **intra-operator** work without graph scheduling. A dependency-aware graph scheduler helps mainly when a representative model has independent branches or operators worth running concurrently; on **two cores**, it is likely to have a smaller payoff and greater complexity. Multiple independent interpreters solve a **throughput/concurrent-model** problem (e.g. an edge gateway), not the same **single-input latency** objective.

## Cross-references

- [STATUS.md](STATUS.md) — shipped kernels, existing *single-inference* MCU/MPU peer data
- [DATATYPES.md](DATATYPES.md) — INT8/Float32 completed; INT16 inference planned
- [GENERIC_KERNELS.md](GENERIC_KERNELS.md#36-optional-im2col--gemm-netkit_im2col) — existing optional im2col-to-GEMM reference strategy, not multicore
- [PLATFORMS.md](PLATFORMS.md#target--cpu-isa) — S3/P4 both use `mcu_esp` despite different ISAs
- [KNOWN_ISSUES.md](KNOWN_ISSUES.md#ki-003--esp32-p4-esp-nn-pie-asm-disabled-under-platformio-gas) — P4 PIE assembler blocker
