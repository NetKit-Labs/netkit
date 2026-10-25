# Hardware Verification and Performance Report — NetKit

**Targets:** NUCLEO-F446RE (Arm Cortex-M4F), ESP32-C3/S3/P4 MCU boards, Pi Zero 2 W / Cortex-A53 embedded Linux  
**Evidence review:** 2026-10-25; retained on-device A/B logs and host validation (**not newly executed**)  
**Scope:** INT8/float32 inference correctness, peer latency, backend isolation, and embedded platform coverage.

## 1. Method and evidence boundaries

On-device MCU peer comparisons use **10 runs × 10 images**, discarding the first invocation of each run; published result sets report **10/10** on their selected digit test images. This establishes accuracy **on those samples**, not statistical MNIST-test-set accuracy. Comparisons separate optimized CMSIS-NN/ESP-NN backends from reference paths; mixing those columns exaggerates or reverses speedup. See [MCU A/B methodology/log index](../benchmark/mcu_ab_logs/README.md), [benchmark methodology](../benchmark/README.md), [platform status](STATUS.md).

## 2. Measured MCU A/B results

All numbers below are **mean inference latency** as recorded for selected models, not whole-application throughput or worst-case timing.

| Board, backend, model | NetKit | Peer | Interpretation / source |
| --- | ---: | ---: | --- |
| STM32F446RE **180 MHz**, CMSIS-NN, MNIST CNN INT8 | **95.3 ms** | TFLM **95.5 ms** | Comparable optimized backend; [MCU logs](../benchmark/mcu_ab_logs/README.md) |
| STM32F446RE, CMSIS-NN, MNIST DS-CNN INT8 | **58.3 ms** | TFLM **61.4 ms** | NetKit modestly faster |
| STM32F446RE, reference (CMSIS off), CNN INT8 | **336.2 ms** | TFLM **2593.5 ms** | **~7.71×** in reference-only comparison, **not** optimized vs optimized |
| ESP32-C3 **160 MHz**, ESP-NN, CNN INT8 | **252.0 ms** | TFLM **251.4 ms** | Essentially equal; peer slightly faster |
| ESP32-C3, reference CNN INT8 | **226.8 ms** | TFLM **1205.5 ms** | ~**5.32×** reference-path comparison |
| ESP32-S3 **240 MHz**, ESP-NN, CNN INT8 | **34.720 ms** | TFLM **34.851 ms** | Near parity; [raw summary](../benchmark/mcu_ab_logs/xiao_esp32s3/SUMMARY.md) |
| ESP32-S3, reference CNN INT8 | **112.069 ms** | TFLM **1113.036 ms** | ~**9.93×** reference-path comparison |
| ESP32-S3, reference DS-CNN INT8 | **64.307 ms** | TFLM **362.838 ms** | ~**5.64×** reference-path comparison |
| ESP32-P4 **360 MHz**, ESP-NN, CNN INT8 | **78.9 ms** | TFLM **79.3 ms** | Near parity, optimized backends |
| ESP32-P4, reference CNN INT8 | **77.1 ms** | TFLM **485.4 ms** | ~**6.29×** reference-path comparison |
| ESP32-P4, reference CNN float32 | **97.5 ms** | TFLM **166.4 ms** | ~**1.71×**; lowered-AOT NetKit path |

All comparisons above are from [archived A/B summaries](../benchmark/mcu_ab_logs/README.md), which link individual on-device UART captures. The P4 ESP-NN numbers use a portable backend, **not** the P4 PIE assembly path. MCU and Pi targets should not be compared directly without controlling compiler/kernel/settings.

### Known correctness failure retained in the evidence

**KI-001:** The ESP32-S3 and ESP32-P4 float32 interpreter-embed path (`--no-lower`) is known to produce incorrect classifications. On S3 the recorded case passed only **2/10** images; published float peer tables use the *lowered AOT* path instead. This is an open, documented limitation, not a passed production criterion. [Known issues](KNOWN_ISSUES.md), [S3 capture](../benchmark/mcu_ab_logs/xiao_esp32s3/cnn_f32_netkit_embed.txt).

## 3. Host/CI tests and reproducibility

The main GitHub Actions workflow builds with Clang; runs `make rebuild test` for C++, C and Python checks; executes a CMake build/test, C++ and C example smoke tests, and CLI checks. The status inventory lists **89 .nk test-case assets** for the C/C++ regression paths; this is **not a coverage percentage**. On-device board runners and benchmark captures are **not** part of hosted CI. [CI workflow](../.github/workflows/ci.yml), [test status](STATUS.md), [testing guide](TESTING.md).

```sh
make rebuild test
# Board A/B reproduction: follow per-board scripts and pinned dependencies.
# Read source logs, toolchains, model hashes and flags before comparing results.
```

## 4. Disposition and additional evidence needed

**Supported:** repeatable, saved physical-MCU benchmark comparisons with correct optimized-versus-reference differentiation, cross-platform tests, and honest open-issue tracking. **Not supported by current records:** an end-to-end power/energy metric, normalized CPU utilization across devices, worst-case inference latency, long-duration reliability, universal `10×` speedup across optimized backends, or resolution of KI-001. New published comparisons should include raw serial logs, model hash, toolchain/backend selections, clock, repeats, warmup policy and pass/fail counts.
