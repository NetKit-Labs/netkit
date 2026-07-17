# Third-party notices

netkit itself is **MIT** — see [LICENSE](LICENSE).

This file lists third-party software used by this repository: **runtime backends**
linked into netkit, **Python tooling**, and **peer-benchmark** stacks that are
fetched locally (not shipped inside `libnetkit`). Full license texts for the
components below live under [`third_party/licenses/`](third_party/licenses/).
Upstream trees (git submodules / fetch scripts) also retain their own `LICENSE`
files when present.

**CMSIS-DSP is not used** by netkit (not fetched, not linked). See
[third_party/README.md](third_party/README.md).

---

## Summary

| Component | Role in netkit | SPDX / license | Texts / upstream |
|-----------|----------------|----------------|------------------|
| **CMSIS-Core** (ARM CMSIS_6) | MCU device headers (`NETKIT_ARCH`) | Apache-2.0 | [`CMSIS-Core.Apache-2.0.txt`](third_party/licenses/CMSIS-Core.Apache-2.0.txt); submodule `third_party/CMSIS-Core` |
| **CMSIS-NN** | Optional Arm MCU int8 kernels | Apache-2.0 | [`CMSIS-NN.Apache-2.0.txt`](third_party/licenses/CMSIS-NN.Apache-2.0.txt); submodule `third_party/CMSIS-NN` |
| **CMSIS-DSP** | **Not used** | — | — |
| **XNNPACK** | Optional cpu / MPU LayerFast (float32 + int8) | BSD-3-Clause | [`XNNPACK.BSD-3-Clause.txt`](third_party/licenses/XNNPACK.BSD-3-Clause.txt); fetch `third_party/XNNPACK/` |
| **pthreadpool** | XNNPACK dependency | BSD-style | [`pthreadpool.BSD.txt`](third_party/licenses/pthreadpool.BSD.txt) |
| **cpuinfo** | XNNPACK dependency | BSD-style | [`cpuinfo.BSD.txt`](third_party/licenses/cpuinfo.BSD.txt) |
| **clog** | cpuinfo dependency | BSD-style | [`clog.BSD.txt`](third_party/licenses/clog.BSD.txt) |
| **FXdiv** | XNNPACK dependency | MIT | [`FXdiv.MIT.txt`](third_party/licenses/FXdiv.MIT.txt) |
| **KleidiAI** | XNNPACK dependency (Arm) | Apache-2.0 (+ BSD-3 files) | [`KleidiAI.Apache-2.0.txt`](third_party/licenses/KleidiAI.Apache-2.0.txt), [`KleidiAI.BSD-3-Clause.txt`](third_party/licenses/KleidiAI.BSD-3-Clause.txt) |
| **ONNX** (Python package) | Packager / model IR (`pip` / `python/`) | Apache-2.0 | [`onnx.Apache-2.0.txt`](third_party/licenses/onnx.Apache-2.0.txt); https://github.com/onnx/onnx |
| **ONNX Runtime** | Python parity + host peer bench | MIT | [`onnxruntime.MIT.txt`](third_party/licenses/onnxruntime.MIT.txt); https://github.com/microsoft/onnxruntime — also see upstream `ThirdPartyNotices.txt` in an ORT checkout for ORT’s own transitive deps |
| **TensorFlow Lite / LiteRT** | Host peer bench (Python wheel) | Apache-2.0 | https://github.com/google-ai-edge/LiteRT (and/or TensorFlow) — same Apache-2.0 family as TFLM |
| **TensorFlow Lite Micro (TFLM)** | MCU / host peer bench | Apache-2.0 | [`tflite-micro.Apache-2.0.txt`](third_party/licenses/tflite-micro.Apache-2.0.txt); fetch `benchmark/tflm/third_party/tflite-micro` |
| **flatbuffers / gemmlowp / ruy** | TFLM make downloads (peer only) | Apache-2.0 | [`flatbuffers`](third_party/licenses/flatbuffers.Apache-2.0.txt), [`gemmlowp`](third_party/licenses/gemmlowp.Apache-2.0.txt), [`ruy`](third_party/licenses/ruy.Apache-2.0.txt) |
| **Apache TVM** | MCU peer bench (external `TVM_HOME`) | Apache-2.0 | [`apache-tvm.Apache-2.0.txt`](third_party/licenses/apache-tvm.Apache-2.0.txt), [`apache-tvm.NOTICE.txt`](third_party/licenses/apache-tvm.NOTICE.txt); https://github.com/apache/tvm |
| **microTVM** | MCU AOT peer (part of Apache TVM) | Apache-2.0 | Same as Apache TVM; boards `boards/nucleo-f446re-tvm-*` |
| **TVM bundled 3rd-party** | Shipped in TVM `licenses/` (peer builds) | various | [`third_party/licenses/apache-tvm/`](third_party/licenses/apache-tvm/); plus [`dmlc-core`](third_party/licenses/apache-tvm-3rdparty-dmlc-core.LICENSE.txt) |

---

## Runtime backends (may link into netkit)

### CMSIS-Core — Apache License 2.0

Copyright © Arm Limited and Contributors.

Fetched as git submodule `third_party/CMSIS-Core` from
[ARM-software/CMSIS_6](https://github.com/ARM-software/CMSIS_6). Used for
Cortex-M device headers when `NETKIT_ARCH` is set.

License text: [third_party/licenses/CMSIS-Core.Apache-2.0.txt](third_party/licenses/CMSIS-Core.Apache-2.0.txt).

### CMSIS-NN — Apache License 2.0

Copyright © Arm Limited and Contributors.

Fetched as git submodule `third_party/CMSIS-NN` from
[ARM-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN). Optional
(`NETKIT_CMSIS_NN=1`) on `NETKIT_TARGET=mcu_arm` + Cortex-M.

License text: [third_party/licenses/CMSIS-NN.Apache-2.0.txt](third_party/licenses/CMSIS-NN.Apache-2.0.txt).

### CMSIS-DSP — not used

netkit does **not** fetch, compile, or link
[ARM-software/CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP).

### XNNPACK — BSD 3-Clause

Copyright (c) Facebook, Inc. and its affiliates.  
Copyright 2019 Google LLC

Fetched by `./tools/fetch_xnnpack.sh` into `third_party/XNNPACK/` (gitignored
build tree). Optional on **cpu** and **any MPU** (`NETKIT_XNNPACK=1`);
**forbidden** on MCU.

License text: [third_party/licenses/XNNPACK.BSD-3-Clause.txt](third_party/licenses/XNNPACK.BSD-3-Clause.txt).

When redistributing binaries that link XNNPACK, retain the copyright notice and
disclaimer (satisfied by this file + the license texts under
`third_party/licenses/`).

### XNNPACK transitive dependencies

Pulled by XNNPACK’s CMake build (typical desktop/MPU pin used by netkit):

| Library | License | Text |
|---------|---------|------|
| pthreadpool | BSD-style | [pthreadpool.BSD.txt](third_party/licenses/pthreadpool.BSD.txt) |
| cpuinfo | BSD-style | [cpuinfo.BSD.txt](third_party/licenses/cpuinfo.BSD.txt) |
| clog | BSD-style | [clog.BSD.txt](third_party/licenses/clog.BSD.txt) |
| FXdiv | MIT | [FXdiv.MIT.txt](third_party/licenses/FXdiv.MIT.txt) |
| KleidiAI | Apache-2.0 (plus BSD-3 license files in upstream) | [KleidiAI.Apache-2.0.txt](third_party/licenses/KleidiAI.Apache-2.0.txt), [KleidiAI.BSD-3-Clause.txt](third_party/licenses/KleidiAI.BSD-3-Clause.txt) |

Exact dependency set can vary by XNNPACK commit / target ISA; after
`make xnnpack-init`, also inspect `LICENSE` / `LICENSES` under
`third_party/XNNPACK/build/*-source/`.

---

## Python tooling (not linked into firmware)

### ONNX — Apache License 2.0

[onnx/onnx](https://github.com/onnx/onnx). Declared in `python/pyproject.toml`
(`pip install -e python`). Used to import/convert graphs to `.nk`.

License text: [third_party/licenses/onnx.Apache-2.0.txt](third_party/licenses/onnx.Apache-2.0.txt).

### ONNX Runtime — MIT

Copyright (c) Microsoft Corporation.

Used for Python ONNX parity tests and as a **host CPU peer** (optional
LiteRT-matched source build via `tools/build_onnxruntime_litert_matched.sh`).
**Not** linked into `libnetkit`.

License text: [third_party/licenses/onnxruntime.MIT.txt](third_party/licenses/onnxruntime.MIT.txt).

ORT itself vendors additional third-party code (e.g. MLAS and other EPs). Those
notices ship in the ORT tree as `ThirdPartyNotices.txt` when you build or
install ORT; they apply to the ORT binary/wheel, not to netkit’s own library.

Optional train extras (`torch`, `timm`, …) are separate PyPI packages with their
own licenses; install only if you use `pip install -e "python[train]"`.

---

## Peer benchmarks only (fetched locally; not part of libnetkit)

These stacks are used to compare latency / accuracy. They are **not** compiled
into netkit’s runtime library. Running the peer suites downloads or expects a
local checkout; license obligations for those trees apply when you use or
redistribute those peer builds.

### TensorFlow Lite / LiteRT — Apache License 2.0

Host peer Python benches under `benchmark/tflite/` (LiteRT / TF Lite wheel).
Upstream: [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT)
(and historically TensorFlow Lite under
[tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)). Apache-2.0.

### TensorFlow Lite Micro — Apache License 2.0

Fetched by `benchmark/tflm/tools/fetch_tflm.sh` into
`benchmark/tflm/third_party/tflite-micro`.

License text: [third_party/licenses/tflite-micro.Apache-2.0.txt](third_party/licenses/tflite-micro.Apache-2.0.txt).

TFLM’s make system downloads additional Apache-2.0 components (examples copied
here for convenience): flatbuffers, gemmlowp, ruy, and a bundled CMSIS-NN pin —
see [`third_party/licenses/`](third_party/licenses/). Further TFLM downloads
(e.g. pigweed) keep their licenses inside the TFLM tree after fetch.

### Apache TVM / microTVM — Apache License 2.0

Copyright 2019–2023 The Apache Software Foundation.

**microTVM** is the embedded / micro controller path inside **Apache TVM** (same
project and license) — ahead-of-time C runtime used by the MCU peer boards
`boards/nucleo-f446re-tvm-cnn-int8` and `boards/nucleo-f446re-tvm-cnn-dw-int8`.
Those boards expect an external `TVM_HOME` (Relay-era **v0.14** with
`USE_MICRO=ON`, and optionally `USE_CMSISNN=ON` for CMSIS-NN BYOC). TVM source
is **not** vendored as a git submodule here; the peer firmware links code
generated from your local TVM build.

Upstream: [apache/tvm](https://github.com/apache/tvm).

| Text | Path |
|------|------|
| Apache License 2.0 | [third_party/licenses/apache-tvm.Apache-2.0.txt](third_party/licenses/apache-tvm.Apache-2.0.txt) |
| ASF NOTICE | [third_party/licenses/apache-tvm.NOTICE.txt](third_party/licenses/apache-tvm.NOTICE.txt) |
| TVM `licenses/` bundle (dlpack, rang, picojson, …) | [third_party/licenses/apache-tvm/](third_party/licenses/apache-tvm/) |
| dmlc-core (common TVM 3rdparty; Apache-2.0) | [apache-tvm-3rdparty-dmlc-core.LICENSE.txt](third_party/licenses/apache-tvm-3rdparty-dmlc-core.LICENSE.txt) |

When a microTVM peer build enables **CMSIS-NN BYOC**, Arm CMSIS-NN Apache-2.0
terms also apply (see CMSIS-NN above). GPU-oriented TVM deps (CUTLASS, flash
attention, …) appear in TVM’s `licenses/` bundle for completeness; the NUCLEO
microTVM peers do not use them.

Texts above were mirrored from a local Apache TVM **v0.14** tree
(`LICENSE`, `NOTICE`, `licenses/`). Refresh after bumping `TVM_HOME`.

---

## Redistribution notes

- **Shipping `libnetkit` with CMSIS-NN and/or XNNPACK enabled:** include this
  file (or equivalent attribution) and the matching texts under
  `third_party/licenses/` for every backend you actually link. Apache-2.0
  requires providing a copy of the Apache license; BSD-3 / MIT require
  retaining copyright notices and disclaimers.
- **Peer-only tools** (TFLM, LiteRT, ORT, TVM): include their notices if you
  redistribute those peer binaries or wheels; they are not required solely
  because this repo documents how to run A/B benches.
- **Submodules / fetch trees** are the source of truth for the exact pin; the
  copies under `third_party/licenses/` are mirrored for offline attribution when
  the large trees are not checked out.

Integration overview: [third_party/README.md](third_party/README.md).
