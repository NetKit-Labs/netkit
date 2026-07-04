# Kernel backends (CRTP)

netkit routes all numeric work through a **compile-time kernel facade**. Layer code (`ops`, `conv2d`, `cnn`, `mlp`) calls `Kernels::Op(...)` only — no `#if CMSIS` and no runtime backend switching in those paths.

## Architecture

```
ops.cpp / conv2d.cpp / cnn.cpp / mlp.cpp
        │
        ▼
  active_kernel.hpp          ← picks backend alias at compile time
        │
        ▼
  ComposedKernel<…>          ← kernel_dispatch.hpp (Try* + fallback)
        │
   ┌────┴────┐
   ▼         ▼
VectorFast  LayerFast        ← CmsisDspKernel, CmsisNnKernel, or ReferenceKernel
   │         │
   └────┬────┘
        ▼
  ReferenceKernel            ← always linked; fallback for any Try* miss
```

| Header | Role |
|--------|------|
| `kernel_crtp.hpp` | `KernelBase<Derived>` — static `Kernels::Mul` → `Derived::MulImpl` |
| `kernel_activation.hpp` | `NetkitKernelActivation` enum for fused conv/FC activations |
| `reference_kernel.hpp` / `reference_kernel.cpp` | Portable float32 implementations |
| `cmsis_dsp_kernel.hpp` / `cmsis_dsp_backend.cpp` | CMSIS-DSP `Try*` for vector ops (add, mul, matmul, clip, batch-norm fallback, LayerNorm2d, GRN) |
| `cmsis_nn_kernel.hpp` / `cmsis_nn_backend.cpp` | CMSIS-NN `Try*` for layer ops (conv, depthwise conv, pool, FC, batch norm, activations, GELU, softmax) |
| `kernel_dispatch.hpp` | `ComposedKernel<VectorFast, LayerFast>` and `Try*` helpers |
| `active_kernel.hpp` | `using Kernels = …` alias for the current build |
| `fused_kernel_ops.hpp` | Inline helpers for fused blocks (BN, ReLU, MatAdd, FC 1×1, GELU, GRN) → `Kernels::` |
| `activation_followup.hpp` | Shared post-kernel activation when not fused in CMSIS-NN |

Backend `.cpp` files **must** include `netkit_config.h` so `NETKIT_CMSIS_NN_ALLOWED` and related macros match the active build profile.

## Active backend aliases

Selected in `active_kernel.hpp` from `NETKIT_USE_CMSIS_NN`, `NETKIT_USE_CMSIS_DSP`, and `NETKIT_CMSIS_NN_ALLOWED`:

| Build profile | `Kernels` alias |
|---------------|-----------------|
| Reference only | `ReferenceKernel` |
| CMSIS-DSP only (desktop / MPU, or MCU without NN) | `ComposedKernel<CmsisDspKernel, ReferenceKernel>` |
| CMSIS-NN only (MCU + Cortex-M) | `ComposedKernel<ReferenceKernel, CmsisNnKernel>` |
| CMSIS-NN + CMSIS-DSP (MCU firmware) | `ComposedKernel<CmsisDspKernel, CmsisNnKernel>` |

### Role split in `ComposedKernel<VectorFast, LayerFast>`

| Role | Typical backend | Ops |
|------|-----------------|-----|
| **VectorFast** | CMSIS-DSP when enabled | `Mul`, `MatMul`, `MulScalar`, `MatAdd`/`MatAddND`, `ReLU6` clip fallback, `BatchNorm2d` (desktop fallback), `LayerNorm2d`, `Grn2dForward` |
| **LayerFast** | CMSIS-NN when enabled | `Conv2d`, depthwise conv, pool, batch norm, FC, NN activations, `Gelu`, softmax |
| **Reference** | Always | Fallback when `Try*` returns false or backend is `ReferenceKernel` |

**GEMM:** there is no separate `Gemm` symbol. General matrix multiply is `Kernels::MatMul` (`Ops::MatMul`); linear layers use `Kernels::FullyConnectedWithBias` (internally matmul + bias). ONNX `Gemm` is lowered to packed dense weights at export time.

On **MCU with both CMSIS flags**, CMSIS-NN owns layer kernels; CMSIS-DSP accelerates vector ops and ops without NN float APIs (LayerNorm, GRN). They do not compete for the same op.

### Float32 op coverage

| Op | Reference | CMSIS-NN | CMSIS-DSP |
|----|-----------|----------|-----------|
| Conv2D, depthwise conv, pool, batch norm, FC, activations, softmax | ✓ | ✓ (`Try*` + fallback) | FC/BN fallback on some builds |
| MatMul, elementwise mul/add/scale | ✓ | add (elementwise) | ✓ |
| LayerNorm2d | ✓ | — | ✓ |
| GELU | ✓ | ✓ (tanh on inner polynomial) | — (falls back to reference) |
| GRN | ✓ | — | ✓ (mean + vector mul/add per pixel) |
| Residual / skip merge | ✓ | add (`MatAddND`) | add |

**Depthwise conv** is **2D-only** in the API (`DepthwiseConv2D`, NHWC `[H,W,C]`, weights `[C,Kh,Kw]`). **1D** along time/height is expressed as a degenerate 2D kernel (e.g. `kernel_h=5`, `kernel_w=1` on input `[T,1,C]`). See [NK_FORMAT.md](NK_FORMAT.md) and `python/README.md`.

**Fused blocks** (ResNet BasicBlock, MobileNetV4 UIB, ConvNeXt V2) route internal BN, ReLU, FC, LayerNorm, GELU, GRN, and residual adds through `fused_kernel_ops.hpp` → `Kernels::`, so CMSIS applies when enabled.

## Dispatch pattern

`kernel_dispatch.hpp` uses **C++20 concepts** (`NkAcceleratedKernel`) and `if constexpr` to pick CMSIS `Try*` paths at compile time — no runtime backend switching, no virtual functions:

```cpp
template<typename T>
concept NkAcceleratedKernel = !std::same_as<T, ReferenceKernel>;

if constexpr (NkAcceleratedKernel<LayerFast>) {
    if (!LayerFast::TryConv2dForward(...)) { /* reference fallback */ }
}
```

Layer code calls `Kernels::Op(...)` only — not raw `ReferenceKernel::` (except inside kernel dispatch fallbacks).

Each fast backend exposes static `Try*` methods that return `bool`:

- `true` — backend handled the op
- `false` — fall through to reference (or the other backend role)

Example from `ComposedKernel`:

```cpp
static void MaxPool2dForwardImpl(const Tensor& input, int pool_size, int stride,
                                 int pad_h, int pad_w, Tensor& output)
{
    if constexpr (NkAcceleratedKernel<LayerFast>)
    {
        if (!LayerFast::TryMaxPool2dForward(input, pool_size, stride, pad_h, pad_w,
                                              NetkitKernelActivation::None, output))
            ReferenceKernel::MaxPool2dForwardImpl(input, pool_size, stride, pad_h, pad_w, output);
    }
    else
        ReferenceKernel::MaxPool2dForwardImpl(input, pool_size, stride, pad_h, pad_w, output);
}
```

No virtual functions; the compiler inlines the selected path for each translation unit.

## Adding a new kernel op

1. Add `OpImpl` to `ReferenceKernel` and declare on `KernelBase`.
2. Add `TryOp` to CMSIS backends if applicable (guard with `#if NETKIT_USE_CMSIS_*` in `.cpp`).
3. Wire `ComposedKernel::OpImpl` via a `Try*` helper in `kernel_dispatch.hpp`.
4. Call `Kernels::Op(...)` from layer code — not from a new `#if` branch.

## Layer dispatch (OpsResolver)

CNN forward uses a **static function-pointer registry** (`ops_resolver.hpp`) — no virtuals, heap, or `std::vector`.

### C++26 compile-time resolver tables

Each layer op is a descriptor struct with `static constexpr NkLayerOpRegistration kRegistration`. `NkOpList<Ops...>` folds descriptors into a **`constinit`** registration array and resolver view — built at compile time with no dynamic static initialization (MCU-safe):

```cpp
struct NkConv2DOpDescriptor {
    static constexpr NkLayerOpRegistration kRegistration = { ... };
};

using MyOps = NkOpList<NkConv2DOpDescriptor, NkDenseOpDescriptor>;
cnn.SetOpsResolver(MyOps::View());  // reference to constinit static storage
```

Descriptors live in `layer_op_registry.hpp`; implementations in `src/layer_ops/*.cpp`. `NkAllLayerOps` is the full six-op table used by `GetDefaultOpsResolver()`.

### Per-op translation units (firmware DCE)

| Descriptor header | Implementation |
|-------------------|----------------|
| `layer_ops/nk_conv2d_op.hpp` | `src/layer_ops/nk_op_conv2d.cpp` |
| `layer_ops/nk_max_pool2d_op.hpp` | `src/layer_ops/nk_op_max_pool2d.cpp` |
| `layer_ops/nk_avg_pool2d_op.hpp` | `src/layer_ops/nk_op_avg_pool2d.cpp` |
| `layer_ops/nk_batch_norm2d_op.hpp` | `src/layer_ops/nk_op_batch_norm2d.cpp` |
| `layer_ops/nk_flatten_op.hpp` | `src/layer_ops/nk_op_flatten.cpp` |
| `layer_ops/nk_dense_op.hpp` | `src/layer_ops/nk_op_dense.cpp` |

Link only the `.cpp` files matching your `NkOpList<...>` — unused `NkEval*` bodies are dropped by the linker.

```
CNNNetwork::forward
        │
        ▼
  NkOpsResolver::Find(opcode)
        │
        ▼
  prepare_output / eval  →  Kernels::…
```

## Related docs

- [BUILD_TARGETS.md](BUILD_TARGETS.md) — Make/CMake flags for CMSIS backends
- [PHILOSOPHY.md](PHILOSOPHY.md) — Phase 1 interpreter vs Phase 2 packager optimizations
