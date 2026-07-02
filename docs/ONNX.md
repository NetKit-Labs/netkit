# ONNX Import

netkit includes a **dependency-free ONNX importer** written in C++26. It parses ONNX `ModelProto` protobuf wire format directly (no libprotobuf, no ONNX Runtime) and loads supported graphs into `MLPNetwork` / `CNNNetwork` for inference and parity testing.

For deployment, convert ONNX to **`.nk`** with the Python packager — the C++ runtime loads `.nk` only.

## Python packager (recommended)

```bash
pip install -e python
python -m netkit convert models/my_model.onnx -o models/my_model.nk
make export-nk    # all bundled regression models

./netkit run models/my_model.nk --input 1,2,3
```

See [python/README.md](../python/README.md) and [NK_FORMAT.md](NK_FORMAT.md).

## C++ runtime importer (parity tests)

The C++ importer loads ONNX directly into arena-backed networks — used by ONNX parity tests in `src/test_onnx.cpp` to compare against committed `.nk` files.

```cpp
#include "onnx_importer.hpp"

MLPNetwork* mlp = nullptr;
CNNNetwork* cnn = nullptr;
NkLoader::NetworkKind kind{};
std::array<uint32_t, kMaxTensorRank> input_shape{};
uint32_t input_rank = 0;

OnnxImporter::LoadFromOnnx("model.onnx", arena, kind, mlp, cnn, input_shape, input_rank);
```

Headers:

| Header | Purpose |
|--------|---------|
| `onnx_importer.hpp` | Load-from-ONNX API |
| `onnx_model.hpp` | Parsed ONNX graph (internal/debug) |
| `protobuf_wire.hpp` | Minimal protobuf wire decoder |

## Supported ONNX operators

| ONNX op | netkit layer | Notes |
|---------|--------------|-------|
| `Gemm` | `dense` | float32 weights/bias initializers; `transB` supported |
| `Conv` | `conv2d` | NCHW weights → netkit `[O,Kh,Kw,I]`; **valid conv, no padding** |
| `MaxPool` | `max_pool2d` | square kernel from `kernel_shape` |
| `Flatten` | `flatten` | CNN head only |
| `Relu` | activation | fused when immediately after Gemm/Conv |
| `Softmax` | activation | fused when immediately after final Gemm |

## Input layouts

| netkit network | ONNX graph input | `.nk` input shape |
|----------------|------------------|-------------------|
| MLP | `[batch, features]` | same |
| CNN | `[N, C, H, W]` (NCHW) | `[H, W, C]` NHWC |

At inference time, feed CNN inputs in **NHWC flatten order** (same as existing netkit CNN models). The importer reorders conv weights; it does **not** transpose runtime inputs.

## Limitations (v1)

- **Float32 only** — other ONNX `TensorProto` types are rejected
- **No external data** — weights must be embedded in the `.onnx` file (`raw_data` or `float_data`)
- **Linear graphs** — no branches, skip connections, or subgraphs
- **No `Pad`** — padded convolutions are not supported (matches netkit valid conv)
- **Square kernels** — `Conv` / `MaxPool` use one `kernel_shape` value for height and width
- **Desktop import tool** — uses `std::vector` while parsing; inference still uses the arena

PyTorch/TensorFlow exports often include `MatMul`, `Add`, `BatchNormalization`, or `Reshape` nodes — re-export or simplify the graph (e.g. `torch.onnx.export` on an `nn.Sequential`) or extend the importer.

## Regenerating test ONNX files

```bash
pip install onnx numpy
python3 tools/export_onnx_test_models.py
make export-nk
make test-cpp
```

Committed assets: `models/*.onnx` for all regression models (`test_mlp`, `mlp_hand`, `test_cnn`, `cnn_4x4_single`, `cnn_hand`, `mnist_mlp`, `mnist_cnn`). Parity tests in `src/test_onnx.cpp` run the same inputs through `.nk` and ONNX import and compare outputs.

## Related docs

- [NK_FORMAT.md](NK_FORMAT.md) — binary model layout
- [ARENA.md](ARENA.md) — size the arena for loaded models
- [CLI.md](CLI.md) — full CLI reference
