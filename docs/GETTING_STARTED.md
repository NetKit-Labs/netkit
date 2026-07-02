# Getting Started

This guide gets you from clone to running inference on the desktop in a few minutes.

**Related docs:** [CLI](CLI.md) · [Arena](ARENA.md) · [Data types](DATATYPES.md) · [.nk format](NK_FORMAT.md) · [Testing](TESTING.md) · [MNIST](MNIST.md) · [MNIST CNN](MNIST_CNN.md) · [C API](c-api.md) · [C++ API](cpp-api.md)

## Requirements

| Component | Standard | Compiler |
|-----------|----------|----------|
| Core engine + C++ API | **C++26** | clang++ 17+, g++ 14+, or MSVC `/std:c++latest` |
| C API (`netkit.h`) | **C23** | clang 17+ or gcc 14+ with `-std=c23` |
| Build | Make | Any Unix-like environment |

No external dependencies beyond the standard library for the C++ runtime. The Python packager (`python/`) needs numpy and onnx.

All inference tensors, weights, and math use **float32** (`float`) today. There is no float64 path. Planned types: float16, int16, int8, int4 — see [DATATYPES.md](DATATYPES.md).

Arena allocations use **explicit alignment** (`alignof(float)` for weights/tensors, `alignof(T)` for network structs). See [ARENA.md](ARENA.md) and [API.md — Memory model](API.md#memory-model).

## Build

```bash
git clone https://github.com/jameslavrenz/netkit.git
cd netkit
make
```

This produces:

- **`netkit`** — CLI tool for tests, one-off inference, and inspection
- **`libnetkit.a`** — static library (C++ core + C API bridge)

Optional usage demos:

```bash
make example-cpp   # C++26: examples/infer_cpp
make example-c     # C23:    examples/infer_c
```

## Run the test suite

```bash
make test        # C++ API tests, then C API tests
make test-cpp    # C++26 only: ./netkit test
make test-c      # C23 only:  ./tests/test_c_api
```

Each suite runs **72 inference regression cases** (16 hand-checked + 10 MNIST MLP + 10 MNIST CNN + 36 ONNX parity) plus C API smoke tests. See [TESTING.md](TESTING.md).

## Run inference from the CLI

See [CLI.md](CLI.md) for full command reference.

```bash
./netkit help    # or: ./netkit -h  /  ./netkit --help

# MLP: 2 inputs -> 2 outputs
./netkit run models/test_mlp.nk --input 1,2

# CNN: 16 inputs (4x4x1)
./netkit run models/cnn_4x4_single.nk --input=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
```

## Inspect a model

```bash
./netkit inspect models/test_mlp.nk
./netkit inspect models/test_mlp.nk --full
```

Default mode prints a boxed network summary. `--full` adds arena memory usage after load and a zero-input forward pass.

## Size the arena buffer

Every inference path uses an arena — there is no heap fallback for weights or activations. The buffer size is **your choice**, not stored in the `.nk` file.

1. Run `./netkit inspect models/your_model.nk --full` (or `nk_inspect_model()` from C).
2. Note **arena bytes after forward**.
3. Declare a static buffer with headroom (often 1.5–2×), then `arena.init(buffer, sizeof(buffer))`.

Hand models fit in **64 KiB** (`Arena::kDefaultCapacity`). MNIST needs **~2 MiB** (MLP) or **~4 MiB** (CNN) — see [ARENA.md](ARENA.md) and [TESTING.md](TESTING.md) for where each harness sets its size.

At load time, netkit allocates weights plus **two ping-pong activation buffers** (largest layer output × 2). Forward passes reuse those buffers instead of allocating one tensor per layer.

## Use the C API (C23)

Build and run the full example:

```bash
make example-c
./examples/infer_c models/test_mlp.nk 1 2
```

Source: [`examples/infer_c.c`](../examples/infer_c.c). Minimal integration pattern:

```c
#include "netkit.h"

alignas(max_align_t) static unsigned char memory[NK_ARENA_DEFAULT_CAPACITY];
nk_arena_t arena;
nk_model_t model;

nk_arena_init(&arena, memory, sizeof(memory));
nk_model_load("models/test_mlp.nk", &arena, &model);

float input[] = {1.0f, 2.0f};
float output[2];
uint32_t output_count = 0;

nk_model_run(&model, &arena, input, 2, output, 2, &output_count);
```

Link with `libnetkit.a` using a C++ linker driver (the library contains C++ object code):

```bash
clang -std=c23 -Iinclude -c my_app.c -o my_app.o
clang++ -std=c++26 -o my_app my_app.o libnetkit.a
```

See [c-api.md](c-api.md) for the full C reference.

## Use the C++ API (C++26)

Build and run the full example:

```bash
make example-cpp
./examples/infer_cpp models/test_mlp.nk 1 2
```

Source: [`examples/infer_cpp.cpp`](../examples/infer_cpp.cpp). Minimal integration pattern:

```cpp
#include "arena.hpp"
#include "nk_loader.hpp"
#include "tensor_factory.hpp"

alignas(std::max_align_t) unsigned char buffer[Arena::kDefaultCapacity];
Arena arena;
arena.init(buffer, sizeof(buffer));

MLPNetwork* network = nullptr;
std::array<uint32_t, kMaxTensorRank> input_shape{};
uint32_t input_rank = 0;

NkLoader::LoadMLP("models/test_mlp.nk", arena, network, input_shape, input_rank);

Tensor input = TensorFactory::Create2D(arena, input_shape[0], input_shape[1]);
TensorFactory::Fill(input, {1.0f, 2.0f});

Tensor output = TensorFactory::Create2D(arena, 1, network->GetLayer(1).weights.shape[1]);
network->forward(input, output, arena);
```

See [cpp-api.md](cpp-api.md) for the full C++ reference.

## Model files

Runtime models are single **`.nk`** files (architecture + float32 weights). Convert from ONNX with the Python packager:

```bash
pip install -e python
python -m netkit convert models/test_mlp.onnx -o models/test_mlp.nk
make export-nk    # regenerate all bundled regression models
```

| File | Purpose |
|------|---------|
| `model.nk` | Single-file model for inference |
| `model.onnx` | Source graph for conversion |
| Embedded tests (optional) | Regression cases in `.nk` — see [NK_FORMAT.md](NK_FORMAT.md) |

Format spec: [NK_FORMAT.md](NK_FORMAT.md).

Example workflow for a new hand-tested model:

1. Export or train an ONNX graph (or use an existing `models/*.onnx`)
2. `python -m netkit convert model.onnx -o model.nk`
3. Add embedded cases when writing the `.nk` file — see [NK_FORMAT.md](NK_FORMAT.md) and `python/netkit/regression_data.py`
4. Register the model in `src/test.cpp` if it is a new bundle
5. Run `make test`

For MNIST or other large embedded regression suites, see [TESTING.md](TESTING.md) and [MNIST.md](MNIST.md).

## Project layout

```
netkit/
├── include/          Headers (C++ + netkit.h)
├── src/              C++26 implementation
├── python/netkit/    ONNX → .nk packager
├── examples/
│   ├── infer_cpp.cpp # C++26 usage demo
│   └── infer_c.c     # C23 usage demo
├── tests/
│   └── test_c_api.c  # C23 API regression tests
├── models/           .nk bundles, *.onnx, test fixtures
├── tools/            Python helpers for MNIST export
└── docs/             Guides and API reference
```

## Next steps

- Read [API.md](API.md) for an overview of both APIs
- Read [TESTING.md](TESTING.md) for regression suite layout
- Read [CLI.md](CLI.md) for `test`, `run`, and `inspect`
- Add a regression case — [NK_FORMAT.md](NK_FORMAT.md)
- Use `./netkit inspect --full` to size your arena before deploying to embedded targets
