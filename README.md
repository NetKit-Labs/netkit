# netkit — Neural Network Kit

netkit is a C++26 neural network kit for on-device inference on MCUs and MPUs. It is developed and validated on the desktop, then deployed to embedded targets. Companion to [memkit](https://github.com/jameslavrenz/memkit) for memory management.

Models are loaded from binary **`.nk`** files (single-file architecture + weights). Convert from ONNX with `python -m netkit convert`. **Inference is float32-only today**; float16, int16, int8, and int4 are on the roadmap — see [docs/DATATYPES.md](docs/DATATYPES.md).

## Documentation

| Guide | Description |
|-------|-------------|
| **[Getting Started](docs/GETTING_STARTED.md)** | Build, test, and first inference in minutes |
| **[API Overview](docs/API.md)** | C vs C++ APIs, linking, memory model |
| **[Arena Memory](docs/ARENA.md)** | Bump allocator — sizing, alignment, reset |
| **[Data Types](docs/DATATYPES.md)** | Float32 today; float16 / int8 roadmap |
| **[ONNX Import](docs/ONNX.md)** | C++ ONNX loader for parity tests; packager converts ONNX → `.nk` |
| **[CLI Reference](docs/CLI.md)** | `test`, `run`, and `inspect` commands |
| **[Binary .nk Format](docs/NK_FORMAT.md)** | Single-file models — Python packager + C++ loader |
| **[Python packager](python/README.md)** | `python -m netkit convert` (ONNX → `.nk`) |
| **[Testing](docs/TESTING.md)** | Regression suites, Make targets, CI |
| **[C++ API Reference](docs/cpp-api.md)** | Headers in `include/` (C++26) |
| **[API Parity Policy](docs/API_PARITY.md)** | C ↔ C++ symbol map and contribution rules |
| **[MNIST MLP Test](docs/MNIST.md)** | Trained 784→128→10 MLP on handwritten digits |
| **[MNIST CNN Test](docs/MNIST_CNN.md)** | Tutorial-style conv+pool CNN on MNIST |
| **[MLP Background](docs/nn.md)** | Optional theory (training/backprop); netkit is inference-only |

## Language standards

| Code | Standard | Role |
|------|----------|------|
| C++ engine | **C++26** | All implementation, primary API, CLI, C++ tests |
| C API | **C23** | `netkit.h` bridge + `tests/test_c_api.c` |

Application code is C++26. C23 is limited to the C header, the `extern "C"` bridge (`src/netkit_api.cpp`), and the C API test harness.

## Features

- **Dual API** — C23 (`netkit.h`) and C++26 (native headers)
- **CLI** — `test`, `run`, and `inspect` commands for desktop development
- **MLP & CNN** — High-level network abstractions with `.nk` loading
- **Arena allocator** — Bump-pointer memory with aligned allocation (no heap in layer paths)
- **Regression tests** — hand-checked plus MNIST MLP and CNN (72 cases via `make test`)
- **Float32 inference** — all tensors, weights, and math use IEEE-754 single precision (`float`)

## Quick start

```bash
make              # build netkit CLI + libnetkit.a
make test         # C++ API tests + C API tests
./netkit run models/test_mlp.nk --input 1,2
make example-cpp    # C++26 usage demo
make example-c      # C23 usage demo
```

See [Getting Started](docs/GETTING_STARTED.md) for full details.

## CLI

Full reference: [docs/CLI.md](docs/CLI.md)

```bash
./netkit help                              # print usage (-h / --help also work)
./netkit test                              # C++ API regression suite
./netkit run models/test_mlp.nk --input 1,2
./netkit inspect models/test_mlp.nk      # boxed network summary
./netkit inspect models/test_mlp.nk --full   # + arena sizing after forward
```

## Examples

| Demo | Language | Build | Run |
|------|----------|-------|-----|
| `examples/infer_cpp.cpp` | C++26 | `make example-cpp` | `./examples/infer_cpp models/test_mlp.nk 1 2` |
| `examples/infer_c.c` | C23 | `make example-c` | `./examples/infer_c models/test_mlp.nk 1 2` |

Both load a `.nk` model and print input/output tensors. See [Getting Started](docs/GETTING_STARTED.md) for minimal code snippets and linking.

## Project structure

```
netkit/
├── include/
│   ├── netkit.h            # C23 public API
│   ├── arena.hpp           # Memory management
│   ├── tensor.hpp          # Tensor definitions
│   ├── mlp.hpp / cnn.hpp   # Network abstractions
│   ├── nk_loader.hpp       # .nk model loader
│   └── ...
├── src/                    # C++26 implementation
├── python/netkit/          # ONNX → .nk packager
├── examples/
│   ├── infer_cpp.cpp       # C++26 usage example
│   └── infer_c.c           # C23 usage example
├── tests/
│   └── test_c_api.c        # C23 API regression tests
├── models/                 # bundled .nk models + matching .onnx sources
├── tools/
│   ├── export_mnist_mlp.py
│   └── export_mnist_cnn.py
└── docs/                   # Guides and API reference
    ├── TESTING.md
    ├── GETTING_STARTED.md
    ├── NK_FORMAT.md
    ├── c-api.md / cpp-api.md
    └── API_PARITY.md
```

## Model files

| File | Purpose |
|------|---------|
| `model.nk` | Single-file model (architecture + float32 weights) |
| `model.onnx` | Source graph for `python -m netkit convert` |
| Embedded tests (optional) | Regression cases in `.nk` `TCAS` section — see [NK_FORMAT.md](docs/NK_FORMAT.md) |

Regenerate `.nk` from ONNX: `make export-nk`. Arena buffer size is **not** in the model file — you provide a caller-owned buffer sized for weights + ping-pong activations. See [docs/ARENA.md](docs/ARENA.md).

Format spec: [docs/NK_FORMAT.md](docs/NK_FORMAT.md). Regression tests: [docs/TESTING.md](docs/TESTING.md).

## Building

### Requirements

- C++26 compiler (clang++ 17+, g++ 14+)
- C23 compiler for C examples (clang 17+, gcc 14+)
- Make

### Targets

```bash
make              # netkit CLI + libnetkit.a
make build-all    # netkit + examples + C API test binary
make test         # C++ API tests + C API tests (72 regression cases)
make test-cpp     # C++ API regression only
make test-c       # C API regression only
make example-cpp  # C++26 usage demo
make example-c    # C23 usage demo
make export-mnist # regenerate MNIST MLP model (requires numpy)
make export-mnist-cnn # regenerate MNIST CNN model (requires numpy)
make clean
make rebuild
```

See [docs/TESTING.md](docs/TESTING.md) for the full regression layout.

## Testing

Full guide: [docs/TESTING.md](docs/TESTING.md)

```bash
make test       # C++ API tests, then C API tests
make test-cpp   # ./netkit test
make test-c     # ./tests/test_c_api
```

| Suite | Language | Entry point | Inference cases |
|-------|----------|-------------|-----------------|
| C++ API | C++26 | `./netkit test` → `src/test.cpp` | 72 (16 hand + 10 MNIST MLP + 10 MNIST CNN + 36 ONNX parity) |
| C API | C23 | `tests/test_c_api.c` | Same 72 + API smoke tests |

Regression cases are embedded in each bundled `.nk` file ([NK_FORMAT.md](docs/NK_FORMAT.md)).  
MNIST MLP: [MNIST.md](docs/MNIST.md). MNIST CNN: [MNIST_CNN.md](docs/MNIST_CNN.md).

## Design principles

- **Lightweight** — Standard C/C++ only, no external dependencies
- **Memory-conscious** — Arena bump allocator with explicit alignment; caller-owned backing buffer
- **Single-threaded** — Sequential forward pass
- **Inference-only** — No training

## Roadmap

- Max/average pooling (max pool supported in CNN pipelines; avg pool not yet)
- Conv padding
- Batch normalization
- Quantization (int8, uint8)

## License

MIT — see [LICENSE](LICENSE).

## Contributing

- C++ sources: C++26
- C sources and `netkit.h`: C23
- All tests must pass (`make test`)
- Update docs when changing public API
- **New C++ public API requires a matching C entry in `netkit.h`** — see [API_PARITY.md](docs/API_PARITY.md)
