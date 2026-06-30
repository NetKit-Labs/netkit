# netkit — Neural Network Kit

netkit is a C++26 neural network kit for on-device inference on MCUs and MPUs. It is developed and validated on the desktop, then deployed to embedded targets. Companion to [memkit](https://github.com/jameslavrenz/memkit) for memory management.

Models are loaded from JSON architecture files and companion float32 `.bin` weight files.

## Documentation

| Guide | Description |
|-------|-------------|
| **[Getting Started](docs/GETTING_STARTED.md)** | Build, test, and first inference in minutes |
| **[API Overview](docs/API.md)** | C vs C++ APIs, CLI, linking |
| **[C API Reference](docs/c-api.md)** | `netkit.h` (C23) |
| **[C++ API Reference](docs/cpp-api.md)** | Headers in `include/` (C++26) |

## Language standards

| Code | Standard | Location |
|------|----------|----------|
| C++ engine | **C++26** | `src/*.cpp`, `include/*.hpp` |
| C API | **C23** | `include/netkit.h`, `examples/infer_c.c` |

The C API is a thin `extern "C"` bridge (`src/netkit_api.cpp`) over the C++ core. Link with `libnetkit.a`.

## Features

- **Dual API** — C23 (`netkit.h`) and C++26 (native headers)
- **CLI** — `test`, `run`, and `inspect` commands for desktop development
- **MLP & CNN** — High-level network abstractions with JSON + `.bin` loading
- **Arena allocator** — Bump-pointer memory management (no heap in layer paths)
- **Vectors test runner** — Declarative regression tests in `*.vectors.json`
- **Activations** — ReLU, LeakyReLU, ReLU6, Sigmoid, Tanh, Softmax

## Quick start

```bash
make              # build netkit CLI + libnetkit.a
make run          # run 8 regression tests
./netkit run models/test_mlp.json --input 1,2
make example-c    # build C23 example
./examples/infer_c models/test_mlp.json 1 2
```

See [Getting Started](docs/GETTING_STARTED.md) for full details.

## CLI

```bash
./netkit test                              # regression suite
./netkit run models/test_mlp.json --input 1,2
./netkit inspect models/test_mlp.json      # architecture + arena sizing
```

## Project structure

```
netkit/
├── include/
│   ├── netkit.h            # C23 public API
│   ├── arena.hpp           # Memory management
│   ├── tensor.hpp          # Tensor definitions
│   ├── mlp.hpp / cnn.hpp   # Network abstractions
│   ├── model_loader.hpp    # JSON + .bin loader
│   └── ...
├── src/                    # C++26 implementation
├── examples/
│   └── infer_c.c           # C23 usage example
├── models/                 # JSON + bin + vectors bundles
├── tools/
│   └── write_hand_models.py
└── docs/                   # Guides and API reference
```

## Model file bundles

| File | Purpose |
|------|---------|
| `model.json` | Architecture (layers, activations, input shape) |
| `model.bin` | Raw float32 weights in layer order |
| `model.vectors.json` | Regression test cases (optional) |

## Building

### Requirements

- C++26 compiler (clang++ 17+, g++ 14+)
- C23 compiler for C examples (clang 17+, gcc 14+)
- Make

### Targets

```bash
make              # netkit CLI + libnetkit.a
make lib          # libnetkit.a only
make run          # build and run tests
make example-c    # C23 inference example
make clean
make rebuild
```

## Testing

```bash
make run
```

Eight vector-driven tests cover MLP and CNN forward passes. To add a test, create or extend a `models/*.vectors.json` file and register it in `src/test.cpp`.

## Design principles

- **Lightweight** — Standard C/C++ only, no external dependencies
- **Memory-conscious** — Arena allocator throughout
- **Single-threaded** — Sequential forward pass
- **Inference-only** — No training

## Roadmap

- Pooling (max, average) and conv padding
- Batch normalization
- Quantization (int8, uint8)
- Python model exporter

## License

[Add license information here]

## Contributing

- C++ sources: C++26
- C sources and `netkit.h`: C23
- All tests must pass (`make run`)
- Update docs when changing public API
