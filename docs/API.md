# API Overview

netkit exposes two language interfaces over the same inference engine:

| API | Header | Language | Use when |
|-----|--------|----------|----------|
| **C API** | `include/netkit.h` | C23 | Embedded firmware, FFI, minimal dependencies at the call site |
| **C++ API** | `include/*.hpp` | C++26 | Application code, tests, extending layers and ops |

Both APIs share:

- Bump-pointer **arena** memory management (no heap in layer code paths)
- **JSON + `.bin`** model loading
- **MLP** and **CNN** forward-only inference
- **NHWC** tensor layout for convolutions

## Quick comparison

### Load and run (C23)

```c
nk_arena_t arena;
nk_model_t model;
nk_arena_init(&arena, memory, size);
nk_model_load("models/test_mlp.json", &arena, &model);
nk_model_run(&model, &arena, input, n, output, cap, &out_n);
```

### Load and run (C++26)

```cpp
Arena arena;
arena.init(buffer, size);
MLPNetwork* net = nullptr;
ModelLoader::LoadMLP("models/test_mlp.json", arena, net, shape, rank);
net->forward(input, output, arena);
```

## CLI

The `netkit` binary is a desktop development tool (implemented in C++26):

| Command | Description |
|---------|-------------|
| `netkit test` | Run all `models/*.vectors.json` regression tests |
| `netkit run <model.json> --input a,b,c` | Single inference |
| `netkit inspect <model.json>` | Architecture, weights, arena sizing |

## Documentation map

| Document | Contents |
|----------|----------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Build, test, first inference |
| [c-api.md](c-api.md) | Full C23 reference (`netkit.h`) |
| [cpp-api.md](cpp-api.md) | Full C++26 reference (headers in `include/`) |

## Language standards

- All files under `src/*.cpp` and `include/*.hpp` are **C++26**.
- `include/netkit.h`, `examples/infer_c.c`, and any user C code should compile as **C23**.
- `src/netkit_api.cpp` is C++26 but exports `extern "C"` symbols declared in `netkit.h`.

## Linking

`libnetkit.a` contains C++ object code. Link C applications with a C++-aware linker:

```bash
clang++ -o app app.o libnetkit.a
```

## Error handling

| API | Pattern |
|-----|---------|
| C | Functions return `nk_status_t`; call `nk_last_error()` for detail |
| C++ | `ModelLoader::LoadResult` with `LoadStatus` and `message` |

## Memory model

Both APIs require a caller-provided buffer for the arena. Size it using `./netkit inspect` or `nk_inspect_model()`. When allocation fails, functions return an arena overflow error — there is no automatic growth.

Call `nk_arena_reset()` / `Arena::reset()` between inference batches to reuse the same buffer.

## Supported model format

- JSON `version` must be `1`
- `network`: `"mlp"` or `"cnn"`
- Activations: `none`, `relu`, `sigmoid`, `tanh`, `leaky_relu`, `relu6`, `softmax`
- Weights: float32 little-endian in companion `.bin` file

See [GETTING_STARTED.md](GETTING_STARTED.md#model-file-bundles) for weight layout details.
