# Build Targets

netkit builds for three deployment profiles. Select one with **`NETKIT_TARGET`** when invoking Make, or define the matching `-D` flag in your embedded toolchain.

| Target | Makefile | Role | What's included |
|--------|----------|------|-----------------|
| **CPU** | `NETKIT_TARGET=cpu` (default) | Desktop dev, debug, CI | Lean runtime **plus** CLI, embedded regression tests, future debug/analysis tooling |
| **MCU** | `NETKIT_TARGET=mcu` | Microcontroller firmware | Lean runtime only — `.nk` load, arena, tensors, kernel ops, inference |
| **MPU** | `NETKIT_TARGET=mpu` | Microprocessor / RTOS | Same lean runtime as MCU |

## Arena backing defaults

| Target | Default arena | Override flag |
|--------|---------------|---------------|
| **CPU** | **Heap** (`malloc` backing via `nk_arena_init_heap` / `Arena::init_heap`) | `NETKIT_GLOBAL_ARENA=1` → static/global buffer only |
| **MCU** | **Global/static** (`nk_arena_init` with your buffer) | `NETKIT_HEAP_ARENA=1` → also compile heap helpers |
| **MPU** | **Global/static** (same as MCU) | `NETKIT_HEAP_ARENA=1` → also compile heap helpers |

Compile-time macros (from `include/netkit_config.h`):

| Macro | Meaning |
|-------|---------|
| `NETKIT_TARGET_CPU` | Desktop / CPU build |
| `NETKIT_TARGET_MCU` | MCU build |
| `NETKIT_TARGET_MPU` | MPU build |
| `NETKIT_DESKTOP` | CPU only — CLI, regression, debug tooling |
| `NETKIT_ARENA_HEAP` | Heap arena API compiled in (CPU default; MCU/MPU when opted in) |
| `NETKIT_GLOBAL_ARENA` | CPU only — force global/static arena instead of heap default |

Default arena constant (`NK_ARENA_DEFAULT_CAPACITY` / `Arena::kDefaultCapacity`):

| Target | Default |
|--------|---------|
| **CPU** | **4 MiB** (MNIST CNN-capable; examples and C API smoke tests) |
| **MCU** | **64 KiB** |
| **MPU** | **128 KiB** |

CLI/regression on CPU still use **model-sized heap** via `ArenaUtil::CapacityForInputElements` (64 KiB hand / 2 MiB MNIST MLP / 4 MiB MNIST CNN).

## Quick commands

```bash
# Desktop (default) — CLI + tests + libnetkit.a, heap arena default
make
make test

# CPU with static/global arena (firmware-style backing on desktop)
make NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1 all

# Lean runtime libraries for firmware
make NETKIT_TARGET=mcu lib
make NETKIT_TARGET=mpu lib

# MCU/MPU with optional heap arena API
make NETKIT_TARGET=mcu NETKIT_HEAP_ARENA=1 lib
make NETKIT_TARGET=mpu NETKIT_HEAP_ARENA=1 lib

# Convenience aliases
make cpu              # NETKIT_TARGET=cpu (heap default)
make cpu-global       # NETKIT_TARGET=cpu NETKIT_GLOBAL_ARENA=1
make mcu              # NETKIT_TARGET=mcu lib
make mcu-heap         # NETKIT_TARGET=mcu NETKIT_HEAP_ARENA=1 lib
make mpu              # NETKIT_TARGET=mpu lib
make mpu-heap         # NETKIT_TARGET=mpu NETKIT_HEAP_ARENA=1 lib
```

## CPU (desktop)

Use for local development, CI, and the **`netkit` CLI**.

**Build outputs:**

- `netkit` — CLI (`test`, `run`, `inspect`)
- `libnetkit.a` — runtime + desktop extras
- `examples/infer_cpp`, `examples/infer_c`, `tests/test_c_api`

**Desktop-only** (guarded by `NETKIT_DESKTOP`):

- `Cli::Run` / `nk_cli_run`
- `run_all_tests` / `nk_run_all_tests`
- Future tensor analysis / debug tooling

**Arena:** CLI and regression use **heap** with model-sized buffers (64 KiB hand / 2 MiB MNIST MLP / 4 MiB MNIST CNN). Examples and C API smoke tests use **`NK_ARENA_DEFAULT_CAPACITY` (4 MiB)**. Build with `NETKIT_GLOBAL_ARENA=1` to use a static buffer instead of heap.

See [CLI.md](CLI.md).

## MCU (lean runtime)

Inference-only library for microcontrollers. **No CLI, no regression runner.**

Default arena: caller-owned static or global buffer sized with **`NK_ARENA_DEFAULT_CAPACITY` (64 KiB)**:

```c
alignas(max_align_t) static unsigned char arena_mem[65536];
nk_arena_t arena;
nk_arena_init(&arena, arena_mem, sizeof(arena_mem));
```

Optional heap arena when built with `NETKIT_HEAP_ARENA=1` — **one** `malloc` at startup, **never** freed:

```c
nk_arena_init_heap(&arena, capacity);
/* ... entire firmware lifetime ... */
/* nk_arena_destroy_heap() is a no-op on MCU/MPU */
```

**Build output:** `libnetkit.a` only.

## MPU (lean runtime)

Same lean runtime as MCU. Default static arena constant is **`NK_ARENA_DEFAULT_CAPACITY` (128 KiB)** — use a caller-owned buffer of at least that size for small models, or size up with `nk_inspect_model()` for larger graphs.

## Source split

| Component | CPU | MCU / MPU |
|-----------|:---:|:---------:|
| Arena, tensor, ops, MLP, CNN | ✓ | ✓ |
| `.nk` loader (`nk_loader`, `nk_format`) | ✓ | ✓ |
| C API bridge (`netkit_api.cpp`) | ✓ | ✓ |
| CLI (`cli.cpp`, `main.cpp`) | ✓ | — |
| Regression (`test.cpp`, `nk_regression.cpp`) | ✓ | — |

## Cross-compilation

The Makefile uses host `clang`/`clang++`. For firmware, either:

1. Build `libnetkit.a` with `NETKIT_TARGET=mcu` or `mpu` and link into your board project, **or**
2. Add the runtime `.cpp` sources to your board build with `-DNETKIT_TARGET_MCU=1` (or `MPU`) and `-std=c++26 -Iinclude`.

## Testing

Full regression (`make test` — 36 embedded C++ cases + 26 Python ONNX parity) requires **`NETKIT_TARGET=cpu`**. Validate on desktop first, then run device smoke tests with the lean MCU/MPU library.

See [TESTING.md](TESTING.md) and [ARENA.md](ARENA.md).

Related: [PHILOSOPHY.md](PHILOSOPHY.md) · [GETTING_STARTED.md](GETTING_STARTED.md)
