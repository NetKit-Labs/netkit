# Arena Memory (Bump Allocator)

netkit uses a **single bump-pointer arena** for all inference-time allocation: network structs, weight blobs loaded from `.bin` files, and intermediate tensors during forward passes. There is no per-object `free()` — memory is reclaimed in bulk with `reset()`.

## Why an arena?

Embedded and firmware targets benefit from:

- **Predictable memory** — one caller-provided buffer, no hidden heap use in layer code paths
- **Fast allocation** — pointer bump only (O(1) per alloc)
- **Simple lifetime** — reset between inferences or test cases instead of tracking individual frees

netkit ships a minimal ~86-line arena rather than linking an external allocator (see [API.md](API.md#memory-model) for the memkit comparison).

## How it works

```
base ──► [ used ........ | free ........................ ] ◄── capacity
         ▲
         offset (next alloc starts here)
```

1. **`init(memory, size)`** — bind a caller-owned byte buffer; offset = 0.
2. **`alloc(size, alignment)`** — if current offset is not aligned, skip padding bytes; carve `size` bytes; advance offset; return pointer (or `nullptr` on overflow).
3. **`reset()`** — offset = 0; all prior pointers are logically invalid.
4. **`remaining()`** — `capacity - offset`.

### Alignment

Weight `.bin` files can have an odd float count, leaving the offset at 4 mod 8 on 64-bit platforms. Without padding, a following `MLPNetwork` or `CNNNetwork` struct would be misaligned for placement-new.

| Allocation | Typical alignment |
|------------|-------------------|
| float weights / tensor payload | `alignof(float)` (4) |
| Network structs, pointers | `alignof(T)` or `alignof(max_align_t)` (8 on 64-bit) |

The engine passes correct alignment at every internal call site. Direct API users must do the same.

### Backing buffer

Declare the buffer with platform max alignment:

```c
alignas(max_align_t) static unsigned char memory[65536];
```

```cpp
alignas(std::max_align_t) unsigned char buffer[65536];
```

## C++ API

```cpp
#include "arena.hpp"

Arena arena;
arena.init(buffer, sizeof(buffer));

void* weights = arena.alloc(weight_bytes, alignof(float));
void* net_mem = arena.alloc(sizeof(CNNNetwork), alignof(CNNNetwork));

arena.reset();  // reuse for next inference
```

Default capacity constant: `Arena::kDefaultCapacity` (64 KiB). MNIST models need larger buffers — see `./netkit inspect` or [MNIST.md](MNIST.md).

## C API

```c
#include "netkit.h"

alignas(max_align_t) static unsigned char memory[NK_ARENA_DEFAULT_CAPACITY];
nk_arena_t arena;
nk_arena_init(&arena, memory, sizeof(memory));

void* block = nk_arena_alloc(&arena, 1024, alignof(float));
nk_arena_reset(&arena);
```

| Function | C++ equivalent |
|----------|----------------|
| `nk_arena_init` | `Arena::init` |
| `nk_arena_alloc` | `Arena::alloc` |
| `nk_arena_reset` | `Arena::reset` |
| `nk_arena_capacity` | `Arena::capacity` |
| `nk_arena_used` | `Arena::offset` |
| `nk_arena_remaining` | `Arena::remaining` |

High-level loaders (`nk_model_load`, `nk_mlp_load`, `nk_cnn_load`) allocate from the arena you pass in. Size buffers with `nk_inspect_model()` or `./netkit inspect`.

## Sizing for deployment

1. Run `./netkit inspect models/your_model.json --full` (or `nk_inspect_model`).
2. Note **arena bytes after forward** — add headroom (typically 1.5–2× for batch variance).
3. Use one arena per model context, or `reset()` between runs on the same buffer.

| Model | Approx. arena high-water |
|-------|--------------------------|
| Hand test MLP/CNN | < 64 KiB |
| MNIST MLP | ~2 MiB (test suite) |
| MNIST CNN | ~4 MiB (test suite) |

## Related docs

- [DATATYPES.md](DATATYPES.md) — float32 weights and tensors today
- [c-api.md](c-api.md) — full C arena reference
- [cpp-api.md](cpp-api.md) — C++ arena reference
- [API.md](API.md#memory-model) — overview and memkit note
