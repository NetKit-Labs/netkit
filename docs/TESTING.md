# Testing

netkit uses **Make** as the primary build and test driver (no CMake required). All regression tests run through `./netkit test` and the C API harness `tests/test_c_api`.

## Quick commands

```bash
make              # netkit CLI + libnetkit.a (default)
make build-all    # netkit + examples + C API test binary
make test         # full regression (C++ then C API)
make test-cpp     # ./netkit test only
make test-c       # ./tests/test_c_api only
make clean        # remove objects and binaries
make rebuild      # clean + make
```

## Regression suites

Both `make test-cpp` and `make test-c` exercise the **same 18 inference cases** via `run_all_tests()` / `nk_run_all_tests()`:

| Suite | Cases | Source | Description |
|-------|------:|--------|-------------|
| Hand MLP vectors | 4 | `models/test_mlp.vectors.json`, `models/mlp_hand.vectors.json` | Small hand-checked MLP forwards |
| Hand CNN vectors | 4 | `models/test_cnn.vectors.json`, `models/cnn_4x4_single.vectors.json`, `models/cnn_hand.vectors.json` | Small hand-checked CNN forwards |
| MNIST MLP | 10 | `models/mnist/manifest.json` | Real 28×28 digits, trained 784→128→10 MLP |

**Total: 18 passed** when healthy (`8` hand vector + `10` MNIST).

Hand vector format: [VECTORS_TESTS.md](VECTORS_TESTS.md)  
MNIST bundle and training: [MNIST.md](MNIST.md)

## C++ API suite (`make test-cpp`)

Entry: `./netkit test` → `run_all_tests()` in `src/test.cpp`.

Sections printed in order:

1. **MLP TESTS** — hand `*.vectors.json` files  
2. **CNN TESTS** — hand `*.vectors.json` files  
3. **MNIST MLP TESTS** — `run_mnist_tests()` in `src/test_mnist.cpp`

## C API suite (`make test-c`)

Entry: `./tests/test_c_api` (C23).

1. Direct smoke tests — arena, tensor, ops, model load/run  
2. Full regression — `nk_run_all_tests()` (same 18 cases as above)

## Adding tests

| Kind | How |
|------|-----|
| Hand vector case | Edit `models/*.vectors.json`, register file in `src/test.cpp` if new bundle |
| MNIST case | Regenerate with `make export-mnist` (requires numpy + MNIST CSV or download) |

Always run `make test` before committing.

## Regenerating MNIST assets

```bash
make export-mnist   # python3 tools/export_mnist_mlp.py
```

Requires **numpy**. Uses `../python/mnist/*.csv` when present, else downloads IDX gzip files to `data/mnist/`. Commit updated `models/mnist_mlp.*` and `models/mnist/` for CI.

## CI

GitHub Actions (`.github/workflows/ci.yml`): `make`, `make test`, example smoke tests, CLI smoke tests. MNIST weights and case files are committed so CI needs no network or Python.
