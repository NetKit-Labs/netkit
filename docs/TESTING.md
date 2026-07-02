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

Both `make test-cpp` and `make test-c` exercise the **same 72 inference cases** via `run_all_tests()` / `nk_run_all_tests()`:

| Suite | Cases | Source | Description |
|-------|------:|--------|-------------|
| Hand MLP | 9 | `models/test_mlp.nk`, `models/mlp_hand.nk` | Small hand-checked MLP forwards |
| Hand CNN | 7 | `models/test_cnn.nk`, `models/cnn_4x4_single.nk`, `models/cnn_hand.nk` | Small hand-checked CNN forwards |
| MNIST MLP | 10 | `models/mnist_mlp.nk` | Trained 784→128→10 MLP (98.06% test acc) |
| MNIST CNN | 10 | `models/mnist_cnn.nk` | Conv+pool+flatten+dense CNN (99.02% test acc) |
| ONNX parity | 36 | embedded `.nk` cases vs matching `.onnx` | Same inputs: `.nk` forward must match ONNX import forward |

**Total: 72 passed** when healthy (`16` hand + `10` MNIST MLP + `10` MNIST CNN + `36` ONNX parity).

Test cases are **embedded in each `.nk` file** (optional `TCAS` section). See [NK_FORMAT.md](NK_FORMAT.md).

| Doc | Contents |
|-----|----------|
| [NK_FORMAT.md](NK_FORMAT.md) | `.nk` layout + embedded regression tests |
| [MNIST.md](MNIST.md) | MNIST MLP model |
| [MNIST_CNN.md](MNIST_CNN.md) | MNIST CNN model |

## Arena buffers in tests

All regression paths use an arena; only the **backing buffer size** varies:

| Harness | Source | Arena size | Models |
|---------|--------|------------|--------|
| Hand tests | `src/nk_regression.cpp` | **64 KiB** | hand `.nk` models |
| MNIST MLP | `src/nk_regression.cpp` | **2 MiB** | `mnist_mlp.nk` |
| MNIST CNN | `src/nk_regression.cpp` | **4 MiB** | `mnist_cnn.nk` |
| ONNX parity | `src/test_onnx.cpp` | **64 KiB / 2 MiB / 4 MiB** | hand / MNIST `.nk` + `.onnx` |
| C API smoke / unit tests | `tests/test_c_api.c` | **64 KiB** | hand models + parse/load smoke |
| CLI `run` / `inspect` | `src/cli.cpp` | **64 KiB** | hand models (MNIST may overflow `--full`) |

The test code does not read arena size from the model file — constants are chosen so weights + ping-pong activation buffers fit. See [ARENA.md](ARENA.md) for sizing your own firmware buffer.

## C++ API suite (`make test-cpp`)

Entry: `./netkit test` → `run_all_tests()` in `src/test.cpp`.

Sections printed in order:

1. **MLP TESTS** — hand `.nk` models with embedded cases  
2. **CNN TESTS** — hand `.nk` models with embedded cases  
3. **MNIST MLP TESTS** — `models/mnist_mlp.nk`  
4. **MNIST CNN TESTS** — `models/mnist_cnn.nk`  
5. **ONNX PARITY TESTS** — `run_onnx_import_tests()` in `src/test_onnx.cpp`

## Test output

**Hand cases** print the input tensor, then a per-output line (`out[i]: actual=… expected=…`) so small models show meaningful numeric checks.

**MNIST cases** print predicted class, winner softmax probability, and any runner-up outputs above `0.01`. All outputs are compared internally within tolerance.

## C API suite (`make test-c`)

Entry: `./tests/test_c_api` (C23).

| Phase | What it covers |
|-------|----------------|
| Arena | init, aligned alloc, reset, capacity |
| Tensor / ops | create, matmul, activations |
| Parse architecture | MLP and CNN `.nk` metadata |
| Model load / run | `nk_model_load` + `nk_model_run` on hand MLP/CNN |
| Hybrid CNN | `nk_parse_architecture` + `nk_cnn_load` on `mnist_cnn.nk` |
| Full regression | `nk_run_all_tests()` — same **72** inference cases as C++ |

The C API regression path uses the same C++ runner internally (`nk_run_all_tests` → `run_all_tests`).

## Adding tests

| Kind | How |
|------|-----|
| Hand case | Add to `python/netkit/regression_data.py`, run `make embed-tests`, register `.nk` in `src/test.cpp` if new bundle |
| ONNX parity case | Add matching `models/<name>.onnx`, convert with `make export-nk`, register pair in `src/test_onnx.cpp` |
| MNIST MLP case | `make export-mnist` (requires numpy) |
| MNIST CNN case | `make export-mnist-cnn` (requires numpy) |

Always run `make test` before committing.

## Regenerating models

Weights and embedded tests are **committed** so CI never trains. Regenerate only when architecture or training changes:

```bash
make export-mnist       # MLP — full 60k, 40 epochs (~8s)
make export-mnist-cnn   # CNN — full 60k, 20 epochs (~18 min)
make export-mnist-all   # both + refresh ONNX from .nk
make export-nk          # ONNX → .nk + embed hand tests
make embed-tests        # re-embed hand tests from regression_data.py
```

Requires **numpy**. Uses `../python/mnist/*.csv` when present, else downloads IDX files to `data/mnist/`.

## CI

GitHub Actions (`.github/workflows/ci.yml`): `make`, `make test`, example smoke tests, CLI smoke tests. All model weights and embedded test cases are in the repo — no network or Python in CI.
