# ONNX Runtime host peer

Third host CPU peer alongside **netkit** and **TF Lite / LiteRT**.

| Mode | netkit | TF Lite | ONNX Runtime |
|------|--------|---------|--------------|
| **xnn** | XNNPACK | XNNPACK (default) | `XNNPACKExecutionProvider` |
| **ref** | reference | `BUILTIN_REF` (intentionally slow) | `CPUExecutionProvider` (**MLAS**) |

**MLAS is not needed for netkit.** With XNNPACK ON (the production peer), netkit ≈ TF Lite and both beat ORT’s XNNPACK EP here — optimized TF Lite builtins and ORT MLAS are moot. ORT OFF never drops to a slow reference (MLAS stays on), so that column is not an apples-to-apples “ref” peer; TF Lite OFF is `BUILTIN_REF` (its slowest path).

## Build (required)

Stock `pip install onnxruntime` **does not** include the XNNPACK EP on desktop.
Build from source with LiteRT-matched compiler/linker flags:

```bash
./tools/build_onnxruntime_litert_matched.sh
```

Uses `gcc`/`g++`, `-O3 -DNDEBUG`, `-fpermissive`, and Darwin arm64 `-ld_classic`
(same policy as `benchmark/common/tflite_host_flags.mk`). Installs a wheel into
`benchmark/onnxruntime/.venv`.

## Assets

```bash
python3 benchmark/tools/export_host_onnx_assets.py --all
```

Writes IR≤10 copies under `benchmark/onnxruntime/models/` (ORT 1.20 max IR).
Do not point ORT at the repo-root `models/*.onnx` fixtures (IR 12).
`benchmark/onnxruntime/models/` is gitignored — regenerate locally.

## Suite

```bash
python3 benchmark/tools/run_host_ab_suite_float32.py
python3 benchmark/tools/run_host_ab_suite_int8.py
```

Int8 ORT models are **QDQ** (float graph I/O, int8 internals) from
`onnxruntime.quantization` — not the same binary as TF Lite int8 `.tflite`.
Latency is still a fair accel ON/OFF peer for XNNPACK; exact bit-match with TF Lite is not claimed.

Results and takeaways: [docs/STATUS.md](../../docs/STATUS.md#host-three-way-suite-netkit-vs-tf-lite-vs-onnx-runtime).
