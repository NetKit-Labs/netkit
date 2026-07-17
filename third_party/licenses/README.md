# Vendored license texts

Copies of third-party license files for components used by netkit (runtime
backends, Python tooling, and peer-benchmark stacks). See
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for roles, copyright
holders, and redistribution notes.

These files are mirrored so attribution remains available when large fetch trees
(`third_party/XNNPACK/`, TFLM downloads, ORT / TVM checkouts) are not present.
Refresh from the corresponding upstream `LICENSE` after bumping pins.

| Prefix / path | Source |
|---------------|--------|
| `CMSIS-*`, `XNNPACK*`, `pthreadpool*`, … | Runtime backends |
| `onnx*`, `onnxruntime*` | Python tooling / host peer |
| `tflite-micro*`, `flatbuffers*`, `gemmlowp*`, `ruy*` | TFLM peer |
| `apache-tvm.Apache-2.0.txt`, `apache-tvm.NOTICE.txt` | Apache TVM / microTVM peer |
| `apache-tvm/` | Upstream TVM `licenses/` bundle |
| `apache-tvm-3rdparty-dmlc-core*` | TVM `3rdparty/dmlc-core` |
