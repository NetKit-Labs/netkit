# netkit-tools (Python)

Convert ONNX models into binary **`.nk`** files for the C++ runtime.

## Install

```bash
pip install -e python
```

Requires **numpy** and **onnx**.

## Usage

```bash
# ONNX -> .nk
python -m netkit convert models/test_mlp.onnx -o models/test_mlp.nk

# Inspect header + tensor catalog
python -m netkit inspect models/test_mlp.nk

# Convert all bundled regression models (from repo root)
make export-nk
```

## C++ runtime

```bash
./netkit inspect models/test_mlp.nk
./netkit run models/test_mlp.nk --input 1,2
```

See [docs/NK_FORMAT.md](../docs/NK_FORMAT.md) for the binary layout.
