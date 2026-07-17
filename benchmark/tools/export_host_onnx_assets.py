#!/usr/bin/env python3
"""Export / quantize ONNX assets for the host three-way suite (netkit / TF / ORT).

Float32:
  - models/mnist_cnn.onnx, models/mnist_cnn_dw.onnx (already committed)
  - models/mobilenetv4_imagenet_f32.onnx (timm → ONNX, kept next to .nk)

Int8 (QDQ, float I/O — ORT static quant; internal int8 ops for XNNPACK):
  - models/mnist_cnn_int8.onnx
  - models/mnist_cnn_dw_int8.onnx
  - models/mobilenetv4_imagenet_int8.onnx

Usage:
  python3 benchmark/tools/export_host_onnx_assets.py --float32
  python3 benchmark/tools/export_host_onnx_assets.py --int8
  python3 benchmark/tools/export_host_onnx_assets.py --all
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmark" / "tflm" / "tools"
CACHE = ROOT / "benchmark" / "tflm" / "generated" / "imagenet_sample_cache"
ORT_MODELS = ROOT / "benchmark" / "onnxruntime" / "models"
# ORT 1.20 max ONNX IR version.
ORT_MAX_IR = 10


def _copy_for_ort(src: Path, dest_name: str | None = None) -> Path:
    """Copy ONNX into benchmark/onnxruntime/models with IR ≤ ORT_MAX_IR."""
    import onnx

    ORT_MODELS.mkdir(parents=True, exist_ok=True)
    dest = ORT_MODELS / (dest_name or src.name)
    m = onnx.load(str(src))
    if m.ir_version > ORT_MAX_IR:
        m.ir_version = ORT_MAX_IR
    onnx.save(m, str(dest))
    print(f"ORT asset {dest} (ir={onnx.load(str(dest)).ir_version})")
    return dest


def _export_imagenet_f32(out: Path) -> None:
    sys.path.insert(0, str(ROOT / "python"))
    import torch
    from netkit.torch_backbone_pack import load_backbone_model

    model = load_backbone_model("mobilenetv4_small", num_classes=1000, pretrained=True)
    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    print(f"exporting {out} ...")
    try:
        torch.onnx.export(
            model,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
        )
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


def _mnist_calib(nk_path: Path, *, nchw: bool = True):
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(TOOLS))
    from export_int8_test_images import NUM_IMAGES, _one_per_digit_cases
    from netkit.reader import read_test_suite

    suite = read_test_suite(nk_path)
    if suite is None:
        raise SystemExit(f"missing TCAS in {nk_path}")
    cases = _one_per_digit_cases(suite.cases, num=NUM_IMAGES)
    for case in cases:
        pixels = np.asarray(case.input, dtype=np.float32).reshape(-1)
        if nchw:
            yield {"input": pixels.reshape(1, 1, 28, 28)}
        else:
            yield {"input": pixels.reshape(1, 28, 28, 1)}


def _imagenet_calib():
    sys.path.insert(0, str(TOOLS))
    from export_imagenet_mnv4_test_images import SAMPLES, preprocess
    from PIL import Image

    for filename, _url, _label, _short in SAMPLES:
        path = CACHE / filename
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        nhwc = np.asarray(preprocess(rgb), dtype=np.float32)
        if nhwc.ndim == 3:
            nhwc = nhwc[None, ...]
        nchw = np.transpose(nhwc, (0, 3, 1, 2))
        yield {"input": nchw}


def _quantize_static(model_fp32: Path, model_int8: Path, data_iter) -> None:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class _Reader(CalibrationDataReader):
        def __init__(self, it):
            self._data = list(it)
            self._i = 0

        def get_next(self):
            if self._i >= len(self._data):
                return None
            item = self._data[self._i]
            self._i += 1
            return item

    if not model_fp32.is_file():
        raise SystemExit(f"missing {model_fp32}")
    model_int8.parent.mkdir(parents=True, exist_ok=True)
    # Resolve actual input name from the model.
    import onnx

    graph_in = onnx.load(str(model_fp32)).graph.input[0].name
    reader_data = []
    for item in data_iter:
        # remap generic "input" key if needed
        if graph_in in item:
            reader_data.append(item)
        elif "input" in item:
            reader_data.append({graph_in: item["input"]})
        else:
            reader_data.append({graph_in: next(iter(item.values()))})

    print(f"quantizing {model_fp32.name} -> {model_int8.name} ({len(reader_data)} calib) ...")
    with tempfile.TemporaryDirectory(prefix="ort_q_") as tmp:
        tmp_out = Path(tmp) / model_int8.name
        quantize_static(
            model_input=str(model_fp32),
            model_output=str(tmp_out),
            calibration_data_reader=_Reader(reader_data),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        shutil.copy2(tmp_out, model_int8)
    print(f"Wrote {model_int8} ({model_int8.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all or not (args.float32 or args.int8):
        args.float32 = True
        args.int8 = True

    if args.float32:
        for p in (ROOT / "models" / "mnist_cnn.onnx", ROOT / "models" / "mnist_cnn_dw.onnx"):
            if not p.is_file():
                raise SystemExit(f"missing {p}")
            _copy_for_ort(p)
        imagenet = ROOT / "models" / "mobilenetv4_imagenet_f32.onnx"
        _export_imagenet_f32(imagenet)
        _copy_for_ort(imagenet)

    if args.int8:
        # Prefer ORT from the LiteRT-matched venv if present.
        ort_py = ROOT / "benchmark" / "onnxruntime" / ".venv" / "bin" / "python"
        if ort_py.is_file():
            import subprocess

            if Path(sys.executable).resolve() != ort_py.resolve():
                return subprocess.call([str(ort_py), __file__, "--int8"])

        # Quantize IR-capped float copies so ORT 1.20 can load the QDQ graphs.
        cnn_f = _copy_for_ort(ROOT / "models" / "mnist_cnn.onnx")
        dw_f = _copy_for_ort(ROOT / "models" / "mnist_cnn_dw.onnx")
        _quantize_static(
            cnn_f,
            ORT_MODELS / "mnist_cnn_int8.onnx",
            _mnist_calib(ROOT / "models" / "mnist_cnn.nk"),
        )
        _quantize_static(
            dw_f,
            ORT_MODELS / "mnist_cnn_dw_int8.onnx",
            _mnist_calib(ROOT / "models" / "mnist_cnn_dw.nk"),
        )
        f32 = ROOT / "models" / "mobilenetv4_imagenet_f32.onnx"
        if not f32.is_file():
            _export_imagenet_f32(f32)
        f32_ort = _copy_for_ort(f32)
        _quantize_static(
            f32_ort,
            ORT_MODELS / "mobilenetv4_imagenet_int8.onnx",
            _imagenet_calib(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
