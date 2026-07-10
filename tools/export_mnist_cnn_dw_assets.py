#!/usr/bin/env python3
"""Export depthwise MNIST CNN TF Lite peers + embedded test images.

Writes under benchmark/tflm/generated/:
  mnist_cnn_dw.tflite
  mnist_cnn_dw_int8.tflite
  cnn_dw/mnist_cnn_test_images.{h,cc}       (MnistCnn* symbols for mnist_cnn_main.cc)
  cnn_dw/mnist_cnn_int8_test_images.{h,cc}  (MnistCnnInt8* symbols for int8 main)
  mnist_cnn_dw_int8_tflite_quant.json

Prerequisites:
  models/mnist_cnn_dw.nk

Run from repo root:
  python3 tools/export_mnist_cnn_dw_assets.py
  python3 tools/export_mnist_cnn_dw_assets.py --float-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benchmark" / "tflm" / "tools"))

from export_assets import (  # noqa: E402
    GENERATED,
    NUM_CALIBRATION,
    NUM_IMAGES,
    _load_mnist_calibration,
    _one_per_digit_cases,
    _onnx2tf_bin,
    _write_test_image_arrays,
)
from export_int8_test_images import INPUT_SIZE, _tflite_input_quant  # noqa: E402

FLOAT_NK = ROOT / "models" / "mnist_cnn_dw.nk"
ONNX = ROOT / "models" / "mnist_cnn_dw.onnx"
FLOAT_TFLITE = GENERATED / "mnist_cnn_dw.tflite"
INT8_TFLITE = GENERATED / "mnist_cnn_dw_int8.tflite"
SAVED_MODEL = GENERATED / "cnn_dw_saved_model"
QUANT_JSON = GENERATED / "mnist_cnn_dw_int8_tflite_quant.json"
DW_GEN = GENERATED / "cnn_dw"

FLOAT_IMAGES = {
    "nk": FLOAT_NK,
    "images_h": DW_GEN / "mnist_cnn_test_images.h",
    "images_cc": DW_GEN / "mnist_cnn_test_images.cc",
    "input_size": 784,
    "prefix": "MnistCnn",
    "array_prefix": "kMnistCnn",
    "legacy_names": False,
}


def _export_tflite_via_onnx2tf(onnx_path: Path, out: Path, saved_model_dir: Path) -> None:
    if not onnx_path.is_file():
        raise SystemExit(f"missing {onnx_path}")

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    print(f"Converting {onnx_path.name} with onnx2tf (-osd) ...")
    # -osd keeps a SavedModel for full-int8 TFLiteConverter; without it newer
    # onnx2tf may emit only flatbuffer_direct .tflite files.
    subprocess.run(
        [
            _onnx2tf_bin(),
            "-i",
            str(onnx_path),
            "-o",
            str(saved_model_dir),
            "-nuo",
            "-osd",
        ],
        check=True,
    )

    candidates = sorted(saved_model_dir.glob("*float32.tflite"))
    if not candidates:
        raise SystemExit(f"onnx2tf did not emit a float32 .tflite under {saved_model_dir}")
    shutil.copy2(candidates[0], out)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")


def _write_int8_images() -> None:
    from netkit.quantize import quantize_float_input
    from netkit.reader import read_test_suite
    from netkit.tflite_quant_align import write_tflite_quant_json

    if not FLOAT_NK.is_file():
        raise SystemExit(f"missing {FLOAT_NK}")
    if not INT8_TFLITE.is_file():
        raise SystemExit(f"missing {INT8_TFLITE}")

    DW_GEN.mkdir(parents=True, exist_ok=True)
    if not QUANT_JSON.is_file():
        write_tflite_quant_json(INT8_TFLITE, QUANT_JSON)
        print(f"Wrote {QUANT_JSON}")

    input_scale, input_zp = _tflite_input_quant(INT8_TFLITE, QUANT_JSON)
    suite = read_test_suite(FLOAT_NK)
    if suite is None:
        raise SystemExit(f"missing TCAS in {FLOAT_NK}")
    cases = _one_per_digit_cases(suite.cases, num=NUM_IMAGES)

    out_h = DW_GEN / "mnist_cnn_int8_test_images.h"
    out_cc = DW_GEN / "mnist_cnn_int8_test_images.cc"
    count_name = "kMnistCnnInt8BenchmarkImageCount"
    size_name = "kMnistCnnInt8BenchmarkInputSize"
    scale_name = "kMnistCnnInt8BenchmarkInputScale"
    zp_name = "kMnistCnnInt8BenchmarkInputZeroPoint"
    sample_name = "MnistCnnInt8BenchmarkSample"
    images_name = "kMnistCnnInt8BenchmarkImages"
    image_symbol = "kMnistCnnInt8Image"

    hdr = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        f"constexpr int {count_name} = {len(cases)};",
        f"constexpr int {size_name} = {INPUT_SIZE};",
        f"constexpr float {scale_name} = {input_scale:.9g}f;",
        f"constexpr int {zp_name} = {int(input_zp)};",
        "",
        f"struct {sample_name} {{",
        "  const char* name;",
        "  int label;",
        "  const int8_t* pixels;",
        "};",
        "",
        f"extern const {sample_name} {images_name}[{len(cases)}];",
        "",
    ]
    cc = [f'#include "{out_h.name}"', ""]
    for idx, case in enumerate(cases):
        pixels = np.asarray(case.input, dtype=np.float32).reshape(-1)
        if pixels.size != INPUT_SIZE:
            raise SystemExit(f"case {case.name} has {pixels.size} inputs")
        pixels_i8 = quantize_float_input(pixels, input_scale, input_zp).astype(np.int8)
        pixel_text = ", ".join(str(int(v)) for v in pixels_i8)
        cc.append(
            f"alignas(16) static const int8_t {image_symbol}{idx}[{INPUT_SIZE}] = {{{pixel_text}}};"
        )
        cc.append("")
    cc.append(f"const {sample_name} {images_name}[{len(cases)}] = {{")
    for idx, case in enumerate(cases):
        cc.append(f'  {{"{case.name}", {int(case.label)}, {image_symbol}{idx}}},')
    cc.append("};")
    cc.append("")
    out_h.write_text("\n".join(hdr) + "\n", encoding="utf-8")
    out_cc.write_text("\n".join(cc) + "\n", encoding="utf-8")
    print(f"Wrote {out_h} and {out_cc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float-only", action="store_true")
    parser.add_argument("--images-only", action="store_true")
    args = parser.parse_args()

    if not FLOAT_NK.is_file():
        raise SystemExit(f"missing {FLOAT_NK} — run: python3 tools/export_mnist_cnn_dw.py")

    GENERATED.mkdir(parents=True, exist_ok=True)
    DW_GEN.mkdir(parents=True, exist_ok=True)
    from netkit.reader import read_test_suite

    if not args.images_only:
        from export_onnx_test_models import export_netkit_nk

        print(f"Exporting ONNX from {FLOAT_NK.name} ...")
        export_netkit_nk(FLOAT_NK)
        _export_tflite_via_onnx2tf(ONNX, FLOAT_TFLITE, SAVED_MODEL)

        if not args.float_only:
            # Prefer onnx2tf -oiqt (avoids a broken host TensorFlow import on some macOS setups).
            cal_npy = GENERATED / "cnn_dw_cal.npy"
            if not cal_npy.is_file():
                cal = _load_mnist_calibration(NUM_CALIBRATION)
                np.save(cal_npy, cal.reshape(NUM_CALIBRATION, 1, 28, 28, 1))
            int8_out = GENERATED / "cnn_dw_int8_onnx2tf"
            if int8_out.exists():
                shutil.rmtree(int8_out)
            print(f"Quantizing {ONNX.name} with onnx2tf -oiqt ...")
            subprocess.run(
                [
                    _onnx2tf_bin(),
                    "-i",
                    str(ONNX),
                    "-o",
                    str(int8_out),
                    "-nuo",
                    "-oiqt",
                    "-qt",
                    "per-channel",
                    "-ioqd",
                    "int8",
                    "-cind",
                    "input",
                    str(cal_npy),
                    "[[[[0.0]]]]",
                    "[[[[1.0]]]]",
                ],
                check=True,
            )
            full_i8 = int8_out / "mnist_cnn_dw_full_integer_quant.tflite"
            if not full_i8.is_file():
                raise SystemExit(f"missing {full_i8}")
            shutil.copy2(full_i8, INT8_TFLITE)
            print(f"Wrote {INT8_TFLITE} ({INT8_TFLITE.stat().st_size} bytes)")
            print("Quantizing netkit int8 .nk ...")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "export_mnist_cnn_dw_int8.py"),
                    "--align-tflite",
                    str(INT8_TFLITE),
                    "--write-tflite-quant-json",
                    str(QUANT_JSON),
                ],
                check=True,
            )

    _write_test_image_arrays(read_test_suite, FLOAT_IMAGES)
    print(f"Wrote {FLOAT_IMAGES['images_h']} and {FLOAT_IMAGES['images_cc']}")

    if not args.float_only:
        if not INT8_TFLITE.is_file():
            raise SystemExit(f"missing {INT8_TFLITE}")
        _write_int8_images()


if __name__ == "__main__":
    main()
