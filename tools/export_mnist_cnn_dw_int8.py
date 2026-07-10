#!/usr/bin/env python3
"""Quantize depthwise MNIST CNN to int8: models/mnist_cnn_dw_int8.nk.

Default: reuse models/mnist_cnn_dw.nk (no training).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionSuite
from netkit.datasets import load_mnist
from netkit.quantize import forward_quantized_cnn, quantize_cnn, quantize_float_input, quantized_cnn_to_spec
from netkit.reader import read_nk
from netkit.tflite_quant_align import load_tflite_quant_json, write_tflite_quant_json
from netkit.writer import RegressionCase, write_nk_bytes

MODELS = ROOT / "models"
DEFAULT_FLOAT_NK = MODELS / "mnist_cnn_dw.nk"
DEFAULT_TFLITE = ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_dw_int8.tflite"
DEFAULT_TFLITE_QUANT_JSON = ROOT / "benchmark" / "tflm" / "generated" / "mnist_cnn_dw_int8_tflite_quant.json"
OUT_PATH = MODELS / "mnist_cnn_dw_int8.nk"

IMG_H = 28
IMG_W = 28
IMG_C = 1
NUM_CASES = 10
NUM_CALIBRATION = 128
ACCURACY_SAMPLES = 256

ARCH = {
    "network": "cnn",
    "input": [IMG_H, IMG_W, IMG_C],
    "layers": [
        {"type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 32, "activation": "relu"},
        {
            "type": "depthwise_conv2d",
            "kernel_h": 3,
            "kernel_w": 3,
            "stride": 1,
            "filters": 32,
            "pad_h": 0,
            "pad_w": 0,
            "activation": "relu",
        },
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "conv2d", "kernel_size": 1, "stride": 1, "filters": 64, "activation": "relu"},
        {
            "type": "depthwise_conv2d",
            "kernel_h": 3,
            "kernel_w": 3,
            "stride": 1,
            "filters": 64,
            "pad_h": 0,
            "pad_w": 0,
            "activation": "relu",
        },
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"},
    ],
}


def _select_digit_cases_fast(x_test, y_test, pack, *, num_cases: int, name_fmt: str):
    cases = []
    used_digits: set[int] = set()
    for i in range(x_test.shape[0]):
        probs = forward_quantized_cnn(x_test[i], ARCH, pack, output_float=True)
        pred = int(np.argmax(probs))
        label = int(y_test[i])
        if pred != label or label in used_digits:
            continue
        input_i8 = quantize_float_input(
            x_test[i].reshape(-1),
            pack.quant_layers[0].input_scale,
            pack.quant_layers[0].input_zero_point,
        )
        cases.append(
            RegressionCase(
                name=name_fmt.format(digit=label, i=i),
                input=input_i8.astype(np.int8),
                expected=probs,
                label=label,
            )
        )
        used_digits.add(label)
        if len(cases) >= num_cases:
            break
    if len(cases) < num_cases:
        raise RuntimeError(f"found only {len(cases)}/{num_cases} digit cases")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-nk", type=Path, default=DEFAULT_FLOAT_NK)
    parser.add_argument("-o", "--output", type=Path, default=OUT_PATH)
    parser.add_argument("--align-tflite", type=Path, default=None)
    parser.add_argument("--write-tflite-quant-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.from_nk.is_file():
        raise SystemExit(f"missing {args.from_nk} — run: python3 tools/export_mnist_cnn_dw.py")

    _, float_weights = read_nk(args.from_nk)
    float_weights = np.asarray(float_weights, dtype=np.float32)
    x_train, _, x_test, y_test = load_mnist()

    aligned_quants = None
    tflite_path = args.align_tflite
    if tflite_path is None and DEFAULT_TFLITE.is_file():
        tflite_path = DEFAULT_TFLITE
    if tflite_path is not None and tflite_path.is_file():
        json_out = args.write_tflite_quant_json or DEFAULT_TFLITE_QUANT_JSON
        if json_out.is_file():
            aligned_quants = load_tflite_quant_json(json_out)
            print(f"Aligned quant params from {json_out}")
        else:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            aligned_quants = write_tflite_quant_json(tflite_path, json_out)
            print(f"Aligned quant params from {tflite_path} -> {json_out}")

    print("Quantizing depthwise MNIST CNN to int8 ...")
    pack = quantize_cnn(
        ARCH,
        float_weights,
        x_train,
        num_calibration=NUM_CALIBRATION,
        aligned_quants=aligned_quants,
    )
    check_x = x_test[:ACCURACY_SAMPLES]
    check_y = y_test[:ACCURACY_SAMPLES]
    quant_probs = np.stack([forward_quantized_cnn(row, ARCH, pack) for row in check_x], axis=0)
    print(
        f"Quantized accuracy ({check_x.shape[0]} samples): "
        f"{(quant_probs.argmax(axis=1) == check_y).mean() * 100:.2f}%"
    )

    cases = _select_digit_cases_fast(
        x_test,
        y_test,
        pack,
        num_cases=NUM_CASES,
        name_fmt="MNIST CNN-DW digit {digit} (test idx {i})",
    )
    spec = quantized_cnn_to_spec(ARCH, pack)
    spec.tests = RegressionSuite(tolerance=0.08, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    blob = write_nk_bytes(spec)
    args.output.write_bytes(blob)
    print(f"Wrote {args.output} ({len(blob)} bytes, {len(cases)} cases)")


if __name__ == "__main__":
    main()
