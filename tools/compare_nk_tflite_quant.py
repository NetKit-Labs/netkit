#!/usr/bin/env python3
"""Compare per-layer quant params between netkit .nk and TFLite int8 models.

Usage (from repo root):
  benchmark/tflm/.venv/bin/python3 tools/compare_nk_tflite_quant.py \\
      models/mnist_cnn_int8.nk benchmark/tflm/generated/mnist_cnn_int8.tflite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit.quant_nk_reader import read_quant_nk  # noqa: E402
from netkit.tflite_quant_align import extract_tflite_cnn_quant_specs  # noqa: E402


def _rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nk", type=Path, help="Quantized netkit .nk model")
    parser.add_argument("tflite", type=Path, help="TFLite int8 model for comparison")
    parser.add_argument(
        "--scale-tol",
        type=float,
        default=0.15,
        help="Relative tolerance for scale comparison (default 0.15)",
    )
    args = parser.parse_args()

    bundle = read_quant_nk(args.nk)
    nk_layers = bundle.quant_layers
    tflite_layers = extract_tflite_cnn_quant_specs(args.tflite)

    print(f"netkit quant layers: {len(nk_layers)}")
    print(f"tflite quant layers: {len(tflite_layers)}")
    if len(nk_layers) != len(tflite_layers):
        raise SystemExit(
            f"layer count mismatch: netkit={len(nk_layers)} tflite={len(tflite_layers)}"
        )

    fields = (
        "input_scale",
        "input_zero_point",
        "output_scale",
        "output_zero_point",
    )
    optional_fields = (
        "weight_scale",
        "weight_zero_point",
        "bias_scale",
    )
    failures = 0
    for index, (nk, tf) in enumerate(zip(nk_layers, tflite_layers)):
        print(f"\nlayer {index}:")
        for field in fields:
            nk_val = getattr(nk, field)
            tf_val = getattr(tf, field)
            if field.endswith("_zero_point"):
                ok = nk_val == tf_val
                print(f"  {field}: nk={nk_val} tflite={tf_val} {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    failures += 1
            else:
                rel = _rel_diff(float(nk_val), float(tf_val))
                ok = rel <= args.scale_tol
                print(
                    f"  {field}: nk={float(nk_val):.8f} tflite={float(tf_val):.8f} "
                    f"rel={rel:.4f} {'OK' if ok else 'WARN'}"
                )
                if not ok:
                    failures += 1
        for field in optional_fields:
            nk_val = getattr(nk, field)
            tf_val = getattr(tf, field)
            if field.endswith("_zero_point"):
                print(f"  {field}: nk={nk_val} tflite={tf_val}")
            else:
                rel = _rel_diff(float(nk_val), float(tf_val))
                print(
                    f"  {field}: nk={float(nk_val):.8f} tflite={float(tf_val):.8f} "
                    f"rel={rel:.4f} (info)"
                )

    if failures:
        raise SystemExit(f"{failures} field(s) outside tolerance")
    print("\nOK: all layers aligned within tolerance")


if __name__ == "__main__":
    main()
