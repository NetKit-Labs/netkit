#!/usr/bin/env python3
"""Write yolox_mnv4_small.nk fixture (MobileNetV4-Small + YOLOX decoupled head)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np

from netkit.arch_writer import pack_random_cnn_weights, write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.writer import RegressionCase, RegressionSuite
from netkit.yolox_detector import build_yolox_mnv4_small_detector


def main() -> None:
    height, width = 56, 56
    num_classes = 10
    hidden_dim = 64
    num_convs = 2

    arch = build_yolox_mnv4_small_detector(
        height=height,
        width=width,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_convs=num_convs,
    )
    rng = np.random.default_rng(42)
    weights = pack_random_cnn_weights(arch, rng, scale=0.02)
    inp = rng.standard_normal(height * width * 3, dtype=np.float32) * 0.1
    expected = forward_cnn(inp, arch, weights)

    out = ROOT / "models" / "yolox_mnv4_small.nk"
    write_nk_from_arch(
        arch,
        weights,
        out,
        RegressionSuite(
            tolerance=1e-4,
            cases=[
                RegressionCase(
                    name="YOLOX MNv4-Small single-scale",
                    input=inp,
                    expected=expected,
                )
            ],
        ),
    )
    print(f"Wrote {out} ({len(arch['layers'])} layers, {weights.nbytes} bytes, output={len(expected)} floats)")


if __name__ == "__main__":
    main()
