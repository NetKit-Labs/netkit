#!/usr/bin/env python3
"""Write yolox_pafpn_taps.nk — synthetic C3→tap→C4→tap→C5→PAFPN (no MNv4 backbone)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np

from netkit.arch_writer import pack_random_cnn_weights, write_nk_from_arch
from netkit.reference_forward import forward_cnn
from netkit.writer import RegressionCase, RegressionSuite

C3_H, C3_W, C3_C = 8, 8, 64
C4_C = 96
C5_C = 960
HIDDEN = 64
NUM_CLASSES = 10
NUM_CONVS = 2


def main() -> None:
    # Sequential path creates C3/C4/C5 spatial sizes and taps them for PAFPN.
    arch = {
        "network": "cnn",
        "input": [C3_H, C3_W, C3_C],
        "layers": [
            {"type": "feature_tap", "channels": C3_C, "tap_id": 0},
            # 8x8x64 → 4x4x96
            {
                "type": "conv2d",
                "kernel_size": 3,
                "stride": 2,
                "filters": C4_C,
                "pad_h": 1,
                "pad_w": 1,
                "activation": "relu",
            },
            {"type": "feature_tap", "channels": C4_C, "tap_id": 1},
            # 4x4x96 → 2x2x960
            {
                "type": "conv2d",
                "kernel_size": 3,
                "stride": 2,
                "filters": C5_C,
                "pad_h": 1,
                "pad_w": 1,
                "activation": "relu",
            },
            {
                "type": "yolox_pafpn_multiscale",
                "c3_channels": C3_C,
                "c4_channels": C4_C,
                "c5_channels": C5_C,
                "hidden_dim": HIDDEN,
                "num_classes": NUM_CLASSES,
                "num_convs": NUM_CONVS,
            },
        ],
    }
    rng = np.random.default_rng(77)
    weights = pack_random_cnn_weights(arch, rng, scale=0.02)
    inp = rng.standard_normal(C3_H * C3_W * C3_C, dtype=np.float32) * 0.05
    expected = forward_cnn(inp, arch, weights)

    out = ROOT / "models" / "yolox_pafpn_taps.nk"
    write_nk_from_arch(
        arch,
        weights,
        out,
        RegressionSuite(
            tolerance=1e-4,
            cases=[
                RegressionCase(
                    name="YOLOX PAFPN on synthetic taps",
                    input=inp,
                    expected=expected,
                )
            ],
        ),
    )
    # Remove legacy head-only fixture if present.
    legacy = ROOT / "models" / "yolox_head_only.nk"
    if legacy.exists():
        legacy.unlink()
    print(
        f"Wrote {out} ({len(arch['layers'])} layers, {weights.nbytes} bytes, "
        f"output={len(expected)} floats)"
    )


if __name__ == "__main__":
    main()
