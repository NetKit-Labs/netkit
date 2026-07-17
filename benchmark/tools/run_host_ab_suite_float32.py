#!/usr/bin/env python3
"""CPU host three-way suite: netkit vs TF Lite vs ONNX Runtime — FLOAT32.

Models:
  - MNIST CNN (digit classifier)
  - MNIST DS-CNN (depthwise-separable digit peer)
  - MobileNetV4-Conv-Small on ImageNet (10-class fixture)

Same fairness policy as host_ab_suite_common (prebuild, discard 1st process,
order swaps, LiteRT-matched -O3). ORT is source-built with the same
compiler/linker flags + XNNPACK EP (tools/build_onnxruntime_litert_matched.sh).

Sweeps XNNPACK ON/OFF on all three peers.
NETKIT_IM2COL is fixed at 0 (direct).

Results default: benchmark/host_ab_suite_results_float32.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from host_ab_suite_common import FLOAT32, cli_entry, ensure_assets_float32

if __name__ == "__main__":
    cli_entry(
        FLOAT32,
        ensure_assets_float32,
        __doc__ or "Host A/B suite (float32)",
    )
