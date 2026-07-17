#!/usr/bin/env python3
"""CPU host three-way suite: netkit vs TF Lite vs ONNX Runtime — INT8.

Models:
  - MNIST CNN (digit classifier)
  - MNIST DS-CNN (depthwise-separable digit peer)
  - MobileNetV4-Conv-Small on ImageNet (10-class fixture)

Same fairness policy as host_ab_suite_common. ORT int8 models are QDQ
(export_host_onnx_assets.py --int8). ORT is source-built with LiteRT-matched
flags + XNNPACK EP (tools/build_onnxruntime_litert_matched.sh).

Sweeps XNNPACK ON/OFF on all three peers.
Results default: benchmark/host_ab_suite_results_int8.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from host_ab_suite_common import INT8, cli_entry, ensure_assets_int8

if __name__ == "__main__":
    cli_entry(
        INT8,
        ensure_assets_int8,
        __doc__ or "Host A/B suite (int8)",
    )
