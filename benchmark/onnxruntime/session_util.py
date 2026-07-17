"""Shared ONNX Runtime session helpers for host peer benches."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_session(
    model_path: Path,
    *,
    num_threads: int,
    use_xnnpack: bool,
):
    """Create an InferenceSession with XNNPACK ON or CPU-only (XNNPACK OFF).

    Requires a LiteRT-matched ORT build with --use_xnnpack
    (tools/build_onnxruntime_litert_matched.sh). Stock pip wheels lack the EP.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1 if use_xnnpack else max(1, num_threads)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    xnn_name = next(
        (p for p in available if p.lower() == "xnnpackexecutionprovider"),
        None,
    )

    providers: list
    if use_xnnpack:
        if xnn_name is None:
            raise SystemExit(
                "XnnpackExecutionProvider not available. Build ORT with:\n"
                "  ./tools/build_onnxruntime_litert_matched.sh\n"
                f"Available: {available}"
            )
        # Match TF Lite / netkit host policy: 1 thread for XNNPACK.
        providers = [
            (
                xnn_name,
                {"intra_op_num_threads": str(max(1, num_threads))},
            ),
            "CPUExecutionProvider",
        ]
        # Avoid fighting XNNPACK's pool with ORT's (ORT XNNPACK EP guidance).
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.intra_op_num_threads = 1
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(str(model_path), sess_options=so, providers=providers)


def backend_name(use_xnnpack: bool) -> str:
    return "xnnpack" if use_xnnpack else "cpu-ep"


def input_meta(session) -> tuple[str, list[int], np.dtype]:
    inp = session.get_inputs()[0]
    shape = [int(d) if isinstance(d, int) or (isinstance(d, str) and d.isdigit()) else d for d in inp.shape]
    # Resolve dynamic dims to 1 for batch when needed by callers.
    fixed = []
    for d in shape:
        if isinstance(d, int):
            fixed.append(d)
        else:
            fixed.append(1)
    dtype = np.float32
    if inp.type == "tensor(int8)":
        dtype = np.int8
    elif inp.type == "tensor(uint8)":
        dtype = np.uint8
    elif inp.type == "tensor(float16)":
        dtype = np.float16
    return inp.name, fixed, dtype


def output_name(session) -> str:
    return session.get_outputs()[0].name
