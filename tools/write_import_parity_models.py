#!/usr/bin/env python3
"""Build ONNX import-extension fixtures with embedded tests and .nk sidecars.

Creates asymmetric conv, rectangular pool, MatMul MLP, and CNN MatMul-head models
for ONNX Runtime parity (``make test-python``).

Run from repo root:
    python3 tools/write_import_parity_models.py
    make export-import-parity
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionCase, RegressionSuite, read_nk, write_nk_from_arch
from netkit.onnx_convert import convert_onnx_to_nk
from netkit.reference_forward import forward_cnn, forward_mlp

MODELS = ROOT / "models"
OPSET = [helper.make_opsetid("", 13)]


def _save(model, path: Path) -> None:
    onnx.save(model, str(path))


def _case(name: str, inp: list[float], arch: dict, weights: np.ndarray) -> RegressionCase:
    if arch["network"] == "mlp":
        expected = forward_mlp(np.asarray(inp, dtype=np.float32), arch, weights)
    else:
        expected = forward_cnn(np.asarray(inp, dtype=np.float32), arch, weights)
    return RegressionCase(name=name, input=inp, expected=expected)


def build_asym_conv() -> tuple[Path, Path, RegressionSuite]:
    """Single conv with asymmetric ONNX pads [1,1,2,1] (top, left, bottom, right)."""
    onnx_path = MODELS / "import_asym_conv.onnx"
    nk_path = MODELS / "import_asym_conv.nk"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 5, 4])
    w = numpy_helper.from_array(
        np.array([[[[1.0, 0.0, -1.0], [0.5, 0.0, -0.5], [1.0, 0.0, -1.0]]]], dtype=np.float32),
        "w",
    )
    b = numpy_helper.from_array(np.array([0.25], dtype=np.float32), "b")
    conv = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 2, 1],
    )
    graph = helper.make_graph([conv], "asym_conv", [x], [y], [w, b])
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False)

    arch, weights = read_nk(nk_path)
    inputs = [
        ("uniform", [1.0] * 16),
        ("corner impulse", [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
        ("checkerboard", [1.0 if (i + j) % 2 == 0 else -0.5 for i in range(4) for j in range(4)]),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, list(inp), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_rect_pool() -> tuple[Path, Path, RegressionSuite]:
    """MaxPool with kernel [2, 3] on 4×6 input."""
    onnx_path = MODELS / "import_rect_pool.onnx"
    nk_path = MODELS / "import_rect_pool.nk"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 4])
    pool = helper.make_node(
        "MaxPool",
        ["x"],
        ["y"],
        kernel_shape=[2, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([pool], "rect_pool", [x], [y])
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False)

    arch, weights = read_nk(nk_path)
    base = np.arange(1.0, 25.0, dtype=np.float32).reshape(4, 6)
    inputs = [
        ("ramp", base.reshape(-1).tolist()),
        ("uniform", [0.5] * 24),
        ("sparse peaks", [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, list(inp), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_matmul_mlp() -> tuple[Path, Path, RegressionSuite]:
    """Two-layer MLP using MatMul + Add instead of Gemm."""
    onnx_path = MODELS / "import_matmul_mlp.onnx"
    nk_path = MODELS / "import_matmul_mlp.nk"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w1 = numpy_helper.from_array(
        np.array([[0.2, -0.1, 0.3], [0.5, 0.0, -0.4], [0.1, 0.2, 0.1]], dtype=np.float32),
        "w1",
    )
    b1 = numpy_helper.from_array(np.array([0.05, -0.1, 0.2], dtype=np.float32), "b1")
    w2 = numpy_helper.from_array(
        np.array([[0.6, -0.2], [-0.3, 0.4], [0.1, 0.2]], dtype=np.float32),
        "w2",
    )
    b2 = numpy_helper.from_array(np.array([0.1, -0.05], dtype=np.float32), "b2")
    mm1 = helper.make_node("MatMul", ["x", "w1"], ["h1"])
    add1 = helper.make_node("Add", ["h1", "b1"], ["a1"])
    relu = helper.make_node("Relu", ["a1"], ["r1"])
    mm2 = helper.make_node("MatMul", ["r1", "w2"], ["h2"])
    add2 = helper.make_node("Add", ["h2", "b2"], ["y"])
    graph = helper.make_graph([mm1, add1, relu, mm2, add2], "matmul_mlp", [x], [y], [w1, b1, w2, b2])
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False)

    arch, weights = read_nk(nk_path)
    inputs = [
        ("mixed", [1.0, -0.5, 2.0]),
        ("zero", [0.0, 0.0, 0.0]),
        ("relu gate", [-2.0, 0.5, 1.0]),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, list(inp), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def build_cnn_matmul_head() -> tuple[Path, Path, RegressionSuite]:
    """Conv → Flatten → Reshape → MatMul → Add (common classifier export pattern)."""
    onnx_path = MODELS / "import_cnn_matmul_head.onnx"
    nk_path = MODELS / "import_cnn_matmul_head.nk"

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w_conv = numpy_helper.from_array(
        np.array(
            [
                [[[1.0, 0.0, -1.0], [0.0, 0.5, 0.0], [-1.0, 0.0, 1.0]]],
                [[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]],
            ],
            dtype=np.float32,
        ),
        "w_conv",
    )
    b_conv = numpy_helper.from_array(np.array([0.0, 0.1], dtype=np.float32), "b_conv")
    conv = helper.make_node(
        "Conv",
        ["x", "w_conv", "b_conv"],
        ["conv"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
    )
    nhwc = helper.make_node("Transpose", ["conv"], ["nhwc"], perm=[0, 2, 3, 1])
    flat = helper.make_node("Flatten", ["nhwc"], ["flat"], axis=1)
    shape = numpy_helper.from_array(np.array([1, 32], dtype=np.int64), "shape")
    reshape = helper.make_node("Reshape", ["flat", "shape"], ["rs"])
    w_head = numpy_helper.from_array(
        np.arange(64, dtype=np.float32).reshape(32, 2) * 0.01,
        "w_head",
    )
    b_head = numpy_helper.from_array(np.array([0.2, -0.1], dtype=np.float32), "b_head")
    mm = helper.make_node("MatMul", ["rs", "w_head"], ["mm"])
    add = helper.make_node("Add", ["mm", "b_head"], ["y"])
    graph = helper.make_graph(
        [conv, nhwc, flat, reshape, mm, add],
        "cnn_matmul_head",
        [x],
        [y],
        [w_conv, b_conv, shape, w_head, b_head],
    )
    _save(helper.make_model(graph, opset_imports=OPSET), onnx_path)
    convert_onnx_to_nk(onnx_path, nk_path, optimize=False)

    arch, weights = read_nk(nk_path)
    inputs = [
        ("uniform", [1.0] * 16),
        ("gradient", [float(i) for i in range(16)]),
        ("center spike", [0.0] * 5 + [4.0] + [0.0] * 10),
    ]
    suite = RegressionSuite(
        tolerance=1e-4,
        cases=[_case(name, list(inp), arch, weights) for name, inp in inputs],
    )
    return onnx_path, nk_path, suite


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    builders = [
        build_asym_conv,
        build_rect_pool,
        build_matmul_mlp,
        build_cnn_matmul_head,
    ]
    for builder in builders:
        _onnx, nk_path, suite = builder()
        write_nk_from_arch(*read_nk(nk_path), nk_path, suite)
        print(f"Wrote {nk_path.name} + sidecar ({len(suite.cases)} cases)")


if __name__ == "__main__":
    main()
