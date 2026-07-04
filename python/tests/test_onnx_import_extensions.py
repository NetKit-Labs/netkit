"""ONNX import extensions: asymmetric pads, non-square pools, MatMul MLP, convert optimize."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from netkit.onnx_convert import convert_onnx_to_nk, onnx_to_spec
from netkit.reader import read_nk
from netkit.reference_forward import forward_cnn, forward_mlp


def _build_asymmetric_conv_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), "w")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), "b")
    conv = helper.make_node(
        "Conv",
        ["x", "w", "b"],
        ["y"],
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 2, 1],
    )
    graph = helper.make_graph([conv], "g", [x], [y], [w, b])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))


def _build_rect_pool_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 6])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3, 5])
    pool = helper.make_node(
        "MaxPool",
        ["x"],
        ["y"],
        kernel_shape=[2, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([pool], "g", [x], [y])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))


def _build_matmul_mlp_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "w")
    b = numpy_helper.from_array(np.array([0.1, 0.2, 0.3], dtype=np.float32), "b")
    mm = helper.make_node("MatMul", ["x", "w"], ["mm"])
    add = helper.make_node("Add", ["mm", "b"], ["y"])
    graph = helper.make_graph([mm, add], "g", [x], [y], [w, b])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))


def test_asymmetric_conv_import():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "asym.onnx"
        _build_asymmetric_conv_onnx(onnx_path)
        spec = onnx_to_spec(onnx_path)
        assert spec.layers[0].pad_h == 1
        assert spec.layers[0].pad_h_end == 2
        assert spec.layers[0].pad_w == 1
        assert spec.layers[0].pad_w_end == 1


def test_non_square_pool_import():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "pool.onnx"
        _build_rect_pool_onnx(onnx_path)
        spec = onnx_to_spec(onnx_path)
        assert spec.layers[0].pool_size == 2
        assert spec.layers[0].pool_w == 3


def test_matmul_mlp_import():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "mlp.onnx"
        nk_path = Path(tmp) / "mlp.nk"
        _build_matmul_mlp_onnx(onnx_path)
        convert_onnx_to_nk(onnx_path, nk_path, optimize=False)
        arch, weights = read_nk(nk_path)
        out = forward_mlp(np.array([1.0, 2.0], dtype=np.float32), arch, weights)
        assert len(out) == 3


def test_convert_runs_optimize_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "avg_bn.onnx"
        nk_path = Path(tmp) / "opt.nk"
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
        avg = helper.make_node("AveragePool", ["x"], ["p"], kernel_shape=[2, 2], strides=[2, 2])
        scale = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "scale")
        beta = numpy_helper.from_array(np.array([0.1, -0.2], dtype=np.float32), "beta")
        mean = numpy_helper.from_array(np.zeros(2, dtype=np.float32), "mean")
        var = numpy_helper.from_array(np.ones(2, dtype=np.float32), "var")
        bn = helper.make_node("BatchNormalization", ["p", "scale", "beta", "mean", "var"], ["y"])
        graph = helper.make_graph([avg, bn], "g", [x], [y], [scale, beta, mean, var])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(onnx_path))
        convert_onnx_to_nk(onnx_path, nk_path)
        arch, weights = read_nk(nk_path)
        assert [layer["type"] for layer in arch["layers"]] == ["avg_pool2d", "batch_norm2d"]
        inp = np.random.randn(4 * 4 * 2).astype(np.float32)
        _ = forward_cnn(inp, arch, weights)
