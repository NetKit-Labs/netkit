"""Depthwise conv ONNX import and reference parity."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from netkit.onnx_convert import convert_onnx_to_nk
from netkit.reference_forward import forward_cnn


class DepthwiseConvOnnxTests(unittest.TestCase):
    def test_depthwise_conv_import_and_forward(self) -> None:
        weight = np.random.randn(4, 1, 3, 3).astype(np.float32) * 0.1
        bias = np.random.randn(4).astype(np.float32) * 0.05
        x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])
        y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])
        w = numpy_helper.from_array(weight, name="w")
        b = numpy_helper.from_array(bias, name="b")
        conv = helper.make_node(
            "Conv",
            inputs=["input", "w", "b"],
            outputs=["output"],
            kernel_shape=[3, 3],
            strides=[1, 1],
            pads=[1, 1, 1, 1],
            group=4,
        )
        graph = helper.make_graph([conv], "dw", [x], [y], [w, b])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "dw.onnx"
            nk_path = Path(tmp) / "dw.nk"
            onnx.save(model, onnx_path)
            convert_onnx_to_nk(onnx_path, nk_path)

            from netkit.reader import read_nk

            arch, weights = read_nk(nk_path)
            self.assertEqual(arch["layers"][0]["type"], "depthwise_conv2d")
            inp = np.random.randn(8 * 8 * 4).astype(np.float32) * 0.2
            out = forward_cnn(inp, arch, weights)
            self.assertEqual(len(out), 8 * 8 * 4)


if __name__ == "__main__":
    unittest.main()
