"""Graph optimizations for AOT / packager."""

from __future__ import annotations

import unittest

import numpy as np

from netkit.nk_optimize import OptimizeOptions, optimize_nk
from netkit.reader import read_nk, read_test_suite
from netkit.reference_forward import forward_cnn, forward_mlp


class TestNkOptimize(unittest.TestCase):
    def test_fold_batch_norm_into_dense_on_extended_cnn(self) -> None:
        arch, weights = read_nk("models/cnn_extended_ops.nk")
        before_layers = len(arch["layers"])
        result = optimize_nk(arch, weights)
        self.assertIn("fold_batch_norm_into_dense", result.applied)
        self.assertLess(len(result.arch["layers"]), before_layers)
        suite = read_test_suite("models/cnn_extended_ops.nk")
        assert suite is not None
        for case in suite.cases:
            inp = np.asarray(case.input, dtype=np.float32)
            expected = np.asarray(case.expected, dtype=np.float32)
            actual = np.asarray(forward_cnn(inp, result.arch, result.weights), dtype=np.float32)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=suite.tolerance)

    def test_fold_conv_batch_norm(self) -> None:
        arch = {
            "network": "cnn",
            "input": [2, 2, 1],
            "layers": [
                {
                    "type": "conv2d",
                    "kernel_size": 1,
                    "stride": 1,
                    "filters": 1,
                    "activation": "none",
                },
                {"type": "batch_norm2d", "channels": 1},
                {"type": "flatten"},
                {"type": "dense", "units": 1, "activation": "none"},
            ],
        }
        weights = np.array(
            [1.0, 2.0, 0.5, 0.25, 1.0, 0.0, 0.0, 0.0, 0.1],
            dtype=np.float32,
        )
        result = optimize_nk(arch, weights)
        self.assertIn("fold_conv_batch_norm", result.applied)
        self.assertEqual([layer["type"] for layer in result.arch["layers"]], ["conv2d", "flatten", "dense"])
        inp = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        expected = forward_cnn(inp, arch, weights)
        actual = forward_cnn(inp, result.arch, result.weights)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_merge_linear_dense(self) -> None:
        arch = {
            "network": "mlp",
            "input": [1, 2],
            "layers": [
                {"type": "dense", "units": 2, "activation": "none"},
                {"type": "dense", "units": 1, "activation": "relu"},
            ],
        }
        weights = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5], dtype=np.float32)
        result = optimize_nk(arch, weights)
        self.assertIn("merge_linear_dense", result.applied)
        self.assertEqual(len(result.arch["layers"]), 1)
        inp = np.array([1.0, 2.0], dtype=np.float32)
        expected = forward_mlp(inp, arch, weights)
        actual = forward_mlp(inp, result.arch, result.weights)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_no_op_when_disabled(self) -> None:
        arch, weights = read_nk("models/cnn_extended_ops.nk")
        result = optimize_nk(
            arch,
            weights,
            options=OptimizeOptions(
                fold_conv_batch_norm=False,
                fold_batch_norm_into_dense=False,
                merge_linear_dense=False,
                remove_identity_batch_norm=False,
            ),
        )
        self.assertEqual(result.applied, [])
        self.assertEqual(len(result.arch["layers"]), len(arch["layers"]))


if __name__ == "__main__":
    unittest.main()
