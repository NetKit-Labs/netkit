"""Packager-side composite fusion from linear .nk layer stacks."""

from __future__ import annotations

import unittest

import numpy as np

from netkit.nk_fuse import (
    FuseOptions,
    expand_mobilenetv4_uib_to_linear,
    fuse_composite_blocks,
)
from netkit.nk_optimize import OptimizeOptions, optimize_nk
from netkit.reader import read_nk, read_test_suite
from netkit.reference_forward import forward_cnn


class TestNkFuse(unittest.TestCase):
    def test_uib_roundtrip_from_fixture(self) -> None:
        arch, weights = read_nk("models/mobilenetv4_small_uib.nk")
        self.assertEqual(arch["layers"][0]["type"], "mobilenetv4_uib")

        linear_arch, linear_weights = expand_mobilenetv4_uib_to_linear(arch, weights)
        self.assertGreater(len(linear_arch["layers"]), 1)
        self.assertTrue(all(layer["type"] != "mobilenetv4_uib" for layer in linear_arch["layers"]))

        suite = read_test_suite("models/mobilenetv4_small_uib.nk")
        assert suite is not None
        suite_inp = np.asarray(suite.cases[0].input, dtype=np.float32)
        expected = forward_cnn(suite_inp, arch, weights)
        fused = fuse_composite_blocks(linear_arch, linear_weights, verify_output=False)
        self.assertIn("mobilenetv4_uib", fused.applied)
        self.assertEqual(len(fused.arch["layers"]), 1)
        self.assertEqual(fused.arch["layers"][0]["type"], "mobilenetv4_uib")
        actual = forward_cnn(suite_inp, fused.arch, fused.weights)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_optimize_can_fuse_uib(self) -> None:
        arch, weights = read_nk("models/mobilenetv4_small_uib.nk")
        linear_arch, linear_weights = expand_mobilenetv4_uib_to_linear(arch, weights)
        result = optimize_nk(
            linear_arch,
            linear_weights,
            options=OptimizeOptions(
                fold_conv_batch_norm=False,
                fold_batch_norm_into_dense=False,
                merge_linear_dense=False,
                remove_identity_batch_norm=False,
                fuse_composite=True,
            ),
        )
        self.assertIn("mobilenetv4_uib", result.applied)
        self.assertEqual(result.arch["layers"][0]["type"], "mobilenetv4_uib")

    def test_fuse_disabled_is_noop(self) -> None:
        arch, weights = read_nk("models/mobilenetv4_small_uib.nk")
        linear_arch, linear_weights = expand_mobilenetv4_uib_to_linear(arch, weights)
        result = fuse_composite_blocks(
            linear_arch,
            linear_weights,
            options=FuseOptions(mobilenetv4_uib=False),
        )
        self.assertEqual(result.applied, [])
        self.assertEqual(len(result.arch["layers"]), len(linear_arch["layers"]))


if __name__ == "__main__":
    unittest.main()
