"""Tests for YOLOX MobileNetV4-Small single-scale detector."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from netkit.arch_writer import pack_random_cnn_weights
from netkit.reader import read_nk, read_test_suite
from netkit.reference_forward import forward_cnn
from netkit.runtime_infer import nk_infer_bin, run_nk_infer
from netkit.yolox_decode import decode_yolox_output
from netkit.yolox_detector import (
    build_yolox_mnv4_small_detector,
    yolox_head_output_channels,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
LIB = ROOT / "libnetkit.a"


class YoloxDetectorTests(unittest.TestCase):
    def test_arch_has_backbone_and_head(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=56, width=56, num_classes=10, hidden_dim=64)
        kinds = [layer["type"] for layer in arch["layers"]]
        self.assertIn("mobilenetv4_uib", kinds)
        self.assertEqual(kinds[-1], "yolox_decoupled_head")
        self.assertEqual(arch["layers"][-1]["in_channels"], 960)

    def test_reference_forward_output_shape(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=56, width=56, num_classes=10, hidden_dim=64)
        rng = np.random.default_rng(0)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(56 * 56 * 3, dtype=np.float32) * 0.05
        out = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        self.assertEqual(out.size, 2 * 2 * yolox_head_output_channels(10))

    def test_decode_runs_on_reference_output(self) -> None:
        arch = build_yolox_mnv4_small_detector(height=56, width=56, num_classes=10, hidden_dim=64)
        rng = np.random.default_rng(1)
        weights = pack_random_cnn_weights(arch, rng, scale=0.02)
        inp = rng.standard_normal(56 * 56 * 3, dtype=np.float32) * 0.05
        out = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        dets = decode_yolox_output(out, num_classes=10, score_threshold=0.0, input_height=56, input_width=56)
        self.assertGreater(len(dets), 0)

    def test_fixture_roundtrip(self) -> None:
        nk_path = MODELS / "yolox_mnv4_small.nk"
        if not nk_path.is_file():
            self.skipTest("yolox_mnv4_small.nk missing — run tools/write_yolox_mnv4_detector_fixture.py")

        arch, weights = read_nk(nk_path)
        suite = read_test_suite(nk_path)
        self.assertEqual(arch["layers"][-1]["type"], "yolox_decoupled_head")
        self.assertEqual(len(suite.cases), 1)

        case = suite.cases[0]
        actual = np.asarray(forward_cnn(np.asarray(case.input, dtype=np.float32), arch, weights), dtype=np.float32)
        expected = np.asarray(case.expected, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=suite.tolerance)

    def test_cpp_runtime_matches_reference(self) -> None:
        if not LIB.is_file() or not nk_infer_bin(ROOT).is_file():
            self.skipTest("build required — run `make lib tools/nk_infer`")

        nk_path = MODELS / "yolox_mnv4_small.nk"
        if not nk_path.is_file():
            self.skipTest("yolox_mnv4_small.nk missing — run tools/write_yolox_mnv4_detector_fixture.py")

        arch, weights = read_nk(nk_path)
        suite = read_test_suite(nk_path)
        case = suite.cases[0]
        inp = np.asarray(case.input, dtype=np.float32)
        expected = np.asarray(forward_cnn(inp, arch, weights), dtype=np.float32)
        actual = run_nk_infer(nk_path, inp, root=ROOT)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=suite.tolerance)


if __name__ == "__main__":
    unittest.main()
