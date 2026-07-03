"""Full ResNet-18 regression."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from netkit.reader import read_nk, read_test_suite
from netkit.reference_forward import forward_cnn

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "models" / "resnet18.nk"


class ResNet18Tests(unittest.TestCase):
    def test_bundled_fixture_matches_reference(self) -> None:
        arch, weights = read_nk(FIXTURE)
        self.assertEqual(arch["input"], [56, 56, 3])
        self.assertEqual(len(arch["layers"]), 13)

        suite = read_test_suite(FIXTURE)
        self.assertEqual(len(suite.cases), 1)
        case = suite.cases[0]
        out = forward_cnn(np.asarray(case.input, dtype=np.float32), arch, weights)
        self.assertEqual(len(out), len(case.expected))
        np.testing.assert_allclose(out, case.expected, rtol=0.0, atol=suite.tolerance)


if __name__ == "__main__":
    unittest.main()
