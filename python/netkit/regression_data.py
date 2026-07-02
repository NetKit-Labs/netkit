"""Embedded regression suites for bundled hand-check models."""

from __future__ import annotations

from .writer import RegressionCase, RegressionSuite

HAND_SUITES: dict[str, RegressionSuite] = {
    "test_mlp.nk": RegressionSuite(
        tolerance=1e-5,
        cases=[
            RegressionCase("2-layer forward", [1, 2], [3, 3]),
            RegressionCase("zero input", [0, 0], [0, 0]),
            RegressionCase("relu clamps negative hidden", [-1, 3], [2, 2]),
        ],
    ),
    "mlp_hand.nk": RegressionSuite(
        tolerance=1e-5,
        cases=[
            RegressionCase("positive features", [1, 2, 3], [8, 5]),
            RegressionCase("zero input (bias only)", [0, 0, 0], [1, -1]),
            RegressionCase("relu zeros negative hidden unit", [-1, 2, 1], [7, 3]),
            RegressionCase("scaled positive features", [4, 5, 6], [17, 11]),
            RegressionCase("fractional input", [0.5, 1.5, 2.5], [6.5, 4]),
            RegressionCase("sparse activation", [2, 0, 1], [3, 2]),
        ],
    ),
    "test_cnn.nk": RegressionSuite(
        tolerance=1e-5,
        cases=[
            RegressionCase(
                "channel stacking",
                [1, 10, 2, 20, 3, 30, 4, 40],
                [21, 42, 42, 84, 63, 126, 84, 168],
            ),
            RegressionCase(
                "mixed channel weights",
                [2, 4, 6, 8, 1, 3, 5, 7],
                [10, 20, 22, 44, 7, 14, 19, 38],
            ),
        ],
    ),
    "cnn_4x4_single.nk": RegressionSuite(
        tolerance=1e-5,
        cases=[
            RegressionCase("3x3 single-layer spatial conv", [1] * 16, [4, 4, 4, 4]),
            RegressionCase(
                "corner impulses",
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                [1, 0, 0, 2],
            ),
        ],
    ),
    "cnn_hand.nk": RegressionSuite(
        tolerance=1e-5,
        cases=[
            RegressionCase(
                "graded 2-channel spatial input",
                [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60, 7, 70, 8, 80, 9, 90],
                [634, 846, 1270, 1482],
            ),
            RegressionCase(
                "uniform channels",
                [1, 2] * 9,
                [50, 50, 50, 50],
            ),
            RegressionCase(
                "checkerboard pattern",
                [9, 1, 8, 2, 7, 3, 6, 4, 5, 5, 4, 6, 3, 7, 2, 8, 1, 9],
                [142, 150, 166, 174],
            ),
        ],
    ),
}
