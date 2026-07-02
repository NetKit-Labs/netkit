#!/usr/bin/env python3
"""Train Fashion-MNIST MLP with PyTorch and export netkit model + test cases.

Run from repo root:
    python3 tools/export_fashion_mnist_mlp.py

Requires: pip install -e "python[train]"

Outputs:
    models/fashion_mnist_mlp.nk (weights + 10 embedded regression cases)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionSuite, write_nk_from_arch
from netkit.datasets import load_fashion_mnist
from netkit.torch_models import TutorialMlp784
from netkit.torch_pack import assert_packed_matches_reference, forward_mlp_netkit, pack_tutorial_mlp
from netkit.torch_train import select_digit_cases, train_classifier

MODELS = ROOT / "models"

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 43
NUM_CASES = 10

ARCH = {
    "network": "mlp",
    "input": [1, 784],
    "layers": [
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"},
    ],
}


def main() -> None:
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    print(
        f"Training Fashion-MNIST MLP on {x_train.shape[0]} images "
        f"(PyTorch Adam lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}) ..."
    )

    model = TutorialMlp784()
    train_classifier(
        model,
        x_train,
        y_train,
        forward_logits=model.forward_logits,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
    )

    model.eval()
    test_probs = forward_mlp_netkit(model, x_test)
    test_acc = (test_probs.argmax(axis=1) == y_test).mean()
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    weights = pack_tutorial_mlp(model)
    assert_packed_matches_reference(
        ARCH,
        weights,
        lambda inp: forward_mlp_netkit(model, inp),
        seed=SEED,
    )

    cases = select_digit_cases(
        lambda x: forward_mlp_netkit(model, x),
        x_test,
        y_test,
        num_cases=NUM_CASES,
        name_fmt="Fashion-MNIST digit {digit} (test idx {i})",
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    nk_path = write_nk_from_arch(
        ARCH,
        weights,
        MODELS / "fashion_mnist_mlp.nk",
        RegressionSuite(tolerance=0.0001, cases=cases),
    )
    print(f"Wrote {nk_path} ({weights.nbytes} bytes, {len(cases)} embedded test cases)")


if __name__ == "__main__":
    main()
