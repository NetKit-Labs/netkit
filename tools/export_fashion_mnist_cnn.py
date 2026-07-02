#!/usr/bin/env python3
"""Train a tutorial-style Fashion-MNIST CNN with PyTorch and export netkit model + cases.

Architecture:
  28x28x1 -> Conv3x3x32 ReLU -> MaxPool2x2 -> Conv3x3x64 ReLU -> MaxPool2x2
         -> Flatten -> Dense128 ReLU -> Dense10 Softmax

Run from repo root:
    python3 tools/export_fashion_mnist_cnn.py

Requires: pip install -e "python[train]"

Outputs:
    models/fashion_mnist_cnn.nk (weights + 10 embedded regression cases)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionSuite, write_nk_from_arch
from netkit.datasets import load_fashion_mnist
from netkit.torch_models import TutorialCnn28
from netkit.torch_pack import assert_packed_matches_reference, forward_cnn_netkit, pack_tutorial_cnn
from netkit.torch_train import select_digit_cases, train_cnn_classifier

MODELS = ROOT / "models"

IMG_H = 28
IMG_W = 28
IMG_C = 1
EPOCHS = 15
BATCH_SIZE = 128
TRAIN_LIMIT = 0
LEARNING_RATE = 0.001
SEED = 43
NUM_CASES = 10

ARCH = {
    "network": "cnn",
    "input": [IMG_H, IMG_W, IMG_C],
    "layers": [
        {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 32, "activation": "relu"},
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "conv2d", "kernel_size": 3, "stride": 1, "filters": 64, "activation": "relu"},
        {"type": "max_pool2d", "pool_size": 2, "stride": 2},
        {"type": "flatten"},
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"},
    ],
}


def main() -> None:
    x_train, y_train, x_test, y_test = load_fashion_mnist()

    print(
        f"Training Fashion-MNIST CNN on {x_train.shape[0]} images "
        f"(PyTorch Adam lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}) ..."
    )
    print("Architecture: Conv32/ReLU/Pool -> Conv64/ReLU/Pool -> Flatten -> Dense128/ReLU -> Dense10/Softmax")

    model = TutorialCnn28()
    train_cnn_classifier(
        model,
        x_train,
        y_train,
        forward_logits=model.forward_logits,
        img_h=IMG_H,
        img_w=IMG_W,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        train_limit=TRAIN_LIMIT,
    )

    model.eval()
    test_probs = forward_cnn_netkit(model, x_test, img_h=IMG_H, img_w=IMG_W)
    test_acc = (test_probs.argmax(axis=1) == y_test).mean()
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("Published Fashion-MNIST CNN tutorials typically report ~88–93% test accuracy.")

    weights = pack_tutorial_cnn(model)
    assert_packed_matches_reference(
        ARCH,
        weights,
        lambda inp: forward_cnn_netkit(model, inp, img_h=IMG_H, img_w=IMG_W),
        seed=SEED,
    )

    cases = select_digit_cases(
        lambda x: forward_cnn_netkit(model, x, img_h=IMG_H, img_w=IMG_W),
        x_test,
        y_test,
        num_cases=NUM_CASES,
        name_fmt="Fashion-MNIST CNN digit {digit} (test idx {i})",
    )

    MODELS.mkdir(parents=True, exist_ok=True)
    nk_path = write_nk_from_arch(
        ARCH,
        weights,
        MODELS / "fashion_mnist_cnn.nk",
        RegressionSuite(tolerance=0.0001, cases=cases),
    )
    print(f"Wrote {nk_path} ({weights.nbytes} bytes, {len(cases)} embedded test cases)")


if __name__ == "__main__":
    main()
