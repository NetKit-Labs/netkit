#!/usr/bin/env python3
"""Train MNIST MLP, quantize to int8, export models/mnist_mlp_int8.nk."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from netkit import RegressionSuite
from netkit.datasets import load_mnist
from netkit.quantize import forward_quantized_mlp, quantize_mlp, quantized_mlp_to_spec
from netkit.torch_models import TutorialMlp784
from netkit.torch_pack import forward_mlp_netkit, pack_tutorial_mlp
from netkit.torch_train import select_digit_cases, train_classifier
from netkit.writer import write_nk_bytes

MODELS = ROOT / "models"

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42
NUM_CASES = 10
NUM_CALIBRATION = 512

ARCH = {
    "network": "mlp",
    "input": [1, 784],
    "layers": [
        {"type": "dense", "units": 128, "activation": "relu"},
        {"type": "dense", "units": 10, "activation": "softmax"},
    ],
}


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist()

    print(
        f"Training MNIST MLP on {x_train.shape[0]} images "
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
    float_weights = pack_tutorial_mlp(model)
    float_probs = forward_mlp_netkit(model, x_test)
    float_acc = (float_probs.argmax(axis=1) == y_test).mean()
    print(f"Float test accuracy: {float_acc * 100:.2f}%")

    pack = quantize_mlp(ARCH, float_weights, x_train, num_calibration=NUM_CALIBRATION)
    quant_probs = np.stack(
        [forward_quantized_mlp(x, ARCH, pack, output_float=True) for x in x_test],
        axis=0,
    )
    quant_acc = (quant_probs.argmax(axis=1) == y_test).mean()
    print(f"Quantized test accuracy: {quant_acc * 100:.2f}%")

    cases = select_digit_cases(
        lambda x: forward_quantized_mlp(x, ARCH, pack, output_float=True),
        x_test,
        y_test,
        num_cases=NUM_CASES,
        name_fmt="MNIST digit {digit} (test idx {i})",
    )

    spec = quantized_mlp_to_spec(ARCH, pack)
    spec.tests = RegressionSuite(tolerance=0.05, cases=cases)

    MODELS.mkdir(parents=True, exist_ok=True)
    out_path = MODELS / "mnist_mlp_int8.nk"
    blob = write_nk_bytes(spec)
    out_path.write_bytes(blob)
    print(f"Wrote {out_path} ({len(blob)} bytes, {len(cases)} embedded test cases)")


if __name__ == "__main__":
    main()
