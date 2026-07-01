#!/usr/bin/env python3
"""Download MNIST, train 784->128->10 MLP (ReLU + softmax), export netkit model + test cases.

Training: Adam + cross-entropy (standard MNIST MLP recipe). Inference matches netkit
Softmax() on the final dense layer (stable softmax over logits).

Run from repo root:
    python3 tools/export_mnist_mlp.py

Requires: numpy

Outputs:
    models/mnist_mlp.json
    models/mnist_mlp.bin
    models/mnist/manifest.json
    models/mnist/case_NNN.input.bin
    models/mnist/case_NNN.expected.bin
"""

from __future__ import annotations

import gzip
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MNIST_DIR = MODELS / "mnist"
DATA_DIR = ROOT / "data" / "mnist"

INPUT_DIM = 784
HIDDEN_DIM = 128
OUTPUT_DIM = 10
NUM_CASES = 10
SEED = 42

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

MNIST_FILES = {
    "train_images": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
    ),
    "train_labels": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ),
    "test_images": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ),
    "test_labels": (
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ),
}

LOCAL_CSV = {
    "train": ROOT.parent / "python" / "mnist" / "mnist_train.csv",
    "test": ROOT.parent / "python" / "mnist" / "mnist_test.csv",
}


class Adam:
    def __init__(self, lr: float, beta1: float, beta2: float, eps: float) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for key in params:
            g = grads[key]
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[key] / (1.0 - self.beta1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, delimiter=",", dtype=np.float32)
    labels = table[:, 0].astype(np.uint8)
    images = table[:, 1:] / 255.0
    return images, labels


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if LOCAL_CSV["train"].is_file() and LOCAL_CSV["test"].is_file():
        print(f"Loading MNIST from {LOCAL_CSV['train'].parent} CSV files")
        x_train, y_train = load_csv(LOCAL_CSV["train"])
        x_test, y_test = load_csv(LOCAL_CSV["test"])
        return x_train, y_train, x_test, y_test

    download_mnist()
    x_train = read_idx_images(DATA_DIR / "train-images-idx3-ubyte.gz")
    y_train = read_idx_labels(DATA_DIR / "train-labels-idx1-ubyte.gz")
    x_test = read_idx_images(DATA_DIR / "t10k-images-idx3-ubyte.gz")
    y_test = read_idx_labels(DATA_DIR / "t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test


def download_mnist() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for _key, (url, name) in MNIST_FILES.items():
        dest = DATA_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic in {path}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax — matches netkit Ops::Softmax (subtract max before exp)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def forward_netkit(
    x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    h = relu(x @ w1 + b1)
    logits = h @ w2 + b2
    return softmax(logits)


def init_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, np.sqrt(2.0 / INPUT_DIM), size=(INPUT_DIM, HIDDEN_DIM)).astype(np.float32),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "w2": rng.normal(0.0, np.sqrt(2.0 / HIDDEN_DIM), size=(HIDDEN_DIM, OUTPUT_DIM)).astype(np.float32),
        "b2": np.zeros(OUTPUT_DIM, dtype=np.float32),
    }


def train(x: np.ndarray, y_one_hot: np.ndarray, y_labels: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    params = init_weights(rng)
    opt = Adam(LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPS)
    n = x.shape[0]

    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            xb = x[idx]
            yb = y_one_hot[idx]
            batch = xb.shape[0]

            z1 = xb @ params["w1"] + params["b1"]
            h1 = relu(z1)
            z2 = h1 @ params["w2"] + params["b2"]
            probs = softmax(z2)

            dz2 = (probs - yb) / batch
            grads_w2 = h1.T @ dz2
            grads_b2 = dz2.sum(axis=0)
            dh1 = dz2 @ params["w2"].T
            dz1 = dh1 * (z1 > 0.0)
            grads_w1 = xb.T @ dz1
            grads_b1 = dz1.sum(axis=0)

            opt.step(
                params,
                {
                    "w1": grads_w1.astype(np.float32),
                    "b1": grads_b1.astype(np.float32),
                    "w2": grads_w2.astype(np.float32),
                    "b2": grads_b2.astype(np.float32),
                },
            )

        train_pred = forward_netkit(x, params["w1"], params["b1"], params["w2"], params["b2"]).argmax(axis=1)
        train_acc = (train_pred == y_labels).mean()
        print(f"epoch {epoch + 1}/{EPOCHS} train acc {train_acc * 100:.2f}%")

    return params


def pack_netkit_weights(params: dict[str, np.ndarray]) -> bytes:
    values: list[float] = []
    values.extend(params["w1"].reshape(-1).tolist())
    values.extend(params["b1"].tolist())
    values.extend(params["w2"].reshape(-1).tolist())
    values.extend(params["b2"].tolist())
    return struct.pack(f"<{len(values)}f", *values)


def write_floats(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(struct.pack(f"<{arr.size}f", *arr.astype(np.float32).reshape(-1)))


def append_case(
    cases: list[dict],
    x_test: np.ndarray,
    y_test: np.ndarray,
    i: int,
    params: dict[str, np.ndarray],
) -> None:
    digit = int(y_test[i])
    out = forward_netkit(x_test[i : i + 1], params["w1"], params["b1"], params["w2"], params["b2"])[0]
    case_id = len(cases)
    prefix = MNIST_DIR / f"case_{case_id:03d}"
    write_floats(prefix.with_suffix(".input.bin"), x_test[i])
    write_floats(prefix.with_suffix(".expected.bin"), out)
    cases.append(
        {
            "name": f"MNIST digit {digit} (test idx {i})",
            "label": digit,
            "input": f"case_{case_id:03d}.input.bin",
            "expected": f"case_{case_id:03d}.expected.bin",
        }
    )


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist()
    y_train_oh = np.eye(OUTPUT_DIM, dtype=np.float32)[y_train]

    print(
        f"Training on {x_train.shape[0]} images "
        f"(Adam lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}, ReLU + softmax) ..."
    )
    params = train(x_train, y_train_oh, y_train)

    test_out = forward_netkit(x_test, params["w1"], params["b1"], params["w2"], params["b2"])
    test_pred = test_out.argmax(axis=1)
    test_acc = (test_pred == y_test).mean()
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    MODELS.mkdir(parents=True, exist_ok=True)
    MNIST_DIR.mkdir(parents=True, exist_ok=True)

    (MODELS / "mnist_mlp.json").write_text(
        json.dumps(
            {
                "version": 1,
                "network": "mlp",
                "input": [1, INPUT_DIM],
                "layers": [
                    {"type": "dense", "units": HIDDEN_DIM, "activation": "relu"},
                    {"type": "dense", "units": OUTPUT_DIM, "activation": "softmax"},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    bin_bytes = pack_netkit_weights(params)
    (MODELS / "mnist_mlp.bin").write_bytes(bin_bytes)
    print(f"Wrote mnist_mlp.bin ({len(bin_bytes)} bytes, {len(bin_bytes) // 4} floats)")

    cases: list[dict] = []
    used_digits: set[int] = set()
    for i in range(x_test.shape[0]):
        if test_pred[i] != y_test[i]:
            continue
        digit = int(y_test[i])
        if digit in used_digits:
            continue
        append_case(cases, x_test, y_test, i, params)
        used_digits.add(digit)
        if len(cases) >= NUM_CASES:
            break

    if len(cases) < NUM_CASES:
        for i in range(x_test.shape[0]):
            if test_pred[i] != y_test[i]:
                continue
            if any(c["label"] == int(y_test[i]) for c in cases):
                continue
            append_case(cases, x_test, y_test, i, params)
            if len(cases) >= NUM_CASES:
                break

    if len(cases) < NUM_CASES:
        raise RuntimeError(f"only found {len(cases)} test cases; train longer or relax selection")

    manifest = {
        "model": "../mnist_mlp.json",
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "output_tolerance": 0.0001,
        "cases": cases,
    }
    (MNIST_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} test cases to {MNIST_DIR}")


if __name__ == "__main__":
    main()
