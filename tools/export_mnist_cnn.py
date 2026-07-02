#!/usr/bin/env python3
"""Train a tutorial-style MNIST CNN and export netkit model + regression cases.

Architecture (common Keras/TensorFlow MNIST tutorial):
  28x28x1 -> Conv3x3x32 ReLU -> MaxPool2x2 -> Conv3x3x64 ReLU -> MaxPool2x2
         -> Flatten -> Dense128 ReLU -> Dense10 Softmax

Training: Adam + cross-entropy. Weights match netkit conv/dense layout.

Run from repo root:
    python3 tools/export_mnist_cnn.py

Requires: numpy

Outputs:
    models/mnist_cnn.nk (weights + 10 embedded regression cases)
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from netkit import RegressionCase, RegressionSuite, write_nk_from_arch
MODELS = ROOT / "models"
DATA_DIR = ROOT / "data" / "mnist"

IMG_H = 28
IMG_W = 28
IMG_C = 1
NUM_CLASSES = 10
NUM_CASES = 10
SEED = 42

EPOCHS = 20
BATCH_SIZE = 128
TRAIN_LIMIT = 0  # 0 = full 60k training set
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


def to_nhwc(flat: np.ndarray) -> np.ndarray:
    return flat.reshape(-1, IMG_H, IMG_W, IMG_C)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def im2col(x: np.ndarray, k: int, stride: int) -> tuple[np.ndarray, int, int]:
    n, h, w, c = x.shape
    out_h = (h - k) // stride + 1
    out_w = (w - k) // stride + 1
    cols = np.zeros((n, out_h, out_w, k * k * c), dtype=np.float32)
    col_idx = 0
    for kh in range(k):
        for kw in range(k):
            patch = x[:, kh : kh + out_h * stride : stride, kw : kw + out_w * stride : stride, :]
            cols[:, :, :, col_idx : col_idx + c] = patch
            col_idx += c
    return cols, out_h, out_w


def col2im(cols: np.ndarray, k: int, stride: int, out_shape: tuple[int, int, int, int]) -> np.ndarray:
    n, h, w, c = out_shape
    x = np.zeros((n, h, w, c), dtype=np.float32)
    _, out_h, out_w, _ = cols.shape
    col_idx = 0
    for kh in range(k):
        for kw in range(k):
            x[:, kh : kh + out_h * stride : stride, kw : kw + out_w * stride : stride, :] += cols[
                :, :, :, col_idx : col_idx + c
            ]
            col_idx += c
    return x


def conv2d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1) -> tuple[np.ndarray, dict]:
    cols, out_h, out_w = im2col(x, w.shape[1], stride)
    w_col = w.reshape(w.shape[0], -1)
    out = cols @ w_col.T
    out = out.reshape(x.shape[0], out_h, out_w, w.shape[0]) + b.reshape(1, 1, 1, -1)
    cache = {"x": x, "w": w, "cols": cols, "k": w.shape[1], "stride": stride, "out_h": out_h, "out_w": out_w}
    return out, cache


def conv2d_backward(dout: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = cache["x"].shape[0]
    k = cache["k"]
    stride = cache["stride"]
    out_h = cache["out_h"]
    out_w = cache["out_w"]
    w = cache["w"]
    cols = cache["cols"]

    dout_flat = dout.reshape(n, out_h, out_w, w.shape[0])
    w_col = w.reshape(w.shape[0], -1)
    dcols = dout_flat @ w_col
    dx = col2im(dcols, k, stride, cache["x"].shape)

    dout_2d = dout_flat.reshape(n * out_h * out_w, w.shape[0])
    cols_2d = cols.reshape(n * out_h * out_w, -1)
    dw_col = dout_2d.T @ cols_2d
    dw = dw_col.reshape(w.shape)
    db = dout_flat.sum(axis=(0, 1, 2))
    return dx, dw, db


def max_pool_forward(x: np.ndarray, pool: int = 2, stride: int = 2) -> tuple[np.ndarray, dict]:
    n, h, w, c = x.shape
    out_h = (h - pool) // stride + 1
    out_w = (w - pool) // stride + 1
    out = np.zeros((n, out_h, out_w, c), dtype=np.float32)
    mask = np.zeros_like(x, dtype=np.float32)
    for oh in range(out_h):
        for ow in range(out_w):
            ih0 = oh * stride
            iw0 = ow * stride
            window = x[:, ih0 : ih0 + pool, iw0 : iw0 + pool, :]
            out[:, oh, ow, :] = window.max(axis=(1, 2))
            win_mask = window == out[:, oh, ow, :][:, None, None, :]
            mask[:, ih0 : ih0 + pool, iw0 : iw0 + pool, :] += win_mask.astype(np.float32)
    return out, {"x": x, "mask": mask, "pool": pool, "stride": stride}


def max_pool_backward(dout: np.ndarray, cache: dict) -> np.ndarray:
    x = cache["x"]
    mask = cache["mask"]
    pool = cache["pool"]
    stride = cache["stride"]
    n, h, w, c = x.shape
    out_h = (h - pool) // stride + 1
    out_w = (w - pool) // stride + 1
    dx = np.zeros_like(x)
    for oh in range(out_h):
        for ow in range(out_w):
            ih0 = oh * stride
            iw0 = ow * stride
            dx[:, ih0 : ih0 + pool, iw0 : iw0 + pool, :] += (
                dout[:, oh, ow, :][:, None, None, :] * mask[:, ih0 : ih0 + pool, iw0 : iw0 + pool, :]
            )
    return dx


def forward_netkit(x_nhwc: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    x, _ = conv2d_forward(x_nhwc, p["c1_w"], p["c1_b"], stride=1)
    x = relu(x)
    x, _ = max_pool_forward(x, pool=2, stride=2)
    x, _ = conv2d_forward(x, p["c2_w"], p["c2_b"], stride=1)
    x = relu(x)
    x, _ = max_pool_forward(x, pool=2, stride=2)
    x = x.reshape(x.shape[0], -1)
    x = relu(x @ p["d1_w"] + p["d1_b"])
    logits = x @ p["d2_w"] + p["d2_b"]
    return softmax(logits)


def forward_backward_fixed(
    x_nhwc: np.ndarray, y_one_hot: np.ndarray, p: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    n = x_nhwc.shape[0]
    c1, c1_cache = conv2d_forward(x_nhwc, p["c1_w"], p["c1_b"], stride=1)
    a1 = relu(c1)
    p1, p1_cache = max_pool_forward(a1, pool=2, stride=2)
    c2, c2_cache = conv2d_forward(p1, p["c2_w"], p["c2_b"], stride=1)
    a2 = relu(c2)
    p2, p2_cache = max_pool_forward(a2, pool=2, stride=2)
    flat = p2.reshape(n, -1)
    z3 = flat @ p["d1_w"] + p["d1_b"]
    a3 = relu(z3)
    logits = a3 @ p["d2_w"] + p["d2_b"]
    probs = softmax(logits)

    dz4 = (probs - y_one_hot) / n
    grads: dict[str, np.ndarray] = {}
    grads["d2_w"] = a3.T @ dz4
    grads["d2_b"] = dz4.sum(axis=0)
    da3 = dz4 @ p["d2_w"].T
    dz3 = da3 * (z3 > 0.0)
    grads["d1_w"] = flat.T @ dz3
    grads["d1_b"] = dz3.sum(axis=0)
    dflat = dz3 @ p["d1_w"].T
    dp2 = dflat.reshape(p2.shape)
    da2 = max_pool_backward(dp2, p2_cache)
    dz2 = da2 * (c2 > 0.0)
    dp1, grads["c2_w"], grads["c2_b"] = conv2d_backward(dz2, c2_cache)
    da1 = max_pool_backward(dp1, p1_cache)
    dz1 = da1 * (c1 > 0.0)
    _, grads["c1_w"], grads["c1_b"] = conv2d_backward(dz1, c1_cache)
    return grads


def init_conv(rng: np.random.Generator, c_out: int, k: int, c_in: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(2.0 / (k * k * c_in))
    w = rng.normal(0.0, scale, size=(c_out, k, k, c_in)).astype(np.float32)
    b = np.zeros(c_out, dtype=np.float32)
    return w, b


def init_dense(rng: np.random.Generator, in_dim: int, out_dim: int) -> tuple[np.ndarray, np.ndarray]:
    w = rng.normal(0.0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim)).astype(np.float32)
    b = np.zeros(out_dim, dtype=np.float32)
    return w, b


def clip_grads(grads: dict[str, np.ndarray], max_norm: float = 5.0) -> None:
    total = 0.0
    for g in grads.values():
        total += float(np.sum(g * g))
    norm = np.sqrt(total)
    if norm > max_norm:
        scale = max_norm / norm
        for key in grads:
            grads[key] *= scale


def train(x_train: np.ndarray, y_train: np.ndarray) -> dict[str, np.ndarray]:
    if TRAIN_LIMIT > 0 and x_train.shape[0] > TRAIN_LIMIT:
        x_train = x_train[:TRAIN_LIMIT]
        y_train = y_train[:TRAIN_LIMIT]
        print(f"Training subset: {TRAIN_LIMIT} images")
    else:
        print(f"Training on full set: {x_train.shape[0]} images")
    rng = np.random.default_rng(SEED)
    c1_w, c1_b = init_conv(rng, 32, 3, 1)
    c2_w, c2_b = init_conv(rng, 64, 3, 32)
    d1_w, d1_b = init_dense(rng, 1600, 128)
    d2_w, d2_b = init_dense(rng, 128, 10)
    p = {"c1_w": c1_w, "c1_b": c1_b, "c2_w": c2_w, "c2_b": c2_b, "d1_w": d1_w, "d1_b": d1_b, "d2_w": d2_w, "d2_b": d2_b}
    opt = Adam(LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPS)
    y_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_train]
    x = to_nhwc(x_train)
    n = x.shape[0]

    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            grads = forward_backward_fixed(x[idx], y_oh[idx], p)
            clip_grads(grads)
            opt.step(p, grads)

        train_pred = forward_netkit(x, p).argmax(axis=1)
        train_acc = (train_pred == y_train).mean()
        print(f"epoch {epoch + 1}/{EPOCHS} train acc {train_acc * 100:.2f}%")

    return p


def pack_conv(w: np.ndarray, b: np.ndarray) -> list[float]:
    values: list[float] = []
    c_out, k, _, c_in = w.shape
    for oc in range(c_out):
        for kh in range(k):
            for kw in range(k):
                for ic in range(c_in):
                    values.append(float(w[oc, kh, kw, ic]))
    values.extend(float(v) for v in b.reshape(-1))
    return values


def pack_dense(w: np.ndarray, b: np.ndarray) -> list[float]:
    values: list[float] = []
    values.extend(w.reshape(-1).tolist())
    values.extend(b.reshape(-1).tolist())
    return values


def pack_netkit_weights(p: dict[str, np.ndarray]) -> bytes:
    values: list[float] = []
    values.extend(pack_conv(p["c1_w"], p["c1_b"]))
    values.extend(pack_conv(p["c2_w"], p["c2_b"]))
    values.extend(pack_dense(p["d1_w"], p["d1_b"]))
    values.extend(pack_dense(p["d2_w"], p["d2_b"]))
    return struct.pack(f"<{len(values)}f", *values)


def append_case(
    cases: list[RegressionCase],
    x_test: np.ndarray,
    y_test: np.ndarray,
    i: int,
    p: dict[str, np.ndarray],
) -> None:
    digit = int(y_test[i])
    img = x_test[i]
    out = forward_netkit(to_nhwc(img[None, :]), p)[0]
    cases.append(
        RegressionCase(
            name=f"MNIST CNN digit {digit} (test idx {i})",
            input=img,
            expected=out,
            label=digit,
        )
    )


def main() -> None:
    x_train, y_train, x_test, y_test = load_mnist()

    print(
        f"Training CNN on {x_train.shape[0]} images "
        f"(Adam lr={LEARNING_RATE}, batch={BATCH_SIZE}, epochs={EPOCHS}) ..."
    )
    print("Architecture: Conv32/ReLU/Pool -> Conv64/ReLU/Pool -> Flatten -> Dense128/ReLU -> Dense10/Softmax")
    params = train(x_train, y_train)

    test_out = forward_netkit(to_nhwc(x_test), params)
    test_pred = test_out.argmax(axis=1)
    test_acc = (test_pred == y_test).mean()
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("Published Keras MNIST CNN tutorials typically report ~98.5–99.3% test accuracy.")

    MODELS.mkdir(parents=True, exist_ok=True)

    bin_bytes = pack_netkit_weights(params)
    arch = {
        "version": 1,
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
    weights = np.frombuffer(bin_bytes, dtype=np.float32)

    cases: list[RegressionCase] = []
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
            if any(c.label == int(y_test[i]) for c in cases):
                continue
            append_case(cases, x_test, y_test, i, params)
            if len(cases) >= NUM_CASES:
                break

    if len(cases) < NUM_CASES:
        raise RuntimeError(f"only found {len(cases)} test cases; train longer or relax selection")

    nk_path = write_nk_from_arch(
        arch,
        weights,
        MODELS / "mnist_cnn.nk",
        RegressionSuite(tolerance=0.0001, cases=cases),
    )
    print(f"Wrote {nk_path} ({len(bin_bytes)} bytes, {len(cases)} embedded test cases)")


if __name__ == "__main__":
    main()
