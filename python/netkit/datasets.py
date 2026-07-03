"""IDX dataset loaders for offline training scripts (NumPy I/O only)."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

MNIST_CSV = {
    "train": REPO_ROOT.parent / "python" / "mnist" / "mnist_train.csv",
    "test": REPO_ROOT.parent / "python" / "mnist" / "mnist_test.csv",
}

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


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, delimiter=",", dtype=np.float32)
    labels = table[:, 0].astype(np.uint8)
    images = table[:, 1:] / 255.0
    return images, labels


def _download_idx(data_dir: Path, files: dict[str, tuple[str, str]]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for _key, (url, name) in files.items():
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(count, rows * cols).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic in {path}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(*, data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test images (N x 784 float32) and labels (uint8)."""
    data_dir = data_dir or REPO_ROOT / "data" / "mnist"
    if MNIST_CSV["train"].is_file() and MNIST_CSV["test"].is_file():
        print(f"Loading MNIST from {MNIST_CSV['train'].parent} CSV files")
        x_train, y_train = _load_csv(MNIST_CSV["train"])
        x_test, y_test = _load_csv(MNIST_CSV["test"])
        return x_train, y_train, x_test, y_test

    _download_idx(data_dir, MNIST_FILES)
    x_train = _read_idx_images(data_dir / "train-images-idx3-ubyte.gz")
    y_train = _read_idx_labels(data_dir / "train-labels-idx1-ubyte.gz")
    x_test = _read_idx_images(data_dir / "t10k-images-idx3-ubyte.gz")
    y_test = _read_idx_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test
