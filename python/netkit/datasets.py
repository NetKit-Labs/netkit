"""IDX dataset loaders for offline training scripts (NumPy I/O only)."""

from __future__ import annotations

import gzip
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from .speech_features import read_wav_mono, waveform_to_feature_map

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

FASHION_MNIST_FILES = {
    "train_images": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
        "train-images-idx3-ubyte.gz",
    ),
    "train_labels": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ),
    "test_images": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ),
    "test_labels": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
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


def load_fashion_mnist(
    *, data_dir: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test images (N x 784 float32) and labels (uint8)."""
    data_dir = data_dir or REPO_ROOT / "data" / "fashion_mnist"
    _download_idx(data_dir, FASHION_MNIST_FILES)
    x_train = _read_idx_images(data_dir / "train-images-idx3-ubyte.gz")
    y_train = _read_idx_labels(data_dir / "train-labels-idx1-ubyte.gz")
    x_test = _read_idx_images(data_dir / "t10k-images-idx3-ubyte.gz")
    y_test = _read_idx_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test


SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
SPEECH_COMMANDS_TAR = "speech_commands_v0.02.tar.gz"

SPEECH_KWS_KEYWORDS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
    "silence",
)

_SPEECH_CORE_WORDS = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)


def _label_for_folder(folder: str) -> int | None:
    if folder in _SPEECH_CORE_WORDS:
        return _SPEECH_CORE_WORDS.index(folder)
    if folder == "_background_noise_":
        return 11
    if folder.startswith("_"):
        return None
    return 10  # unknown — all non-core command words


def _download_speech_commands(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / SPEECH_COMMANDS_TAR
    if not tar_path.exists() or tar_path.stat().st_size == 0:
        print(f"Downloading {SPEECH_COMMANDS_TAR} ...")
        urllib.request.urlretrieve(SPEECH_COMMANDS_URL, tar_path)
    return tar_path


def _speech_root(data_dir: Path) -> Path:
    nested = data_dir / "speech_commands_v0.02"
    if nested.is_dir() and (nested / "yes").is_dir():
        return nested
    if (data_dir / "yes").is_dir():
        return data_dir
    raise FileNotFoundError(f"speech commands dataset not found under {data_dir}")


def _read_split_lists(data_dir: Path) -> tuple[set[str], set[str]]:
    root = _speech_root(data_dir)
    val_names: set[str] = set()
    test_names: set[str] = set()
    for base in (root, data_dir):
        val_file = base / "validation_list.txt"
        test_file = base / "testing_list.txt"
        if val_file.is_file():
            val_names = {line.strip() for line in val_file.read_text().splitlines() if line.strip()}
        if test_file.is_file():
            test_names = {line.strip() for line in test_file.read_text().splitlines() if line.strip()}
    return val_names, test_names


def _extract_speech_commands(data_dir: Path) -> Path:
    try:
        return _speech_root(data_dir)
    except FileNotFoundError:
        pass
    tar_path = _download_speech_commands(data_dir)
    print(f"Extracting {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    return _speech_root(data_dir)


def _feature_cache_path(data_dir: Path) -> Path:
    return data_dir / "kws_16x10.npz"


def _build_speech_feature_cache(
    data_dir: Path,
    *,
    per_class_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = _extract_speech_commands(data_dir)
    val_names, test_names = _read_split_lists(data_dir)

    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    class_counts = {label: 0 for label in range(len(SPEECH_KWS_KEYWORDS))}

    for folder in sorted(p.name for p in root.iterdir() if p.is_dir()):
        label = _label_for_folder(folder)
        if label is None:
            continue
        wav_dir = root / folder
        for wav_path in sorted(wav_dir.glob("*.wav")):
            if per_class_limit > 0 and class_counts[label] >= per_class_limit:
                break
            rel = f"{folder}/{wav_path.name}"
            features = waveform_to_feature_map(read_wav_mono(wav_path)).reshape(-1)
            if rel in test_names:
                test_x.append(features)
                test_y.append(label)
            elif rel in val_names:
                test_x.append(features)
                test_y.append(label)
            else:
                train_x.append(features)
                train_y.append(label)
            class_counts[label] += 1

    if not train_x or not test_x:
        raise RuntimeError("failed to build speech feature cache — dataset tree missing wav files")

    return (
        np.stack(train_x).astype(np.float32),
        np.asarray(train_y, dtype=np.uint8),
        np.stack(test_x).astype(np.float32),
        np.asarray(test_y, dtype=np.uint8),
    )


def load_speech_commands_kws(
    *,
    data_dir: Path | None = None,
    per_class_limit: int = 0,
    rebuild_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test feature vectors (N x 160 float32) and labels (uint8, 12 classes)."""
    data_dir = data_dir or REPO_ROOT / "data" / "speech_commands"
    cache_path = _feature_cache_path(data_dir)
    if cache_path.is_file() and not rebuild_cache:
        data = np.load(cache_path)
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    x_train, y_train, x_test, y_test = _build_speech_feature_cache(
        data_dir, per_class_limit=per_class_limit
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Wrote feature cache {cache_path}")
    return x_train, y_train, x_test, y_test
