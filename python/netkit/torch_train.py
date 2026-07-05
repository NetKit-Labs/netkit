"""Shared PyTorch training helpers for offline model export scripts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .writer import RegressionCase


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 0,
    train_limit: int = 0,
) -> None:
    if train_limit > 0 and x_train.shape[0] > train_limit:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
        print(f"Training subset: {train_limit} images")
    else:
        print(f"Training on full set: {x_train.shape[0]} images")

    set_seed(seed)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    x_tensor = torch.from_numpy(x_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.int64))
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = forward_logits(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = forward_logits(x_tensor.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
        train_acc = (pred == y_train).mean()
        print(f"epoch {epoch + 1}/{epochs} train acc {train_acc * 100:.2f}%")
        model.train()


def train_cnn_classifier(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    img_h: int,
    img_w: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 0,
    train_limit: int = 0,
) -> None:
    if train_limit > 0 and x_train.shape[0] > train_limit:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
        print(f"Training subset: {train_limit} images")
    else:
        print(f"Training on full set: {x_train.shape[0]} images")

    set_seed(seed)
    device = torch.device("cpu")
    model.to(device)
    model.train()

    nchw = x_train.reshape(-1, img_h, img_w, 1).transpose(0, 3, 1, 2).astype(np.float32)
    x_tensor = torch.from_numpy(nchw)
    y_tensor = torch.from_numpy(y_train.astype(np.int64))
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = forward_logits(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = forward_logits(x_tensor.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
        train_acc = (pred == y_train).mean()
        print(f"epoch {epoch + 1}/{epochs} train acc {train_acc * 100:.2f}%")
        model.train()


def select_digit_cases(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    num_cases: int,
    name_fmt: str,
) -> list[RegressionCase]:
    """Pick one correctly-classified example per digit when possible."""
    probs = predict_proba(x_test)
    pred = probs.argmax(axis=1)

    cases: list[RegressionCase] = []
    used_digits: set[int] = set()
    for i in range(x_test.shape[0]):
        if pred[i] != y_test[i]:
            continue
        digit = int(y_test[i])
        if digit in used_digits:
            continue
        cases.append(
            RegressionCase(
                name=name_fmt.format(digit=digit, i=i),
                input=x_test[i],
                expected=probs[i],
                label=digit,
            )
        )
        used_digits.add(digit)
        if len(cases) >= num_cases:
            break

    if len(cases) < num_cases:
        for i in range(x_test.shape[0]):
            if pred[i] != y_test[i]:
                continue
            if any(c.label == int(y_test[i]) for c in cases):
                continue
            cases.append(
                RegressionCase(
                    name=name_fmt.format(digit=int(y_test[i]), i=i),
                    input=x_test[i],
                    expected=probs[i],
                    label=int(y_test[i]),
                )
            )
            if len(cases) >= num_cases:
                break

    if len(cases) < num_cases:
        raise RuntimeError(f"only found {len(cases)} test cases; train longer or relax selection")
    cases.sort(key=lambda c: c.label)
    return cases


def select_keyword_cases(
    predict_logits: Callable[[np.ndarray], np.ndarray],
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    keyword_names: list[str],
    num_cases: int,
    name_fmt: str,
    x_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> list[RegressionCase]:
    """Pick one correctly-classified example per keyword when possible."""

    def collect(pool_x: np.ndarray, pool_y: np.ndarray) -> list[RegressionCase]:
        logits = predict_logits(pool_x)
        pred = logits.argmax(axis=1)
        cases: list[RegressionCase] = []
        used_labels: set[int] = set()
        for i in range(pool_x.shape[0]):
            if pred[i] != pool_y[i]:
                continue
            label = int(pool_y[i])
            if label in used_labels:
                continue
            cases.append(
                RegressionCase(
                    name=name_fmt.format(keyword=keyword_names[label], i=i),
                    input=pool_x[i],
                    expected=logits[i],
                    label=label,
                )
            )
            used_labels.add(label)
            if len(cases) >= num_cases:
                break
        return cases

    cases = collect(x_test, y_test)
    if len(cases) < num_cases and x_train is not None and y_train is not None:
        for case in collect(x_train, y_train):
            if any(existing.label == case.label for existing in cases):
                continue
            cases.append(case)
            if len(cases) >= num_cases:
                break

    if len(cases) < num_cases:
        raise RuntimeError(f"only found {len(cases)} keyword test cases; train longer or relax selection")
    return cases
