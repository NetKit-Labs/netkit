"""Host-side decode helpers for YOLOX single-scale detector outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .yolox_detector import yolox_head_output_channels


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


def decode_yolox_output(
    output: np.ndarray,
    *,
    num_classes: int,
    score_threshold: float = 0.25,
    input_height: int | None = None,
    input_width: int | None = None,
) -> list[Detection]:
    """Decode raw NHWC head output [reg(4), obj(1), cls(N)] into axis-aligned boxes."""
    arr = np.asarray(output, dtype=np.float32)
    if arr.ndim == 1:
        out_c = yolox_head_output_channels(num_classes)
        side = int(round(np.sqrt(arr.size / out_c)))
        if side * side * out_c != arr.size:
            raise ValueError(f"cannot infer square grid from flat output size {arr.size}")
        arr = arr.reshape(side, side, out_c)
    if arr.ndim != 3:
        raise ValueError(f"expected HxWxC output, got shape {arr.shape}")

    h, w, c = arr.shape
    expected = yolox_head_output_channels(num_classes)
    if c != expected:
        raise ValueError(f"expected {expected} channels, got {c}")

    img_h = float(input_height if input_height is not None else h)
    img_w = float(input_width if input_width is not None else w)
    stride_h = img_h / float(h)
    stride_w = img_w / float(w)

    reg = arr[..., 0:4]
    obj = 1.0 / (1.0 + np.exp(-arr[..., 4:5]))
    cls = 1.0 / (1.0 + np.exp(-arr[..., 5:]))

    detections: list[Detection] = []
    for y in range(h):
        for x in range(w):
            class_scores = cls[y, x] * obj[y, x, 0]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < score_threshold:
                continue

            cx = (float(x) + 0.5) * stride_w
            cy = (float(y) + 0.5) * stride_h
            l, t, r, b = (float(v) for v in reg[y, x])
            x1 = cx - l * stride_w
            y1 = cy - t * stride_h
            x2 = cx + r * stride_w
            y2 = cy + b * stride_h
            detections.append(
                Detection(
                    x1=max(0.0, x1),
                    y1=max(0.0, y1),
                    x2=min(img_w, x2),
                    y2=min(img_h, y2),
                    score=score,
                    class_id=class_id,
                )
            )

    detections.sort(key=lambda det: det.score, reverse=True)
    return detections
