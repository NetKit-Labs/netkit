"""YOLOX Nano-style PAFPN + multi-scale decoupled heads for netkit."""

from __future__ import annotations

from typing import Any

import numpy as np

from .mobilenetv4_small import MOBILENETV4_CONV_SMALL_BLOCKS, build_mobilenetv4_small_arch
from .reference_forward import _conv_nhwc, _depthwise_conv_nhwc

YOLOX_MAX_STACKED_CONVS = 4

# Backbone block indices (before inserting feature taps) that produce C3/C4/C5.
MNV4_SMALL_C3_BLOCK_INDEX = 4  # conv_bn 1x1 → 64, stride 8
MNV4_SMALL_C4_BLOCK_INDEX = 10  # uib 3,0,1,96,4.0, stride 16
MNV4_SMALL_C5_BLOCK_INDEX = 17  # final conv_bn 1x1 → 960, stride 32

MNV4_SMALL_C3_CHANNELS = 64
MNV4_SMALL_C4_CHANNELS = 96
MNV4_SMALL_C5_CHANNELS = 960

YOLOX_PAFPN_STRIDES = (8, 16, 32)


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def _upsample_nearest_2x(x: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


def _dw_pw_silu(
    x: np.ndarray,
    *,
    dw_w: np.ndarray,
    dw_b: np.ndarray,
    pw_w: np.ndarray,
    pw_b: np.ndarray,
    stride: int,
) -> np.ndarray:
    y = _depthwise_conv_nhwc(x, dw_w, dw_b, stride=stride, pad_h=1, pad_w=1)
    y = _conv_nhwc(y, pw_w, pw_b, stride=1, pad_h=0, pad_w=0)
    return _silu(y)


def verify_mnv4_small_tap_blocks() -> None:
    """Assert C3/C4/C5 block indices match stride/channel accumulation."""
    ch = 3
    stride = 1
    for i, block in enumerate(MOBILENETV4_CONV_SMALL_BLOCKS):
        if block[0] == "conv_bn":
            _, _k, s, out_ch = block
            stride *= int(s)
            ch = int(out_ch)
        else:
            _, _sd, _md, s, out_ch, _er = block
            stride *= int(s)
            ch = int(out_ch)
        if i == MNV4_SMALL_C3_BLOCK_INDEX:
            assert stride == 8 and ch == MNV4_SMALL_C3_CHANNELS, (i, stride, ch)
        if i == MNV4_SMALL_C4_BLOCK_INDEX:
            assert stride == 16 and ch == MNV4_SMALL_C4_CHANNELS, (i, stride, ch)
        if i == MNV4_SMALL_C5_BLOCK_INDEX:
            assert stride == 32 and ch == MNV4_SMALL_C5_CHANNELS, (i, stride, ch)


def build_yolox_mnv4_small_pafpn_detector(
    *,
    height: int,
    width: int,
    channels: int = 3,
    num_classes: int = 10,
    hidden_dim: int = 64,
    num_convs: int = 2,
) -> dict[str, Any]:
    """MobileNetV4-Conv-Small + feature taps + fused YOLOX Nano PAFPN + 3 heads."""
    if num_convs < 1 or num_convs > YOLOX_MAX_STACKED_CONVS:
        raise ValueError(f"num_convs must be in 1..{YOLOX_MAX_STACKED_CONVS}, got {num_convs}")
    verify_mnv4_small_tap_blocks()

    backbone = build_mobilenetv4_small_arch(
        height=height, width=width, channels=channels, include_head=False
    )
    layers: list[dict[str, Any]] = []
    for i, layer in enumerate(backbone["layers"]):
        layers.append(layer)
        if i == MNV4_SMALL_C3_BLOCK_INDEX:
            layers.append(
                {"type": "feature_tap", "channels": MNV4_SMALL_C3_CHANNELS, "tap_id": 0}
            )
        elif i == MNV4_SMALL_C4_BLOCK_INDEX:
            layers.append(
                {"type": "feature_tap", "channels": MNV4_SMALL_C4_CHANNELS, "tap_id": 1}
            )

    # C5 is the backbone output (after final conv); no tap needed — PAFPN input.
    layers.append(
        {
            "type": "yolox_pafpn_multiscale",
            "c3_channels": MNV4_SMALL_C3_CHANNELS,
            "c4_channels": MNV4_SMALL_C4_CHANNELS,
            "c5_channels": MNV4_SMALL_C5_CHANNELS,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "num_convs": num_convs,
        }
    )
    return {"network": "cnn", "input": list(backbone["input"]), "layers": layers}


def pack_yolox_pafpn_weights_flat(
    rng: np.random.Generator,
    *,
    c3_channels: int,
    c4_channels: int,
    c5_channels: int,
    hidden_dim: int,
    num_classes: int,
    num_convs: int,
    scale: float = 0.05,
) -> list[np.ndarray]:
    """Flat W/B pairs in .nk catalog order for yolox_pafpn_multiscale."""
    parts: list[np.ndarray] = []
    H = hidden_dim

    def lat(in_c: int) -> None:
        parts.append(rng.standard_normal(H * in_c, dtype=np.float32) * scale)
        parts.append(rng.standard_normal(H, dtype=np.float32) * 0.01)

    def dw_pw() -> None:
        parts.append(rng.standard_normal(H * 9, dtype=np.float32) * scale)
        parts.append(rng.standard_normal(H, dtype=np.float32) * 0.01)
        parts.append(rng.standard_normal(H * H, dtype=np.float32) * scale)
        parts.append(rng.standard_normal(H, dtype=np.float32) * 0.01)

    lat(c3_channels)
    lat(c4_channels)
    lat(c5_channels)
    dw_pw()  # td_p4
    dw_pw()  # td_p3
    dw_pw()  # bu_n4
    dw_pw()  # bu_n5

    for _ in range(3):
        from .yolox_detector import pack_yolox_head_weights_flat

        parts.extend(
            pack_yolox_head_weights_flat(
                rng,
                in_channels=H,
                hidden_dim=H,
                num_classes=num_classes,
                num_convs=num_convs,
                scale=scale,
            )
        )
    return parts


def yolox_pafpn_forward_from_offset(
    c3: np.ndarray,
    c4: np.ndarray,
    c5: np.ndarray,
    *,
    c3_channels: int,
    c4_channels: int,
    c5_channels: int,
    hidden_dim: int,
    num_classes: int,
    num_convs: int,
    weights: np.ndarray,
    offset: int,
) -> tuple[np.ndarray, int]:
    """Catalog-order reference forward for yolox_pafpn_multiscale."""
    H = hidden_dim

    def take_lat(in_c: int, feat: np.ndarray) -> np.ndarray:
        nonlocal offset
        w = weights[offset : offset + H * in_c].reshape(H, 1, 1, in_c)
        offset += H * in_c
        b = weights[offset : offset + H]
        offset += H
        return _conv_nhwc(feat, w, b, stride=1, pad_h=0, pad_w=0)

    def take_dw_pw(feat: np.ndarray, stride: int) -> np.ndarray:
        nonlocal offset
        dw_w = weights[offset : offset + H * 9].reshape(H, 3, 3)
        offset += H * 9
        dw_b = weights[offset : offset + H]
        offset += H
        pw_w = weights[offset : offset + H * H].reshape(H, 1, 1, H)
        offset += H * H
        pw_b = weights[offset : offset + H]
        offset += H
        return _dw_pw_silu(feat, dw_w=dw_w, dw_b=dw_b, pw_w=pw_w, pw_b=pw_b, stride=stride)

    # 1. laterals lat3, lat4, lat5
    l3 = take_lat(c3_channels, c3)
    l4 = take_lat(c4_channels, c4)
    l5 = take_lat(c5_channels, c5)

    # 2. top-down
    p5 = l5
    p4 = take_dw_pw(l4 + _upsample_nearest_2x(p5), stride=1)
    p3 = take_dw_pw(l3 + _upsample_nearest_2x(p4), stride=1)

    # 3. bottom-up
    n3 = p3
    n4 = take_dw_pw(n3, stride=2) + p4
    n5 = take_dw_pw(n4, stride=2) + p5

    outs: list[np.ndarray] = []
    from .yolox_detector import yolox_decoupled_head_forward_nhwc

    for feat in (n3, n4, n5):
        head_out, offset = yolox_decoupled_head_forward_nhwc(
            feat,
            in_channels=H,
            hidden_dim=H,
            num_classes=num_classes,
            num_convs=num_convs,
            weights=weights,
            offset=offset,
        )
        outs.append(head_out.reshape(-1))

    return np.concatenate(outs).astype(np.float32), offset


def pafpn_output_elements(
    *,
    input_height: int,
    input_width: int,
    num_classes: int,
) -> int:
    """Elements in flat multi-scale concat for a given input size (stride 32 backbone)."""
    out_c = 4 + 1 + int(num_classes)
    h5, w5 = input_height // 32, input_width // 32
    h4, w4 = h5 * 2, w5 * 2
    h3, w3 = h5 * 4, w5 * 4
    return (h3 * w3 + h4 * w4 + h5 * w5) * out_c
