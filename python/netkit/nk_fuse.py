"""Packager-side fusion of primitive CNN layers into composite blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .arch_writer import _make_divisible, _out_dim, _uib_output_spatial
from .cnn_layers import depthwise_kernel_hw
from .reference_forward import forward_cnn


@dataclass
class FuseOptions:
    """Composite block fusion toggles (packager-side)."""

    resnet_basic_block: bool = True
    mobilenetv4_uib: bool = True


@dataclass
class FuseArchResult:
    arch: dict[str, Any]
    weights: np.ndarray
    applied: list[str] = field(default_factory=list)


@dataclass
class _FuseLayer:
    spec: dict[str, Any]
    tensor_pairs: list[tuple[np.ndarray, np.ndarray]]


@dataclass
class _ShapeState:
    height: int
    width: int
    channels: int


def _pack_tensor_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for weight, bias in pairs:
        parts.append(np.asarray(weight, dtype=np.float32).reshape(-1))
        parts.append(np.asarray(bias, dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def _decompose_cnn(arch: dict[str, Any], weights: np.ndarray) -> list[_FuseLayer]:
    layers: list[_FuseLayer] = []
    height, width, channels = arch["input"]
    offset = 0
    dense_in = 0

    for layer in arch["layers"]:
        layer_type = layer["type"]
        if layer_type == "conv2d":
            k = layer["kernel_size"]
            out_c = layer["filters"]
            kernel_elems = k * k * channels
            w = weights[offset : offset + kernel_elems * out_c].reshape(out_c, k, k, channels).copy()
            offset += kernel_elems * out_c
            b = weights[offset : offset + out_c].copy()
            offset += out_c
            layers.append(_FuseLayer(spec=dict(layer), tensor_pairs=[(w, b)]))
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            stride = layer.get("stride", 1)
            height = _out_dim(height, k, stride, pad_h)
            width = _out_dim(width, k, stride, pad_w)
            channels = out_c
        elif layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            ch = layer["filters"]
            kernel_elems = kh * kw * ch
            w = weights[offset : offset + kernel_elems].reshape(ch, kh, kw).copy()
            offset += kernel_elems
            b = weights[offset : offset + ch].copy()
            offset += ch
            layers.append(_FuseLayer(spec=dict(layer), tensor_pairs=[(w, b)]))
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            stride = layer.get("stride", 1)
            height = _out_dim(height, kh, stride, pad_h)
            width = _out_dim(width, kw, stride, pad_w)
        elif layer_type in {"max_pool2d", "avg_pool2d"}:
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            layers.append(_FuseLayer(spec=dict(layer), tensor_pairs=[]))
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "batch_norm2d":
            ch = layer["channels"]
            scale = weights[offset : offset + ch].copy()
            offset += ch
            beta = weights[offset : offset + ch].copy()
            offset += ch
            layers.append(_FuseLayer(spec=dict(layer), tensor_pairs=[(scale, beta)]))
        elif layer_type == "mobilenetv4_uib":
            in_c = layer["in_channels"]
            out_c = layer["out_channels"]
            start_k = int(layer.get("start_dw_kernel", 0))
            middle_k = int(layer.get("middle_dw_kernel", 0))
            expand_c = _make_divisible(in_c * float(layer["expand_ratio"]), 8)
            pairs: list[tuple[np.ndarray, np.ndarray]] = []
            if start_k:
                dw_elems = start_k * start_k * in_c
                w = weights[offset : offset + dw_elems].reshape(in_c, start_k, start_k).copy()
                offset += dw_elems
                b = weights[offset : offset + in_c].copy()
                offset += in_c
                pairs.append((w, b))
                scale = weights[offset : offset + in_c].copy()
                offset += in_c
                beta = weights[offset : offset + in_c].copy()
                offset += in_c
                pairs.append((scale, beta))
            expand_elems = expand_c * in_c
            w = weights[offset : offset + expand_elems].reshape(expand_c, 1, 1, in_c).copy()
            offset += expand_elems
            b = weights[offset : offset + expand_c].copy()
            offset += expand_c
            pairs.append((w, b))
            scale = weights[offset : offset + expand_c].copy()
            offset += expand_c
            beta = weights[offset : offset + expand_c].copy()
            offset += expand_c
            pairs.append((scale, beta))
            if middle_k:
                dw_elems = middle_k * middle_k * expand_c
                w = weights[offset : offset + dw_elems].reshape(expand_c, middle_k, middle_k).copy()
                offset += dw_elems
                b = weights[offset : offset + expand_c].copy()
                offset += expand_c
                pairs.append((w, b))
                scale = weights[offset : offset + expand_c].copy()
                offset += expand_c
                beta = weights[offset : offset + expand_c].copy()
                offset += expand_c
                pairs.append((scale, beta))
            proj_elems = out_c * expand_c
            w = weights[offset : offset + proj_elems].reshape(out_c, 1, 1, expand_c).copy()
            offset += proj_elems
            b = weights[offset : offset + out_c].copy()
            offset += out_c
            pairs.append((w, b))
            scale = weights[offset : offset + out_c].copy()
            offset += out_c
            beta = weights[offset : offset + out_c].copy()
            offset += out_c
            pairs.append((scale, beta))
            layers.append(_FuseLayer(spec=dict(layer), tensor_pairs=pairs))
            height, width = _uib_output_spatial(height, width, layer)
            channels = out_c
        elif layer_type == "flatten":
            layers.append(_FuseLayer(spec=dict(layer), tensor_pairs=[]))
            dense_in = height * width * channels
        elif layer_type == "dense":
            out_f = layer["units"]
            w_size = dense_in * out_f
            w = weights[offset : offset + w_size].reshape(out_f, dense_in).copy()
            offset += w_size
            b = weights[offset : offset + out_f].copy()
            offset += out_f
            layers.append(_FuseLayer(spec=dict(layer), tensor_pairs=[(w, b)]))
            dense_in = out_f
        else:
            raise ValueError(f"unsupported layer type for fusion decompose: {layer_type}")

    if offset != len(weights):
        raise ValueError(f"weight count mismatch: used {offset}, file has {len(weights)}")
    return layers


def _compose_cnn(arch: dict[str, Any], layers: list[_FuseLayer]) -> tuple[dict[str, Any], np.ndarray]:
    new_arch = {"network": "cnn", "input": list(arch["input"]), "layers": [layer.spec for layer in layers]}
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for layer in layers:
        pairs.extend(layer.tensor_pairs)
    return new_arch, _pack_tensor_pairs(pairs)


def _simulate_shapes(arch: dict[str, Any]) -> dict[int, _ShapeState]:
    shapes: dict[int, _ShapeState] = {}
    height, width, channels = arch["input"]
    for index, layer in enumerate(arch["layers"]):
        shapes[index] = _ShapeState(height=height, width=width, channels=channels)
        layer_type = layer["type"]
        if layer_type == "conv2d":
            k = layer["kernel_size"]
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, k, stride, pad_h)
            width = _out_dim(width, k, stride, pad_w)
            channels = layer["filters"]
        elif layer_type == "depthwise_conv2d":
            kh, kw = depthwise_kernel_hw(layer)
            stride = layer.get("stride", 1)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, kh, stride, pad_h)
            width = _out_dim(width, kw, stride, pad_w)
        elif layer_type in {"max_pool2d", "avg_pool2d"}:
            pool = layer["pool_size"]
            stride = layer.get("stride", pool)
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            height = _out_dim(height, pool, stride, pad_h)
            width = _out_dim(width, pool, stride, pad_w)
        elif layer_type == "mobilenetv4_uib":
            height, width = _uib_output_spatial(height, width, layer)
            channels = layer["out_channels"]
    return shapes


def _symmetric_same_pad(kernel: int, pad_h: int, pad_w: int) -> bool:
    expected = (kernel - 1) // 2
    return pad_h == expected and pad_w == expected


def _match_conv1x1_bn(
    layers: list[_FuseLayer], index: int, *, in_channels: int
) -> tuple[_FuseLayer, _FuseLayer, int] | None:
    if index + 1 >= len(layers):
        return None
    conv = layers[index]
    bn = layers[index + 1]
    if conv.spec.get("type") != "conv2d" or bn.spec.get("type") != "batch_norm2d":
        return None
    if conv.spec.get("kernel_size") != 1 or conv.spec.get("stride", 1) != 1:
        return None
    if conv.spec.get("pad_h", 0) or conv.spec.get("pad_w", 0):
        return None
    if conv.spec.get("activation", "none") != "none":
        return None
    out_c = int(conv.spec["filters"])
    if int(bn.spec["channels"]) != out_c:
        return None
    assert conv.tensor_pairs and bn.tensor_pairs
    w = conv.tensor_pairs[0][0]
    if w.shape != (out_c, 1, 1, in_channels):
        return None
    return conv, bn, 2


def _match_depthwise_bn(
    layers: list[_FuseLayer], index: int, *, channels: int, allowed_k: tuple[int, ...] = (3, 5)
) -> tuple[_FuseLayer, _FuseLayer, int, int] | None:
    if index + 1 >= len(layers):
        return None
    dw = layers[index]
    bn = layers[index + 1]
    if dw.spec.get("type") != "depthwise_conv2d" or bn.spec.get("type") != "batch_norm2d":
        return None
    if dw.spec.get("activation", "none") != "none":
        return None
    kh, kw = depthwise_kernel_hw(dw.spec)
    if kh != kw or kh not in allowed_k:
        return None
    if int(dw.spec["filters"]) != channels or int(bn.spec["channels"]) != channels:
        return None
    pad_h = int(dw.spec.get("pad_h", 0))
    pad_w = int(dw.spec.get("pad_w", 0))
    if not _symmetric_same_pad(kh, pad_h, pad_w):
        return None
    stride = int(dw.spec.get("stride", 1))
    assert dw.tensor_pairs and bn.tensor_pairs
    w = dw.tensor_pairs[0][0]
    if w.shape != (channels, kh, kw):
        return None
    return dw, bn, stride, kh


def _infer_uib_stride(
    *,
    start_k: int,
    start_stride: int,
    middle_k: int,
    middle_stride: int,
) -> tuple[int, int]:
    """Return (block_stride, middle_dw_downsample)."""
    if middle_k > 0:
        middle_down = 1 if middle_stride > 1 else 0
        if middle_down:
            return middle_stride, 1
        return start_stride if start_k > 0 else 1, 0
    if start_k > 0:
        return start_stride, 1
    return 1, 1


def _try_match_mobilenetv4_uib(
    layers: list[_FuseLayer], index: int, shape: _ShapeState
) -> tuple[_FuseLayer, int] | None:
    in_c = shape.channels
    cursor = index
    start_k = 0
    start_stride = 1
    start_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    matched = _match_depthwise_bn(layers, cursor, channels=in_c)
    if matched is not None:
        start_dw, start_bn, start_stride, start_k = matched
        start_pairs.extend(start_dw.tensor_pairs)
        start_pairs.extend(start_bn.tensor_pairs)
        cursor += 2

    expand = _match_conv1x1_bn(layers, cursor, in_channels=in_c)
    if expand is None:
        return None
    expand_conv, expand_bn, expand_span = expand
    expand_c = int(expand_conv.spec["filters"])
    cursor += expand_span

    middle_k = 0
    middle_stride = 1
    middle_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    matched = _match_depthwise_bn(layers, cursor, channels=expand_c)
    if matched is not None:
        middle_dw, middle_bn, middle_stride, middle_k = matched
        middle_pairs.extend(middle_dw.tensor_pairs)
        middle_pairs.extend(middle_bn.tensor_pairs)
        cursor += 2

    project = _match_conv1x1_bn(layers, cursor, in_channels=expand_c)
    if project is None:
        return None
    project_conv, project_bn, project_span = project
    out_c = int(project_conv.spec["filters"])
    cursor += project_span

    block_stride, middle_down = _infer_uib_stride(
        start_k=start_k,
        start_stride=start_stride,
        middle_k=middle_k,
        middle_stride=middle_stride,
    )
    if middle_k > 0 and middle_down and start_k > 0 and start_stride != 1:
        return None
    if middle_k > 0 and not middle_down and middle_stride != 1:
        return None

    expand_ratio = float(expand_c) / float(in_c)
    uib_spec = {
        "type": "mobilenetv4_uib",
        "in_channels": in_c,
        "out_channels": out_c,
        "start_dw_kernel": start_k,
        "middle_dw_kernel": middle_k,
        "stride": block_stride,
        "expand_ratio": expand_ratio,
        "middle_dw_downsample": middle_down,
    }
    tensor_pairs = start_pairs + expand_conv.tensor_pairs + expand_bn.tensor_pairs
    tensor_pairs += middle_pairs + project_conv.tensor_pairs + project_bn.tensor_pairs
    consumed = cursor - index
    return _FuseLayer(spec=uib_spec, tensor_pairs=tensor_pairs), consumed


def expand_mobilenetv4_uib_to_linear(arch: dict[str, Any], weights: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Expand composite UIB layers into primitive conv / depthwise / batch-norm sequences."""
    if arch["network"] != "cnn":
        raise ValueError("expand_mobilenetv4_uib_to_linear requires a CNN arch")
    layers = _decompose_cnn(arch, weights)
    linear_layers: list[_FuseLayer] = []
    for layer in layers:
        if layer.spec.get("type") != "mobilenetv4_uib":
            linear_layers.append(layer)
            continue
        spec = layer.spec
        in_c = int(spec["in_channels"])
        out_c = int(spec["out_channels"])
        start_k = int(spec.get("start_dw_kernel", 0))
        middle_k = int(spec.get("middle_dw_kernel", 0))
        stride = int(spec.get("stride", 1))
        middle_down = bool(spec.get("middle_dw_downsample", 1))
        expand_c = _make_divisible(in_c * float(spec["expand_ratio"]), 8)
        start_stride = 1 if middle_down and middle_k > 0 else stride
        middle_stride = stride if middle_down else 1
        pairs = list(layer.tensor_pairs)
        pair_idx = 0

        def take_pair() -> tuple[np.ndarray, np.ndarray]:
            nonlocal pair_idx
            pair = pairs[pair_idx]
            pair_idx += 1
            return pair

        if start_k:
            w, b = take_pair()
            scale, beta = take_pair()
            pad = (start_k - 1) // 2
            linear_layers.append(
                _FuseLayer(
                    spec={
                        "type": "depthwise_conv2d",
                        "kernel_h": start_k,
                        "kernel_w": start_k,
                        "stride": start_stride,
                        "filters": in_c,
                        "pad_h": pad,
                        "pad_w": pad,
                        "activation": "none",
                    },
                    tensor_pairs=[(w, b)],
                )
            )
            linear_layers.append(
                _FuseLayer(spec={"type": "batch_norm2d", "channels": in_c}, tensor_pairs=[(scale, beta)])
            )

        w, b = take_pair()
        scale, beta = take_pair()
        linear_layers.append(
            _FuseLayer(
                spec={
                    "type": "conv2d",
                    "kernel_size": 1,
                    "stride": 1,
                    "filters": expand_c,
                    "activation": "none",
                },
                tensor_pairs=[(w, b)],
            )
        )
        linear_layers.append(
            _FuseLayer(spec={"type": "batch_norm2d", "channels": expand_c}, tensor_pairs=[(scale, beta)])
        )

        if middle_k:
            w, b = take_pair()
            scale, beta = take_pair()
            pad = (middle_k - 1) // 2
            linear_layers.append(
                _FuseLayer(
                    spec={
                        "type": "depthwise_conv2d",
                        "kernel_h": middle_k,
                        "kernel_w": middle_k,
                        "stride": middle_stride,
                        "filters": expand_c,
                        "pad_h": pad,
                        "pad_w": pad,
                        "activation": "none",
                    },
                    tensor_pairs=[(w, b)],
                )
            )
            linear_layers.append(
                _FuseLayer(spec={"type": "batch_norm2d", "channels": expand_c}, tensor_pairs=[(scale, beta)])
            )

        w, b = take_pair()
        scale, beta = take_pair()
        linear_layers.append(
            _FuseLayer(
                spec={
                    "type": "conv2d",
                    "kernel_size": 1,
                    "stride": 1,
                    "filters": out_c,
                    "activation": "none",
                },
                tensor_pairs=[(w, b)],
            )
        )
        linear_layers.append(
            _FuseLayer(spec={"type": "batch_norm2d", "channels": out_c}, tensor_pairs=[(scale, beta)])
        )

    return _compose_cnn(arch, linear_layers)


def fuse_composite_blocks(
    arch: dict[str, Any],
    weights: np.ndarray,
    *,
    options: FuseOptions | None = None,
    atol: float = 1e-5,
    verify_output: bool = True,
) -> FuseArchResult:
    """Collapse primitive layer sequences into composite blocks where patterns match."""
    opts = options or FuseOptions()
    if arch.get("network") != "cnn":
        return FuseArchResult(arch=arch, weights=weights, applied=[])

    weights = np.asarray(weights, dtype=np.float32)
    h, w, c = arch["input"]
    probes = [np.random.default_rng(17 + i).standard_normal(h * w * c).astype(np.float32) for i in range(3)]
    baseline = (
        [np.asarray(forward_cnn(probe, arch, weights), dtype=np.float32) for probe in probes]
        if verify_output
        else None
    )

    current_arch = arch
    current_weights = weights
    applied: list[str] = []

    while True:
        layers = _decompose_cnn(current_arch, current_weights)
        shapes = _simulate_shapes(current_arch)
        changed = False
        index = 0
        while index < len(layers):
            if layers[index].spec.get("type") == "mobilenetv4_uib":
                index += 1
                continue
            if not opts.mobilenetv4_uib:
                break
            shape = shapes.get(index)
            if shape is None:
                index += 1
                continue
            matched = _try_match_mobilenetv4_uib(layers, index, shape)
            if matched is None:
                index += 1
                continue
            fused_layer, consumed = matched
            layers[index : index + consumed] = [fused_layer]
            current_arch, current_weights = _compose_cnn(current_arch, layers)
            if verify_output and baseline is not None:
                for probe, expected in zip(probes, baseline):
                    actual = np.asarray(forward_cnn(probe, current_arch, current_weights), dtype=np.float32)
                    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
                        raise ValueError("UIB fusion changed model output beyond tolerance")
            applied.append("mobilenetv4_uib")
            changed = True
            break
        if not changed:
            break

    seen: set[str] = set()
    unique_applied: list[str] = []
    for name in applied:
        if name not in seen:
            seen.add(name)
            unique_applied.append(name)

    return FuseArchResult(arch=current_arch, weights=current_weights, applied=unique_applied)
