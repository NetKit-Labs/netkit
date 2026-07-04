"""Encode/decode asymmetric ONNX padding and non-square pool metadata in .nk layer bytes."""

from __future__ import annotations


def _attr_ints(node, name: str, default: list[int] | None = None) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return default or []


def onnx_spatial_pads(node) -> tuple[int, int, int, int]:
    """Return ONNX pads as (top, left, bottom, right)."""
    pads = _attr_ints(node, "pads", [0, 0, 0, 0])
    if len(pads) < 2:
        return 0, 0, 0, 0
    if len(pads) < 4:
        top, left = int(pads[0]), int(pads[1])
        return top, left, top, left
    return int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])


def encode_pad_extra(top: int, left: int, bottom: int, right: int) -> int:
    """Pack extra bottom/right padding into one byte (Conv2D kernel_w / reserved slot)."""
    extra_h = bottom - top
    extra_w = right - left
    if extra_h < 0 or extra_w < 0 or extra_h > 15 or extra_w > 15:
        raise ValueError(
            f"asymmetric padding out of range (max +15 per side): "
            f"top={top} left={left} bottom={bottom} right={right}"
        )
    return ((extra_w & 0xF) << 4) | (extra_h & 0xF)


def decode_pad_extra(pad_h: int, pad_w: int, extra: int) -> tuple[int, int, int, int]:
    """Return (top, left, bottom, right) from stored pad_h/pad_w and extra byte."""
    pad_h_end = pad_h + (extra & 0xF)
    pad_w_end = pad_w + ((extra >> 4) & 0xF)
    return pad_h, pad_w, pad_h_end, pad_w_end


def spatial_output_dim(size: int, kernel: int, stride: int, pad_before: int, pad_after: int) -> int:
    return (size + pad_before + pad_after - kernel) // stride + 1


def encode_pool_reserved(
    *,
    pool_h: int,
    pool_w: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> int:
    """Pack non-square pool kernel and asymmetric pad extras into pool.reserved u16."""
    extra_h = bottom - top
    extra_w = right - left
    if extra_h > 15 or extra_w > 15:
        raise ValueError(
            f"asymmetric pool padding out of range: top={top} left={left} bottom={bottom} right={right}"
        )
    w_byte = pool_w if pool_w != pool_h else 0
    if w_byte > 255:
        raise ValueError(f"pool_w must fit in 8 bits, got {pool_w}")
    return w_byte | (extra_h << 8) | (extra_w << 12)


def decode_pool_reserved(
    pool_h: int,
    pad_h: int,
    pad_w: int,
    reserved: int,
) -> tuple[int, int, int, int, int, int]:
    """Return (pool_h, pool_w, top, left, bottom, right)."""
    pool_w = reserved & 0xFF if (reserved & 0xFF) else pool_h
    extra_h = (reserved >> 8) & 0xF
    extra_w = (reserved >> 12) & 0xF
    top, left = pad_h, pad_w
    bottom = top + extra_h
    right = left + extra_w
    return pool_h, pool_w, top, left, bottom, right
