"""Post-training int8 quantization for MLP and CNN models."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

import numpy as np

from .arch_writer import _split_cnn_weights
from .cnn_layers import conv2d_input_channels
from .reference_forward import _activate, _max_pool_nhwc, _out_dim
from .format import DType, activation_from_name
from .writer import LayerSpec, ModelSpec, QuantLayerParams


@dataclass
class QuantizedMlpLayer:
    weight: np.ndarray
    bias: np.ndarray
    quant: QuantLayerParams


@dataclass
class QuantizedMlpPack:
    weight_tensors: list[np.ndarray]
    bias_tensors: list[np.ndarray]
    quant_layers: list[QuantLayerParams]


@dataclass
class QuantizedCnnPack:
    weight_tensors: list[np.ndarray]
    bias_tensors: list[np.ndarray]
    quant_layers: list[QuantLayerParams]


def _symmetric_scale(values: np.ndarray) -> tuple[float, int]:
    amax = float(np.max(np.abs(values)))
    if amax == 0.0:
        return 1.0, 0
    scale = amax / 127.0
    return scale, 0


def quantize_symmetric_int8(values: np.ndarray) -> tuple[np.ndarray, float, int]:
    scale, zero_point = _symmetric_scale(values)
    if scale <= 0.0:
        return np.zeros_like(values, dtype=np.int8), 1.0, 0
    q = np.clip(np.round(values / scale), -128, 127).astype(np.int8)
    return q, scale, zero_point


def quantize_float_input(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0.0:
        return np.zeros_like(values, dtype=np.int8)
    q = np.round(values / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def _split_mlp_layers(arch: dict[str, Any], flat_weights: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    offset = 0
    in_features = arch["input"][1]
    layers: list[tuple[np.ndarray, np.ndarray]] = []

    for layer in arch["layers"]:
        out_features = layer["units"]
        w_size = in_features * out_features
        w = flat_weights[offset : offset + w_size].reshape(out_features, in_features).astype(np.float32)
        offset += w_size
        b = flat_weights[offset : offset + out_features].astype(np.float32)
        offset += out_features
        layers.append((w, b))
        in_features = out_features

    return layers


def _fc_quant_int8(
    input_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    quant: QuantLayerParams,
    *,
    apply_relu: bool,
) -> np.ndarray:
    in_q = input_q.astype(np.int32) - quant.input_zero_point
    wt_q = weight_q.astype(np.int32) - quant.weight_zero_point
    effective_scale = quant.input_scale * quant.weight_scale
    acc = in_q @ wt_q.T + bias_q.astype(np.int32)
    out_real = acc.astype(np.float32) * effective_scale
    if apply_relu:
        out_real = np.maximum(0.0, out_real)
    return out_real


SOFTMAX_OUTPUT_SCALE = 1.0 / 256.0
SOFTMAX_OUTPUT_ZERO_POINT = -128
_SCALED_DIFF_INTEGER_BITS = 5
_ACCUM_BITS = 12


def _quantize_multiplier(double_multiplier: float) -> tuple[int, int]:
    if double_multiplier <= 0.0:
        return 0, 0
    shift = 0
    q = np.frexp(double_multiplier)
    mantissa, exp = q[0], int(q[1])
    q_fixed = int(np.round(mantissa * (1 << 31)))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        exp += 1
    if exp < -31:
        exp = 0
        q_fixed = 0
    if exp > 30:
        exp = 30
        q_fixed = (1 << 31) - 1
    return int(q_fixed), exp


def _calculate_input_radius(input_integer_bits: int, input_left_shift: int) -> int:
    max_input_rescaled = 255.0 * float(1 << (input_left_shift + input_integer_bits))
    radius = int(np.round(max_input_rescaled))
    return struct.unpack("i", struct.pack("I", radius & 0xFFFFFFFF))[0]


def _softmax_s8_params(logit_scale: float, beta: float = 1.0) -> tuple[int, int, int]:
    if logit_scale <= 0.0:
        return 0, 0, 0
    min_real = min(
        beta * logit_scale * float(1 << (31 - _SCALED_DIFF_INTEGER_BITS)),
        float((1 << 31) - 1),
    )
    mult, shift = _quantize_multiplier(min_real)
    diff_min = -_calculate_input_radius(_SCALED_DIFF_INTEGER_BITS, shift)
    return mult, shift, diff_min


def _doubling_high_mult(m1: int, m2: int) -> int:
    mult = 1 << 30
    if (m1 < 0) ^ (m2 < 0):
        mult = 1 - mult
    mult = mult + int(m1) * int(m2)
    return int(mult >> 31)


def _divide_by_power_of_two(dividend: int, exponent: int) -> int:
    shift = -exponent
    fixup = (dividend & shift) >> 31
    fixed = dividend + fixup
    return int((fixed + (1 << (-shift - 1))) >> (-shift))


def _exp_on_negative_values(val: int) -> int:
    shift = 24
    val_mod = (val & ((1 << shift) - 1)) - (1 << shift)
    remainder = val_mod - val
    x = (val_mod << 5) + (1 << 28)
    x2 = _doubling_high_mult(x, x)
    op_1 = _divide_by_power_of_two(_doubling_high_mult(x2, x2), 2) + _doubling_high_mult(x2, x)
    op_2 = x + _divide_by_power_of_two(_doubling_high_mult(op_1, 715827883) + x2, 1)
    result = 1895147668 + _doubling_high_mult(1895147668, op_2)
    for sel in (1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242):
        if (remainder & (1 << shift)) != 0:
            result = _doubling_high_mult(result, sel)
        shift += 1
    if val == 0:
        result = 0x7FFFFFFF
    return result


def _one_over_one_plus_x(val: int) -> int:
    total = int(val) + (1 << 31)
    if total <= (1 << 31):
        return 0x7FFFFFFF
    return int(((1 << 62) + (total >> 1)) // total)


def _clz32(value: int) -> int:
    if value == 0:
        return 32
    return 32 - int(value).bit_length()


def softmax_s8(logits_i8: np.ndarray, logit_scale: float) -> np.ndarray:
    logits = np.asarray(logits_i8, dtype=np.int8).reshape(-1)
    mult, shift, diff_min = _softmax_s8_params(logit_scale if logit_scale > 0.0 else 1.0)
    mask = 1 << shift
    max_val = int(np.max(logits))
    sum_val = 0
    for value in logits:
        diff = int(value) - max_val
        if diff >= diff_min:
            sum_val += _divide_by_power_of_two(
                _exp_on_negative_values(_doubling_high_mult(diff * mask, mult)),
                _ACCUM_BITS,
            )
    headroom = _clz32(sum_val)
    shifted_scale = _one_over_one_plus_x((sum_val << headroom if sum_val > 0 else 0) - (1 << 31))
    bits_over_unit = _ACCUM_BITS - headroom + 23
    output = np.empty(logits.shape[0], dtype=np.int8)
    for i, value in enumerate(logits):
        diff = int(value) - max_val
        if diff >= diff_min:
            res = (
                _divide_by_power_of_two(
                    _doubling_high_mult(
                        shifted_scale,
                        _exp_on_negative_values(_doubling_high_mult(diff * mask, mult)),
                    ),
                    bits_over_unit,
                )
                + SOFTMAX_OUTPUT_ZERO_POINT
            )
            output[i] = np.int8(np.clip(res, -128, 127))
        else:
            output[i] = np.int8(SOFTMAX_OUTPUT_ZERO_POINT)
    return output


def dequant_softmax_output(probs_i8: np.ndarray) -> np.ndarray:
    values = np.asarray(probs_i8, dtype=np.float32)
    return (values - SOFTMAX_OUTPUT_ZERO_POINT) * SOFTMAX_OUTPUT_SCALE


def forward_quantized_mlp(
    flat_input: np.ndarray,
    arch: dict[str, Any],
    pack: QuantizedMlpPack,
    *,
    output_float: bool = False,
) -> np.ndarray:
    x_float = np.asarray(flat_input, dtype=np.float32).reshape(-1)
    x_i8: np.ndarray | None = None

    for layer_idx, (layer_spec, weight_q, bias_q, quant) in enumerate(
        zip(arch["layers"], pack.weight_tensors, pack.bias_tensors, pack.quant_layers)
    ):
        activation = layer_spec.get("activation", "none")
        apply_relu = activation == "relu"
        is_last = layer_idx == len(arch["layers"]) - 1

        if x_i8 is None:
            input_q = quantize_float_input(x_float, quant.input_scale, quant.input_zero_point)
        else:
            input_q = x_i8

        out_real = _fc_quant_int8(input_q, weight_q, bias_q, quant, apply_relu=apply_relu)
        if is_last:
            if activation == "softmax":
                logit_scale = quant.output_scale if quant.output_scale > 0.0 else 1.0
                logits_i8 = np.clip(np.round(out_real / logit_scale), -128, 127).astype(np.int8)
                probs_i8 = softmax_s8(logits_i8, logit_scale)
                if output_float:
                    return dequant_softmax_output(probs_i8).astype(np.float32).reshape(-1)
                return probs_i8.astype(np.int8).reshape(-1)
            return _activate(out_real, activation).astype(np.float32).reshape(-1)

        if quant.output_scale <= 0.0:
            raise ValueError("hidden layer output_scale must be positive")
        q = np.round(out_real / quant.output_scale) + quant.output_zero_point
        x_i8 = np.clip(q, -128, 127).astype(np.int8)

    return x_float


def quantize_mlp(
    arch: dict[str, Any],
    flat_weights: np.ndarray,
    calibration_inputs: np.ndarray,
    *,
    num_calibration: int = 256,
) -> QuantizedMlpPack:
    if arch.get("network") != "mlp":
        raise ValueError("quantize_mlp only supports MLP architectures")

    cal = np.asarray(calibration_inputs, dtype=np.float32)
    if cal.ndim == 1:
        cal = cal.reshape(1, -1)
    if cal.shape[0] > num_calibration:
        cal = cal[:num_calibration]

    float_layers = _split_mlp_layers(arch, flat_weights)
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    quant_layers: list[QuantLayerParams] = []

    input_scale, input_zp = _symmetric_scale(cal.reshape(-1))

    hidden_real = cal.copy()
    for layer_idx, (w_float, b_float) in enumerate(float_layers):
        layer_spec = arch["layers"][layer_idx]
        activation = layer_spec.get("activation", "none")
        apply_relu = activation == "relu"
        is_last = layer_idx == len(float_layers) - 1

        weight_q, weight_scale, weight_zp = quantize_symmetric_int8(w_float)
        effective_scale = input_scale * weight_scale
        if effective_scale <= 0.0:
            effective_scale = 1.0
        bias_q = np.round(b_float / effective_scale).astype(np.int32)

        quant = QuantLayerParams(
            input_scale=input_scale,
            input_zero_point=input_zp,
            weight_scale=weight_scale,
            weight_zero_point=weight_zp,
            bias_scale=effective_scale,
            bias_zero_point=0,
            output_scale=1.0,
            output_zero_point=0,
        )

        layer_outputs: list[np.ndarray] = []
        for sample in hidden_real:
            input_q = quantize_float_input(sample, input_scale, input_zp)
            layer_outputs.append(
                _fc_quant_int8(input_q, weight_q, bias_q, quant, apply_relu=apply_relu)
            )
        layer_outputs_arr = np.stack(layer_outputs, axis=0)

        if is_last:
            output_scale = 1.0
            output_zp = 0
            hidden_real = layer_outputs_arr
        else:
            output_scale, output_zp = _symmetric_scale(layer_outputs_arr.reshape(-1))
            hidden_real = layer_outputs_arr

        quant.output_scale = output_scale
        quant.output_zero_point = output_zp

        weight_tensors.append(weight_q)
        bias_tensors.append(bias_q)
        quant_layers.append(quant)

        if not is_last:
            input_scale = output_scale
            input_zp = output_zp

    return QuantizedMlpPack(
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
        quant_layers=quant_layers,
    )


def quantized_mlp_to_spec(
    arch: dict[str, Any],
    pack: QuantizedMlpPack,
) -> ModelSpec:
    layers: list[LayerSpec] = []
    for layer in arch["layers"]:
        layers.append(
            LayerSpec(
                kind="dense",
                units=layer["units"],
                activation=activation_from_name(layer.get("activation", "none")),
                alpha=float(layer.get("alpha", 0.01)),
            )
        )

    return ModelSpec(
        network="mlp",
        input_shape=list(arch["input"]),
        layers=layers,
        weight_tensors=pack.weight_tensors,
        bias_tensors=pack.bias_tensors,
        weight_dtypes=[DType.INT8] * len(pack.weight_tensors),
        bias_dtypes=[DType.INT32] * len(pack.bias_tensors),
        quant_layers=pack.quant_layers,
    )


def _conv2d_quant_nhwc(
    input_q: np.ndarray,
    weight_q: np.ndarray,
    bias_q: np.ndarray,
    quant: QuantLayerParams,
    *,
    kernel_size: int,
    stride: int,
    pad_h: int,
    pad_w: int,
    pad_h_end: int,
    pad_w_end: int,
    apply_relu: bool,
) -> np.ndarray:
    in_h, in_w, in_c = input_q.shape
    out_c = weight_q.shape[0]
    out_h = _out_dim(in_h, kernel_size, stride, pad_h, pad_h_end)
    out_w = _out_dim(in_w, kernel_size, stride, pad_w, pad_w_end)
    out = np.zeros((out_h, out_w, out_c), dtype=np.int8)
    effective_scale = quant.input_scale * quant.weight_scale

    for oh in range(out_h):
        for ow in range(out_w):
            for oc in range(out_c):
                acc = int(bias_q[oc])
                for kh in range(kernel_size):
                    ih = oh * stride + kh - pad_h
                    if ih < 0 or ih >= in_h:
                        continue
                    for kw in range(kernel_size):
                        iw = ow * stride + kw - pad_w
                        if iw < 0 or iw >= in_w:
                            continue
                        in_row = input_q[ih, iw].astype(np.int32) - quant.input_zero_point
                        wt_row = weight_q[oc, kh, kw].astype(np.int32) - quant.weight_zero_point
                        acc += int(np.dot(in_row, wt_row))
                out_real = float(acc) * effective_scale
                if apply_relu:
                    out_real = max(0.0, out_real)
                out[oh, ow, oc] = np.clip(
                    int(round(out_real / quant.output_scale)) + quant.output_zero_point, -128, 127
                )
    return out


def _max_pool_quant_nhwc(
    input_q: np.ndarray,
    *,
    pool_h: int,
    pool_w: int,
    stride: int,
    pad_h: int,
    pad_w: int,
    pad_h_end: int,
    pad_w_end: int,
) -> np.ndarray:
    in_h, in_w, in_c = input_q.shape
    out_h = _out_dim(in_h, pool_h, stride, pad_h, pad_h_end)
    out_w = _out_dim(in_w, pool_w, stride, pad_w, pad_w_end)
    out = np.zeros((out_h, out_w, in_c), dtype=np.int8)
    for oh in range(out_h):
        for ow in range(out_w):
            for c in range(in_c):
                best = -128
                found = False
                for kh in range(pool_h):
                    ih = oh * stride + kh - pad_h
                    if ih < 0 or ih >= in_h:
                        continue
                    for kw in range(pool_w):
                        iw = ow * stride + kw - pad_w
                        if iw < 0 or iw >= in_w:
                            continue
                        val = int(input_q[ih, iw, c])
                        if not found or val > best:
                            best = val
                            found = True
                out[oh, ow, c] = best if found else 0
    return out


def forward_quantized_cnn(
    flat_input: np.ndarray,
    arch: dict[str, Any],
    pack: QuantizedCnnPack,
    *,
    output_float: bool = False,
) -> np.ndarray:
    h, w, channels = arch["input"]
    x = np.asarray(flat_input, dtype=np.float32).reshape(h, w, channels)
    x_i8: np.ndarray | None = None
    x_2d: np.ndarray | None = None
    quant_idx = 0

    for layer in arch["layers"]:
        layer_type = layer["type"]

        if layer_type == "conv2d":
            quant = pack.quant_layers[quant_idx]
            weight_q = pack.weight_tensors[quant_idx]
            bias_q = pack.bias_tensors[quant_idx]
            if x_i8 is None:
                x_q = quantize_float_input(x.reshape(-1), quant.input_scale, quant.input_zero_point)
                x_i8 = x_q.reshape(h, w, channels)
            k = layer["kernel_size"]
            pad_h = layer.get("pad_h", 0)
            pad_w = layer.get("pad_w", 0)
            pad_h_end = layer.get("pad_h_end", pad_h)
            pad_w_end = layer.get("pad_w_end", pad_w)
            x_i8 = _conv2d_quant_nhwc(
                x_i8,
                weight_q.reshape(layer["filters"], k, k, -1),
                bias_q,
                quant,
                kernel_size=k,
                stride=layer.get("stride", 1),
                pad_h=pad_h,
                pad_w=pad_w,
                pad_h_end=pad_h_end,
                pad_w_end=pad_w_end,
                apply_relu=layer.get("activation") == "relu",
            )
            h, w, channels = x_i8.shape
            x_2d = None
            quant_idx += 1
        elif layer_type == "max_pool2d":
            pool_h = layer["pool_size"]
            pool_w = layer.get("pool_w", pool_h)
            x_i8 = _max_pool_quant_nhwc(
                x_i8,
                pool_h=pool_h,
                pool_w=pool_w,
                stride=layer.get("stride", pool_h),
                pad_h=layer.get("pad_h", 0),
                pad_w=layer.get("pad_w", 0),
                pad_h_end=layer.get("pad_h_end", layer.get("pad_h", 0)),
                pad_w_end=layer.get("pad_w_end", layer.get("pad_w", 0)),
            )
            h, w, channels = x_i8.shape
        elif layer_type == "flatten":
            x_2d = x_i8.reshape(1, -1)
            x_i8 = None
        elif layer_type == "dense":
            quant = pack.quant_layers[quant_idx]
            weight_q = pack.weight_tensors[quant_idx]
            bias_q = pack.bias_tensors[quant_idx]
            if x_2d is None:
                raise ValueError("dense layer expects flattened input")
            out_real = _fc_quant_int8(
                x_2d.reshape(-1),
                weight_q,
                bias_q,
                quant,
                apply_relu=layer.get("activation") == "relu",
            )
            if layer.get("activation") == "softmax":
                logit_scale = quant.output_scale if quant.output_scale > 0.0 else 1.0
                logits_i8 = np.clip(np.round(out_real / logit_scale), -128, 127).astype(np.int8)
                probs_i8 = softmax_s8(logits_i8, logit_scale)
                if output_float:
                    return dequant_softmax_output(probs_i8).astype(np.float32).reshape(-1)
                return probs_i8.astype(np.int8).reshape(-1)
            q = np.round(out_real / quant.output_scale) + quant.output_zero_point
            x_2d = np.clip(q, -128, 127).astype(np.int8).reshape(1, -1)
            quant_idx += 1
        else:
            raise ValueError(f"unsupported quantized CNN layer: {layer_type}")

    return x.reshape(-1).astype(np.float32)


def quantize_cnn(
    arch: dict[str, Any],
    flat_weights: np.ndarray,
    calibration_inputs: np.ndarray,
    *,
    num_calibration: int = 256,
    aligned_quants: list[QuantLayerParams] | None = None,
) -> QuantizedCnnPack:
    if arch.get("network") != "cnn":
        raise ValueError("quantize_cnn only supports CNN architectures")

    cal = np.asarray(calibration_inputs, dtype=np.float32)
    if cal.ndim == 1:
        cal = cal.reshape(1, -1)
    if cal.shape[0] > num_calibration:
        cal = cal[:num_calibration]

    float_weights, float_biases = _split_cnn_weights(arch, flat_weights)
    weight_tensors: list[np.ndarray] = []
    bias_tensors: list[np.ndarray] = []
    quant_layers: list[QuantLayerParams] = []

    if aligned_quants is not None:
        expected = sum(1 for layer in arch["layers"] if layer["type"] in ("conv2d", "dense"))
        if len(aligned_quants) != expected:
            raise ValueError(f"aligned_quants length {len(aligned_quants)} != {expected}")

    h, w, channels = arch["input"]
    if aligned_quants is not None:
        input_scale = aligned_quants[0].input_scale
        input_zp = aligned_quants[0].input_zero_point
    else:
        input_scale, input_zp = _symmetric_scale(cal.reshape(-1))

    hidden_samples = [sample.reshape(h, w, channels) for sample in cal]
    tensor_idx = 0

    for layer in arch["layers"]:
        layer_type = layer["type"]
        if layer_type not in ("conv2d", "dense"):
            if layer_type == "max_pool2d":
                pool_h = layer["pool_size"]
                pool_w = layer.get("pool_w", pool_h)
                hidden_samples = [
                    _max_pool_nhwc(
                        sample,
                        pool_h=pool_h,
                        pool_w=pool_w,
                        stride=layer.get("stride", pool_h),
                        pad_h=layer.get("pad_h", 0),
                        pad_w=layer.get("pad_w", 0),
                        pad_h_end=layer.get("pad_h_end", layer.get("pad_h", 0)),
                        pad_w_end=layer.get("pad_w_end", layer.get("pad_w", 0)),
                    )
                    for sample in hidden_samples
                ]
            elif layer_type == "flatten":
                hidden_samples = [sample.reshape(-1) for sample in hidden_samples]
            continue

        w_float = float_weights[tensor_idx]
        b_float = float_biases[tensor_idx]
        if aligned_quants is not None:
            prescribed = aligned_quants[tensor_idx]
            if tensor_idx == 0:
                input_scale = prescribed.input_scale
                input_zp = prescribed.input_zero_point
            # Later layers keep input_scale/input_zp chained from calibrated outputs.
            # TFLite weight/output scales target TFLite's own weights — not netkit's.
            weight_q, weight_scale, weight_zp = quantize_symmetric_int8(w_float)
            effective_scale = input_scale * weight_scale
            if effective_scale <= 0.0:
                effective_scale = 1.0
            bias_q = np.round(b_float / effective_scale).astype(np.int32)
            quant = QuantLayerParams(
                input_scale=input_scale,
                input_zero_point=input_zp,
                weight_scale=weight_scale,
                weight_zero_point=weight_zp,
                bias_scale=effective_scale,
                bias_zero_point=0,
                output_scale=1.0,
                output_zero_point=0,
            )
        else:
            weight_q, weight_scale, weight_zp = quantize_symmetric_int8(w_float)
            effective_scale = input_scale * weight_scale
            if effective_scale <= 0.0:
                effective_scale = 1.0
            bias_q = np.round(b_float / effective_scale).astype(np.int32)

            quant = QuantLayerParams(
                input_scale=input_scale,
                input_zero_point=input_zp,
                weight_scale=weight_scale,
                weight_zero_point=weight_zp,
                bias_scale=effective_scale,
                bias_zero_point=0,
                output_scale=1.0,
                output_zero_point=0,
            )

        next_samples: list[np.ndarray] = []
        if layer_type == "conv2d":
            k = layer["kernel_size"]
            w_q = weight_q.reshape(layer["filters"], k, k, -1)
            for sample in hidden_samples:
                input_q = quantize_float_input(sample.reshape(-1), input_scale, input_zp).reshape(
                    sample.shape
                )
                out = _conv2d_quant_nhwc(
                    input_q,
                    w_q,
                    bias_q,
                    quant,
                    kernel_size=k,
                    stride=layer.get("stride", 1),
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                    pad_h_end=layer.get("pad_h_end", layer.get("pad_h", 0)),
                    pad_w_end=layer.get("pad_w_end", layer.get("pad_w", 0)),
                    apply_relu=layer.get("activation") == "relu",
                )
                next_samples.append(out)
        else:
            w_q = weight_q.reshape(layer["units"], -1)
            for sample in hidden_samples:
                input_q = quantize_float_input(sample.reshape(-1), input_scale, input_zp)
                out = _fc_quant_int8(
                    input_q, w_q, bias_q, quant, apply_relu=layer.get("activation") == "relu"
                )
                next_samples.append(out)

        if layer.get("activation") == "softmax":
            output_scale, output_zp = 1.0, 0
        else:
            stacked = np.stack(next_samples, axis=0)
            output_scale, output_zp = _symmetric_scale(stacked.reshape(-1))

        quant.output_scale = output_scale
        quant.output_zero_point = output_zp
        weight_tensors.append(weight_q)
        bias_tensors.append(bias_q)
        quant_layers.append(quant)
        hidden_samples = next_samples
        input_scale = output_scale
        input_zp = output_zp
        tensor_idx += 1

    return QuantizedCnnPack(
        weight_tensors=weight_tensors,
        bias_tensors=bias_tensors,
        quant_layers=quant_layers,
    )


def quantized_cnn_to_spec(arch: dict[str, Any], pack: QuantizedCnnPack) -> ModelSpec:
    layers: list[LayerSpec] = []
    for layer in arch["layers"]:
        layer_type = layer["type"]
        if layer_type == "conv2d":
            layers.append(
                LayerSpec(
                    kind="conv2d",
                    kernel_size=layer["kernel_size"],
                    stride=layer.get("stride", 1),
                    filters=layer["filters"],
                    activation=activation_from_name(layer.get("activation", "none")),
                    alpha=float(layer.get("alpha", 0.01)),
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                    pad_h_end=layer.get("pad_h_end", 0),
                    pad_w_end=layer.get("pad_w_end", 0),
                )
            )
        elif layer_type == "max_pool2d":
            layers.append(
                LayerSpec(
                    kind="max_pool2d",
                    pool_size=layer["pool_size"],
                    stride=layer.get("stride", layer["pool_size"]),
                    pad_h=layer.get("pad_h", 0),
                    pad_w=layer.get("pad_w", 0),
                )
            )
        elif layer_type == "flatten":
            layers.append(LayerSpec(kind="flatten"))
        elif layer_type == "dense":
            layers.append(
                LayerSpec(
                    kind="dense",
                    units=layer["units"],
                    activation=activation_from_name(layer.get("activation", "none")),
                    alpha=float(layer.get("alpha", 0.01)),
                )
            )
        else:
            raise ValueError(f"unsupported CNN layer for int8 export: {layer_type}")

    return ModelSpec(
        network="cnn",
        input_shape=list(arch["input"]),
        layers=layers,
        weight_tensors=pack.weight_tensors,
        bias_tensors=pack.bias_tensors,
        weight_dtypes=[DType.INT8] * len(pack.weight_tensors),
        bias_dtypes=[DType.INT32] * len(pack.bias_tensors),
        quant_layers=pack.quant_layers,
    )
