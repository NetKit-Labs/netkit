"""Extract TFLite int8 quant params and align netkit CNN quantization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .writer import QuantLayerParams


def _tensor_scale_zp(detail: dict) -> tuple[float, int]:
    q = detail.get("quantization_parameters") or {}
    scales = q.get("scales")
    zps = q.get("zero_points")
    if scales is None or len(scales) == 0:
        return 1.0, 0
    scale = float(scales[0])
    zp = int(zps[0]) if zps is not None and len(zps) else 0
    return scale, zp


def _import_tflite_interpreter():
    try:
        import tensorflow as tf

        return tf.lite.Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError as exc:
        raise SystemExit(
            "tensorflow or tflite_runtime required for TFLite quant alignment"
        ) from exc


def extract_tflite_cnn_quant_specs(tflite_path: str | Path) -> list[QuantLayerParams]:
    """Return per-layer quant params for CONV_2D and FULLY_CONNECTED ops in graph order."""
    Interpreter = _import_tflite_interpreter()
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    tensors = interp.get_tensor_details()
    specs: list[QuantLayerParams] = []

    for op in interp._get_ops_details():
        name = op.get("op_name", "")
        if name not in ("CONV_2D", "FULLY_CONNECTED"):
            continue
        inputs = op.get("inputs")
        outputs = op.get("outputs")
        if inputs is None or outputs is None:
            continue
        inputs = list(inputs)
        outputs = list(outputs)
        if not inputs or not outputs:
            continue

        in_detail = tensors[inputs[0]]
        weight_detail = tensors[inputs[1]]
        out_detail = tensors[outputs[0]]

        input_scale, input_zp = _tensor_scale_zp(in_detail)
        weight_scale, weight_zp = _tensor_scale_zp(weight_detail)
        output_scale, output_zp = _tensor_scale_zp(out_detail)
        bias_scale = input_scale * weight_scale
        if bias_scale <= 0.0:
            bias_scale = 1.0

        specs.append(
            QuantLayerParams(
                input_scale=input_scale,
                input_zero_point=input_zp,
                weight_scale=weight_scale,
                weight_zero_point=weight_zp,
                bias_scale=bias_scale,
                bias_zero_point=0,
                output_scale=output_scale,
                output_zero_point=output_zp,
            )
        )

    if not specs:
        raise ValueError(f"no quantized conv/fc ops found in {tflite_path}")
    return specs


def quant_specs_to_json(specs: list[QuantLayerParams]) -> str:
    payload = [
        {
            "input_scale": s.input_scale,
            "input_zero_point": s.input_zero_point,
            "weight_scale": s.weight_scale,
            "weight_zero_point": s.weight_zero_point,
            "bias_scale": s.bias_scale,
            "bias_zero_point": s.bias_zero_point,
            "output_scale": s.output_scale,
            "output_zero_point": s.output_zero_point,
        }
        for s in specs
    ]
    return json.dumps(payload, indent=2)


def quant_specs_from_json(text: str) -> list[QuantLayerParams]:
    data = json.loads(text)
    return [QuantLayerParams(**entry) for entry in data]


def write_tflite_quant_json(tflite_path: Path, out_json: Path) -> list[QuantLayerParams]:
    specs = extract_tflite_cnn_quant_specs(tflite_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(quant_specs_to_json(specs) + "\n", encoding="utf-8")
    return specs


def load_tflite_quant_json(path: Path) -> list[QuantLayerParams]:
    return quant_specs_from_json(path.read_text(encoding="utf-8"))


def count_quantizable_layers(arch: dict[str, Any]) -> int:
    return sum(1 for layer in arch["layers"] if layer["type"] in ("conv2d", "dense"))


def validate_specs_for_arch(arch: dict[str, Any], specs: list[QuantLayerParams]) -> None:
    expected = count_quantizable_layers(arch)
    if len(specs) != expected:
        raise ValueError(f"expected {expected} TFLite quant layers, got {len(specs)}")
