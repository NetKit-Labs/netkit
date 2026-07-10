#!/usr/bin/env python3
"""Generate models/mobilenetv4_imagenet_int8.nk from the float32 ImageNet pack.

Calibrates with the same 10 ImageNet-preprocessed samples used by the host benches
(benchmark/tflm/generated/imagenet_sample_cache/), optionally augmented with light
crops/flips for stabler activation ranges.

PTQ uses TFLite-style affine (min/max) input quant by default. Pass
--align-tflite-input to pin the first-layer input scale/zp to the peer .tflite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmark" / "tflm" / "tools"))

from export_imagenet_mnv4_test_images import CACHE, SAMPLES, _load_rgb, preprocess  # noqa: E402
from netkit.quantize import forward_quantized_cnn, quantize_cnn, quantized_cnn_to_spec  # noqa: E402
from netkit.reader import read_nk  # noqa: E402
from netkit.writer import write_nk  # noqa: E402

SRC = ROOT / "models" / "mobilenetv4_imagenet_f32.nk"
OUT = ROOT / "models" / "mobilenetv4_imagenet_int8.nk"
DEFAULT_TFLITE = ROOT / "benchmark" / "tflm" / "generated" / "mobilenetv4_imagenet_int8.tflite"
DEFAULT_TFLITE_QUANT_JSON = DEFAULT_TFLITE.with_name("mobilenetv4_imagenet_int8_tflite_quant.json")


def _tflite_input_quant(tflite_path: Path, quant_json: Path) -> tuple[float, int]:
    if quant_json.is_file():
        data = json.loads(quant_json.read_text(encoding="utf-8"))
        if data:
            first = data[0]
            return float(first["input_scale"]), int(first["input_zero_point"])

    try:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise SystemExit(
                f"missing {quant_json} and no TFLite interpreter to read {tflite_path}"
            ) from exc

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    scale, zp = inp["quantization"]
    quant_json.write_text(
        json.dumps(
            [{"input_scale": float(scale), "input_zero_point": int(zp)}],
            indent=2,
        ),
        encoding="utf-8",
    )
    return float(scale), int(zp)


def calibration_samples(*, augment: bool) -> tuple[np.ndarray, list[int]]:
    """Base 10 bench images; optional light augmentations for richer calib ranges."""
    samples = []
    labels = []
    for filename, _url, label, _short in SAMPLES:
        path = CACHE / filename
        if not path.is_file():
            raise SystemExit(
                f"missing {path} — run: make -C benchmark/tflm export-imagenet-mnv4-images"
            )
        rgb = _load_rgb(path)
        base = preprocess(rgb)
        samples.append(base)
        labels.append(int(label))
        if augment:
            # Horizontal flip (common ImageNet TTA; keeps mean/std preprocess).
            flipped = preprocess(np.ascontiguousarray(rgb[:, ::-1, :]))
            samples.append(flipped)
            labels.append(int(label))
            # Slightly tighter / looser center crops via crop_pct variants.
            from export_imagenet_mnv4_test_images import _center_crop_resize

            for crop_pct in (0.85, 0.90):
                cropped = _center_crop_resize(rgb, 224, crop_pct=crop_pct).astype(np.float32)
                cropped /= 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                cropped = (cropped - mean) / std
                samples.append(cropped.astype(np.float32))
                labels.append(int(label))
    return np.stack(samples, axis=0), labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--align-tflite-input",
        action="store_true",
        help="Pin first-layer input scale/zp to the peer TFLite model",
    )
    parser.add_argument(
        "--tflite",
        type=Path,
        default=DEFAULT_TFLITE,
        help="TFLite model used with --align-tflite-input",
    )
    parser.add_argument(
        "--augment-calib",
        action="store_true",
        default=True,
        help="Augment the 10 bench images (flip + crop variants); default on",
    )
    parser.add_argument(
        "--no-augment-calib",
        action="store_false",
        dest="augment_calib",
        help="Calibrate on the exact 10 bench images only",
    )
    parser.add_argument("-o", "--output", type=Path, default=OUT)
    args = parser.parse_args()

    if not SRC.is_file():
        raise SystemExit(
            f"missing {SRC} — run:\n"
            "  python -m netkit pack --arch mobilenetv4_small --pretrained "
            "-o models/mobilenetv4_imagenet_f32.nk --height 224 --width 224 --num-classes 1000"
        )

    print(f"reading {SRC} ...")
    arch, weights = read_nk(SRC)
    cal, labels = calibration_samples(augment=args.augment_calib)
    # Eval top-1 only on the first occurrence of each bench image (unaugmented).
    eval_n = len(SAMPLES)
    print(
        f"calibrating with {len(cal)} samples "
        f"({cal.shape[1:]} each; eval on first {eval_n}) ..."
    )

    pin_scale = None
    pin_zp = None
    if args.align_tflite_input:
        if not args.tflite.is_file() and not DEFAULT_TFLITE_QUANT_JSON.is_file():
            raise SystemExit(
                f"missing {args.tflite} — run: make -C benchmark/tflm export-mobilenetv4-imagenet-int8"
            )
        pin_scale, pin_zp = _tflite_input_quant(args.tflite, DEFAULT_TFLITE_QUANT_JSON)
        print(f"pinning input quant to TFLite: scale={pin_scale} zp={pin_zp}")

    pack = quantize_cnn(
        arch,
        weights,
        cal,
        num_calibration=len(cal),
        input_scale=pin_scale,
        input_zero_point=pin_zp,
    )
    q0 = pack.quant_layers[0]
    print(f"layer-0 input quant: scale={q0.input_scale} zp={q0.input_zero_point}")

    correct = 0
    for i in range(eval_n):
        sample, label = cal[i], labels[i]
        logits = forward_quantized_cnn(sample.reshape(-1), arch, pack, output_float=True)
        pred = int(np.argmax(logits))
        ok = pred == label
        correct += int(ok)
        print(f"  [{i}] label={label} pred={pred} {'OK' if ok else 'MISS'}")
    print(f"quantized top-1: {correct}/{eval_n}")
    if correct < eval_n:
        print(
            f"WARNING: ImageNet int8 top-1 {correct}/{eval_n} "
            "(expected near float accuracy with per-channel weight scales; "
            "check calibration samples / preprocessing if this stays low)"
        )

    spec = quantized_cnn_to_spec(arch, pack)
    write_nk(args.output, spec)
    print(
        f"Wrote {args.output} ({len(pack.weight_tensors)} weight tensors, "
        f"{args.output.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
