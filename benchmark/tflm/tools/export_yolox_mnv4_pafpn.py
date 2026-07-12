#!/usr/bin/env python3
"""Export YOLOX MNv4-PAFPN float32 TFLite (320²) for host A/B.

Loads the same MiniDetector checkpoint used to pack models/yolox_mnv4_pafpn_trained.nk,
wraps multi-scale heads into one flat concat (NHWC decode layout), exports ONNX →
onnx2tf → float32 TFLite under benchmark/tflm/generated/.

Requires: torch, timm, onnx, onnx2tf (benchmark/tflm/.venv recommended).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
GENERATED = Path(__file__).resolve().parents[1] / "generated"
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

from train_yolox_mnv4_pafpn_mini import MiniDetector  # noqa: E402


class FlatDetector(nn.Module):
    """MiniDetector → single flat concat matching netkit PAFPN output layout."""

    def __init__(self, det: MiniDetector):
        super().__init__()
        self.det = det

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: NCHW; preds_to_flat expects list of 1,C,H,W
        preds = self.det(x)
        parts = []
        for p in preds:
            # N,C,H,W → N,H,W,C → flat per batch
            arr = p.permute(0, 2, 3, 1).contiguous()
            parts.append(arr.reshape(arr.shape[0], -1))
        return torch.cat(parts, dim=1)


def _onnx2tf_bin() -> str:
    found = shutil.which("onnx2tf")
    if found:
        return found
    venv_bin = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "onnx2tf"
    if venv_bin.is_file():
        return str(venv_bin)
    raise SystemExit(
        "onnx2tf not found. Install with:\n"
        "  python3 -m venv benchmark/tflm/.venv && "
        "benchmark/tflm/.venv/bin/pip install onnx onnx2tf"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "yolox_mnv4_pafpn_coco_train_50k_ema_v2.pt",
    )
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=GENERATED / "yolox_mnv4_pafpn_320.tflite",
    )
    parser.add_argument(
        "--saved-model-dir",
        type=Path,
        default=GENERATED / "yolox_mnv4_pafpn_320_saved_model",
    )
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise SystemExit(f"missing checkpoint {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    det = MiniDetector(hidden=args.hidden, freeze_backbone=True, pretrained=False)
    det.load_state_dict(ckpt["state_dict"], strict=False)
    det.eval()
    model = FlatDetector(det)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yolox_onnx_") as tmp:
        onnx_path = Path(tmp) / "yolox_mnv4_pafpn.onnx"
        dummy = torch.randn(1, 3, args.height, args.width)
        print(
            f"exporting ONNX {onnx_path.name} "
            f"({args.height}x{args.width}, hidden={args.hidden}) ..."
        )
        try:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["detections"],
                opset_version=17,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["detections"],
                opset_version=17,
            )

        if args.saved_model_dir.exists():
            shutil.rmtree(args.saved_model_dir)
        print(f"converting with onnx2tf -> {args.saved_model_dir} ...")
        subprocess.run(
            [_onnx2tf_bin(), "-i", str(onnx_path), "-o", str(args.saved_model_dir), "-nuo"],
            check=True,
        )

    candidates = sorted(args.saved_model_dir.glob("*float32.tflite"))
    if not candidates:
        raise SystemExit(f"onnx2tf did not emit a float32 .tflite under {args.saved_model_dir}")
    shutil.copy2(candidates[0], args.out)
    print(f"Wrote {args.out} ({args.out.stat().st_size} bytes)")

    # Quick shape check vs torch
    with torch.no_grad():
        torch_out = model(dummy).numpy().reshape(-1)
    try:
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=str(args.out))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        print("input:", inp["shape"], inp["dtype"])
        print("output:", out["shape"], "elems", int(np.prod(out["shape"])))
        print("torch flat elems", torch_out.size)
    except BaseException as exc:  # noqa: BLE001 — LiteRT/TF can abort oddly on some hosts
        print(f"(skip tflite inspect: {exc})")


if __name__ == "__main__":
    main()
