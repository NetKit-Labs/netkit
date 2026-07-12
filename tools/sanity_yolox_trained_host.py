#!/usr/bin/env python3
"""Host sanity: packed YOLOX .nk (C++ nk_infer) vs torch EMA on val images.

Decodes with class-aware NMS and writes demo JPGs under models/checkpoints/.
Optionally also runs the slow Python reference_forward path (--with-py-ref).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
from PIL import Image, ImageDraw

from netkit.reader import read_nk
from netkit.runtime_infer import nk_infer_bin
from netkit.yolox_decode import decode_yolox_output
from train_yolox_mnv4_pafpn_mini import (
    MiniDetector,
    NUM_CLASSES,
    boxes_to_xyxy_pixels,
    ensure_coco_val,
    letterbox,
    list_coco_val_samples,
    load_yolo_label,
    preds_to_flat,
)


def _run_nk_bin(nk_path: Path, flat: np.ndarray) -> np.ndarray:
    bin_path = nk_infer_bin(ROOT)
    if not bin_path.is_file():
        raise FileNotFoundError(f"missing {bin_path}; run: make tools/nk_infer")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_bin = td_path / "in.bin"
        out_bin = td_path / "out.bin"
        flat.astype(np.float32).tofile(in_bin)
        proc = subprocess.run(
            [str(bin_path), str(nk_path), f"@{in_bin}", "--out-bin", str(out_bin)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"nk_infer failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
            )
        return np.fromfile(out_bin, dtype=np.float32)


def _draw(rgb: np.ndarray, dets, path: Path, title: str, max_dets: int = 15) -> None:
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    draw.text((4, 4), title, fill=(255, 255, 0))
    for d in sorted(dets, key=lambda x: -x.score)[:max_dets]:
        draw.rectangle([d.x1, d.y1, d.x2, d.y2], outline=(0, 255, 0), width=2)
        draw.text((d.x1, max(0, d.y1 - 12)), f"{d.class_id}:{d.score:.2f}", fill=(0, 255, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nk",
        type=Path,
        default=ROOT / "models" / "yolox_mnv4_pafpn_trained.nk",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "yolox_mnv4_pafpn_coco_train_50k_ema.pt",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument(
        "--max-abs",
        type=float,
        default=1e-3,
        help="Fail if max|cpp-torch| exceeds this (packing/runtime parity)",
    )
    parser.add_argument(
        "--with-py-ref",
        action="store_true",
        help="Also run Python forward_cnn (slow; overflow-prone on large SiLU nets)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "models" / "checkpoints" / "yolox_host_sanity",
    )
    args = parser.parse_args()

    arch, _weights = read_nk(args.nk)
    h, w, _c = arch["input"]
    assert (h, w) == (args.size, args.size), f"nk input {h}x{w} != --size {args.size}"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hidden = int(ckpt.get("hidden", 64))
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model = MiniDetector(hidden=hidden, freeze_backbone=True).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    val_root = ensure_coco_val(args.data)
    pairs = list_coco_val_samples(val_root, max(args.images, 200))[-args.images :]

    summary = []
    for img_path, lab_path in pairs:
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        oh, ow = rgb.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(rgb, args.size)
        flat = (np.asarray(canvas, dtype=np.float32) / 255.0).reshape(-1)

        torch_tensor = (
            torch.from_numpy(np.array(canvas, copy=True))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(device)
            / 255.0
        )
        with torch.no_grad():
            torch_flat = preds_to_flat(model(torch_tensor)).reshape(-1).astype(np.float32)

        cpp_flat = _run_nk_bin(args.nk, flat)
        max_cpp = float(np.max(np.abs(cpp_flat - torch_flat)))

        py_flat = None
        max_py = None
        max_cpp_py = None
        if args.with_py_ref:
            from netkit.reference_forward import forward_cnn

            arch_w, weights = read_nk(args.nk)
            py_flat = np.asarray(forward_cnn(flat, arch_w, weights), dtype=np.float32).reshape(-1)
            max_py = float(np.max(np.abs(py_flat - torch_flat)))
            max_cpp_py = float(np.max(np.abs(cpp_flat - py_flat)))

        def dets(arr: np.ndarray):
            return decode_yolox_output(
                arr,
                num_classes=NUM_CLASSES,
                score_threshold=args.score_thr,
                input_height=args.size,
                input_width=args.size,
                nms_iou_threshold=args.nms_iou,
            )

        torch_d = dets(torch_flat)
        cpp_d = dets(cpp_flat)
        py_d = dets(py_flat) if py_flat is not None else []

        gt = boxes_to_xyxy_pixels(
            load_yolo_label(lab_path),
            size=args.size,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            orig_w=ow,
            orig_h=oh,
        )

        stem = img_path.stem
        _draw(canvas, torch_d, args.out_dir / f"{stem}_torch.jpg", f"torch n={len(torch_d)}")
        _draw(canvas, cpp_d, args.out_dir / f"{stem}_cpp_nk.jpg", f"cpp.nk n={len(cpp_d)}")
        if py_flat is not None:
            _draw(canvas, py_d, args.out_dir / f"{stem}_py_nk.jpg", f"py.nk n={len(py_d)}")

        row = {
            "image": img_path.name,
            "gt_boxes": len(gt),
            "torch_dets": len(torch_d),
            "cpp_nk_dets": len(cpp_d),
            "max_abs_cpp_vs_torch": max_cpp,
            "torch_top": [
                {"cls": d.class_id, "score": round(d.score, 3)} for d in torch_d[:5]
            ],
            "cpp_top": [
                {"cls": d.class_id, "score": round(d.score, 3)} for d in cpp_d[:5]
            ],
        }
        if max_py is not None:
            row["py_nk_dets"] = len(py_d)
            row["max_abs_py_vs_torch"] = max_py
            row["max_abs_cpp_vs_py"] = max_cpp_py
        summary.append(row)
        print(json.dumps(row, indent=2))

    out_json = args.out_dir / "summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_json}")

    for r in summary:
        if not np.isfinite(r["max_abs_cpp_vs_torch"]):
            raise SystemExit(f"non-finite cpp vs torch on {r['image']}")
        if r["max_abs_cpp_vs_torch"] > args.max_abs:
            raise SystemExit(
                f"cpp vs torch max|diff|={r['max_abs_cpp_vs_torch']:.4g} "
                f"> {args.max_abs} on {r['image']}"
            )
        if r["torch_dets"] > 0 and r["cpp_nk_dets"] == 0:
            raise SystemExit(
                f"cpp produced 0 dets while torch had {r['torch_dets']} on {r['image']}"
            )
    print(
        f"OK: C++ nk_infer matches torch within {args.max_abs} abs; decode+NMS looks sane"
    )


if __name__ == "__main__":
    main()
