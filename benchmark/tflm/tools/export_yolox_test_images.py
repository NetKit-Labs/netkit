#!/usr/bin/env python3
"""Export 10 letterboxed 320x320x3 float32 samples for YOLOX host benches.

Uses COCO val2017 images (local data/coco_val2017) with the same letterbox+/255
preprocess as train_yolox_mnv4_pafpn_mini / sanity_yolox_trained_host.
Emits C arrays for netkit and a numpy cache for the TF Lite peer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GENERATED = Path(__file__).resolve().parents[1] / "generated"
CACHE = GENERATED / "yolox_sample_cache"

sys.path.insert(0, str(ROOT / "tools"))
from train_yolox_mnv4_pafpn_mini import (  # noqa: E402
    ensure_coco_val,
    letterbox,
    list_coco_val_samples,
)


def _c_float_array(name: str, values: np.ndarray) -> str:
    flat = values.reshape(-1)
    # Always emit a decimal so 0 becomes "0.0f" (bare "0f" is invalid C).
    body = ",\n".join(
        ", ".join(f"{float(v):.8f}f" for v in flat[i : i + 8]) for i in range(0, flat.size, 8)
    )
    return f"alignas(16) const float {name}[] = {{\n{body}\n}};\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out-h", type=Path, default=GENERATED / "yolox_test_images.h")
    parser.add_argument("--out-cc", type=Path, default=GENERATED / "yolox_test_images.cc")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args()

    from PIL import Image

    val_root = ensure_coco_val(args.data)
    pairs = list_coco_val_samples(val_root, max(args.count, 200))[-args.count :]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    samples: list[tuple[str, np.ndarray]] = []
    for img_path, _lab in pairs:
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        canvas, _, _, _ = letterbox(rgb, args.size)
        pixels = (np.asarray(canvas, dtype=np.float32) / 255.0).astype(np.float32)
        short = img_path.stem
        np.save(args.cache_dir / f"{short}.npy", pixels)
        # Keep a JPEG copy for TF Lite peer convenience
        Image.fromarray(canvas).save(args.cache_dir / f"{short}.jpg", quality=95)
        samples.append((short, pixels))

    n = len(samples)
    input_size = args.size * args.size * 3
    hdr = "\n".join(
        [
            "#pragma once",
            "",
            "#include <cstdint>",
            "",
            f"constexpr int kYoloxBenchmarkImageCount = {n};",
            f"constexpr int kYoloxBenchmarkInputSize = {input_size};",
            f"constexpr int kYoloxBenchmarkHeight = {args.size};",
            f"constexpr int kYoloxBenchmarkWidth = {args.size};",
            "constexpr int kYoloxBenchmarkChannels = 3;",
            # 80-class PAFPN flat: 85 * (40^2 + 20^2 + 10^2) at 320
            "constexpr int kYoloxBenchmarkOutputElems = 178500;",
            "",
            "struct YoloxBenchmarkImage {",
            "    const char* name;",
            "    const float* pixels;",
            "};",
            "",
            "extern const YoloxBenchmarkImage kYoloxBenchmarkImages[kYoloxBenchmarkImageCount];",
            "",
        ]
    )
    for i, (short, _) in enumerate(samples):
        hdr += f"extern const float kYoloxBenchPixels{i}[];\n"
    hdr += "\n"

    cc_parts = [
        '#include "yolox_test_images.h"',
        "",
    ]
    for i, (short, pixels) in enumerate(samples):
        cc_parts.append(_c_float_array(f"kYoloxBenchPixels{i}", pixels))
        cc_parts.append("")

    cc_parts.append(
        "const YoloxBenchmarkImage kYoloxBenchmarkImages[kYoloxBenchmarkImageCount] = {"
    )
    for i, (short, _) in enumerate(samples):
        cc_parts.append(f'    {{"{short}", kYoloxBenchPixels{i}}},')
    cc_parts.append("};")
    cc_parts.append("")

    args.out_h.write_text(hdr)
    args.out_cc.write_text("\n".join(cc_parts))
    print(f"wrote {args.out_h}")
    print(f"wrote {args.out_cc} ({n} images @ {args.size}^2)")


if __name__ == "__main__":
    main()
