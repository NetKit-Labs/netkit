#!/usr/bin/env python3
"""Host ONNX Runtime float32 MobileNetV4-Conv-Small ImageNet bench.

Pairs with TF Lite / netkit ImageNet host benches (same 10 images, same loops).
Model: models/mobilenetv4_imagenet_f32.onnx (NCHW) — export via
  python3 benchmark/tools/export_host_onnx_assets.py --float32

Requires LiteRT-matched ORT build with XNNPACK EP.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmark" / "tflm" / "tools"
DEFAULT_MODEL = (
    ROOT / "benchmark" / "onnxruntime" / "models" / "mobilenetv4_imagenet_f32.onnx"
)
CACHE = ROOT / "benchmark" / "tflm" / "generated" / "imagenet_sample_cache"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TOOLS))
from export_imagenet_mnv4_test_images import SAMPLES, preprocess  # noqa: E402
from session_util import backend_name, input_meta, make_session, output_name  # noqa: E402


def _load_images_nchw() -> list[tuple[str, int, np.ndarray]]:
    from PIL import Image

    out = []
    for filename, _url, label, short in SAMPLES:
        path = CACHE / filename
        if not path.is_file():
            raise SystemExit(
                f"missing {path} — run: make -C benchmark/tflm export-imagenet-mnv4-images"
            )
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        nhwc = np.asarray(preprocess(rgb), dtype=np.float32)  # HWC
        if nhwc.ndim == 3:
            nhwc = nhwc[None, ...]  # NHWC
        nchw = np.transpose(nhwc, (0, 3, 1, 2))
        out.append((short, label, nchw))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--no-xnnpack", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(
            f"missing {args.model} — run: "
            "python3 benchmark/tools/export_host_onnx_assets.py --float32"
        )

    use_xnnpack = not args.no_xnnpack
    backend = backend_name(use_xnnpack)
    images = _load_images_nchw()
    num_images = len(images)

    session = make_session(
        args.model, num_threads=args.num_threads, use_xnnpack=use_xnnpack
    )
    in_name, in_shape, in_dtype = input_meta(session)
    out_name = output_name(session)
    if in_dtype != np.float32:
        raise SystemExit(f"expected float32 input, got {in_dtype}")

    print("ONNX Runtime MobileNetV4 ImageNet float32 benchmark")
    print("  runtime:     ONNX Runtime (host)")
    print(f"  backend:     {backend}")
    print(f"  providers:   {session.get_providers()}")
    print(f"  threads:     {args.num_threads}")
    print("  dtype:       float32")
    print(f"  model:       {args.model}")
    print(f"  model bytes: {args.model.stat().st_size}")
    print(f"  input:       {in_name} {in_shape}")
    print(
        f"  method:      {num_images} images x {args.loops} loops = "
        f"{num_images * args.loops} invokes (all timed)"
    )

    samples: list[float] = []
    correct = 0
    for loop in range(args.loops):
        for i, (short, label, batch) in enumerate(images):
            feeds = {in_name: batch.reshape(in_shape)}
            t0 = time.perf_counter()
            outs = session.run([out_name], feeds)
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1e6)
            logits = np.asarray(outs[0]).reshape(-1)
            pred = int(logits.argmax())
            if loop == 0:
                ok = pred == label
                correct += int(ok)
                print(
                    f"  image {i} ImageNet {short} (class {label})".ljust(48)
                    + f" label={label:4d} pred={pred:4d} {'OK' if ok else 'MISS'}"
                )

    cold_us = samples[0]
    first_pass_mean = statistics.fmean(samples[:num_images])
    warm = samples[num_images:]
    warm_mean = statistics.fmean(warm)
    warm_median = statistics.median(warm)
    warm_min = min(warm)
    warm_max = max(warm)
    warm_std = statistics.pstdev(warm)
    top1 = 100.0 * correct / num_images

    print()
    print(f"ONNX Runtime MobileNetV4 ImageNet summary ({backend})")
    print(f"  top-1 accuracy:   {correct} / {num_images}  ({top1:.1f}%)")
    print(
        f"  10-image mean:    {first_pass_mean:9.3f} us "
        f"({first_pass_mean / 1000.0:7.3f} ms)"
    )
    print(f"  cold invoke:      {cold_us:9.3f} us ({cold_us / 1000.0:7.3f} ms)")
    print(f"  warm median:      {warm_median:9.3f} us ({warm_median / 1000.0:7.3f} ms)")
    print(
        f"  warm mean:        {warm_mean:9.3f} us ({warm_mean / 1000.0:7.3f} ms)"
        f"  over {len(warm)} invokes"
    )
    print(f"  warm min/max:     {warm_min:9.3f} / {warm_max:.3f} us")
    print(f"  warm stddev:      {warm_std:9.3f} us")
    print(
        "BENCHMARK_SUMMARY runtime=onnxruntime model=mobilenetv4_imagenet "
        f"dtype=float32 backend={backend} threads={args.num_threads} "
        f"top1_correct={correct} top1_total={num_images} top1_pct={top1:.1f} "
        f"ten_image_mean_us={first_pass_mean:.3f} warm_median_us={warm_median:.3f} "
        f"warm_mean_us={warm_mean:.3f} cold_us={cold_us:.3f} invokes={len(samples)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
